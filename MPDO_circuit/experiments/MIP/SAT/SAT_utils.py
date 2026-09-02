import numpy as np
from typing import Tuple
import itertools
from typing import List, Tuple


import numpy as np
import sparse
import torch

from src.mpdo_torch import MPDOtorch


"""
Load a DIMACS CNF file into a list of clause tuples.

DIMACS CNF format:
    c ...                     comment lines (ignored)
    p cnf <n_vars> <n_clauses>   header line
    <lit> <lit> ... <lit> 0   one or more clauses, each terminated by 0
                              (a clause may be split across multiple lines)
    %                       end

Literals are 1-indexed signed ints: i = positive literal on var i,
-i = negated literal on var i. The trailing 0 is a clause delimiter,
not a literal, and must be dropped.
"""


def load_cnf(filename: str) -> Tuple[int, int, List[Tuple[int, ...]]]:
    """
    Parse a DIMACS .cnf file.

    Parameters
    ----------
    filename : str
        Path to a .cnf file (plain text, not gzipped -- decompress first
        if you're reading straight from a SATLIB .tar.gz).

    Returns
    -------
    n_vars : int
        Number of variables, from the 'p cnf' header line.
    n_clauses_declared : int
        Number of clauses declared in the header (sanity-check against
        len(clauses) if you want to catch malformed files).
    clauses : list of tuples of nonzero ints
        Each tuple is one clause, e.g. (-5, 41, 40).
    """
    n_vars = None
    n_clauses_declared = None
    clauses: List[Tuple[int, ...]] = []
    current: List[int] = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # check for end sign
            if '%' in line:
                break

            if not line or line.startswith("c"):
                continue  # skip blank lines and comments
            if line.startswith("p"):
                # header: p cnf <n_vars> <n_clauses>
                parts = line.split()
                n_vars = int(parts[2])
                n_clauses_declared = int(parts[3])
                continue

            for tok in line.split():
                lit = int(tok)
                if lit == 0:
                    # end of current clause
                    clauses.append(tuple(current))
                    current = []
                else:
                    current.append(lit)

    if current:
        # file didn't end with a trailing 0 on the last clause -- still keep it
        clauses.append(tuple(current))

    if n_vars is None:
        raise ValueError(f"no 'p cnf' header line found in {filename}")

    return n_vars, n_clauses_declared, clauses


def evaluate_clauses(rho: MPDOtorch, clauses: List[Tuple[int,...]]) -> torch.Tensor:

    device = rho.device
    dtype = rho.dtype

    # projection operators
    P0 = torch.tensor([[0., 0,], [0., 1.]], device=device, dtype=dtype)
    P1 = torch.tensor([[1., 0.], [0., 0.]], device=device, dtype=dtype)

    # loop over clauses, evaluate cumulant and add to result
    results = []
    for cl in clauses:

        ops = [P0 if l > 0 else P1 for l in cl]
        inds = [np.abs(l) - 1 for l in cl] # do not forget 0- vs 1-based indexing!

        results.append(rho.cumulant(ops=ops, sites=inds).real)

    return torch.stack(results)


def clauses_satisfied(bits: torch.Tensor, clauses: List[Tuple[int, ...]]) -> torch.Tensor:
    """
    bits: 1-D bool tensor, shape (n_vars,) -- a single variable assignment (0-indexed internally)
    clauses: list of DIMACS-style clauses, 1-indexed signed ints, e.g. (-5, 41, 40)

    Returns: bool tensor of shape (len(clauses),) -- True where that clause is satisfied.
    """
    literals = []
    for cl in clauses:
        lit_true = torch.stack([
            bits[abs(l) - 1] if l > 0 else ~bits[abs(l) - 1]
            for l in cl
        ])
        literals.append(lit_true.any())
    return torch.stack(literals)






