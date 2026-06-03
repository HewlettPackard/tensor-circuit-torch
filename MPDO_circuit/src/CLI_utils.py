"""
CLI_utils.py
------------
Lightweight helper for reading typed command-line arguments without
requiring a full argparse setup in each script.
"""

import argparse
from typing import Any, Type


def get_CLI_input(argument: str, type: Type = str, default: Any = None) -> Any:
    """
    Read a single named CLI argument, returning `default` if absent.

    Uses `parse_known_args` so that unrecognised flags from other scripts
    are silently ignored rather than raising an error.

    Parameters
    ----------
    argument : str
        Flag name including leading dashes, e.g. '--learning_rate'.
    type     : Type
        Python type to cast the parsed string to (e.g. float, int, str).
    default  : Any
        Value to return when the flag is not present on the command line.

    Returns
    -------
    Any
        The parsed (and cast) value, or `default`.

    Examples
    --------
    >>> lr = get_CLI_input('--lr', type=float, default=1e-3)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(argument, type=type, default=default)
    args, _ = parser.parse_known_args()

    # '--config-file' → 'config_file'
    arg_name = argument.lstrip('-').replace('-', '_')
    return getattr(args, arg_name)
