"""
tracker.py
----------
A dict-like container for logging scalar and tensor quantities during
optimisation, with built-in plotting and pickle persistence.
"""

import pickle
from collections.abc import MutableMapping
from typing import Any, Iterator, Tuple, List

import matplotlib.pyplot as plt
import torch


class Tracker(MutableMapping):
    """
    Ordered log of optimisation metrics.

    Behaves like a plain ``dict`` (supports ``[]``, ``in``, iteration, etc.)
    but adds:
    - ``add(key, val)``  — append a value to a keyed list (auto-detaches Tensors)
    - ``visualize(...)`` — plot logged series with matplotlib
    - ``save / load``    — pickle serialisation

    Example
    -------
    >>> tracker = Tracker()
    >>> for it in range(100):
    ...     tracker.add('loss', loss.item())
    >>> tracker.visualize(['loss'], filename='loss.png')
    """

    def __init__(self):
        """Initialise an empty Tracker with no logged keys."""
        self._results: dict = {}

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def add(self, key: str, val: Any) -> None:
        """
        Append `val` to the list stored under `key`.

        Tensors are automatically detached and moved to CPU as numpy arrays
        before storage so that the tracker never holds computation graphs.
        """
        if isinstance(val, torch.Tensor):
            val = val.detach().cpu().numpy()

        if key in self._results:
            self._results[key].append(val)
        else:
            self._results[key] = [val]

    # ------------------------------------------------------------------
    # MutableMapping abstract interface
    # ------------------------------------------------------------------

    def __getitem__(self, key: str) -> list:
        """Return the list of logged values for ``key``."""
        return self._results[key]

    def __setitem__(self, key: str, value: Any) -> None:
        """Overwrite the entry for ``key`` with ``value`` (replaces entire list)."""
        self._results[key] = value

    def __delitem__(self, key: str) -> None:
        """Remove ``key`` and its logged values from the tracker."""
        del self._results[key]

    def __iter__(self) -> Iterator:
        """Iterate over all logged keys in insertion order."""
        return iter(self._results)

    def __len__(self) -> int:
        """Return the number of distinct keys currently tracked."""
        return len(self._results)

    def __repr__(self) -> str:
        """Return a string representation showing all tracked data."""
        return f"{self.__class__.__name__}({self._results})"

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, filepath: str) -> None:
        """Pickle the Tracker to `filepath`."""
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "Tracker":
        """Load and return a Tracker from a pickle file."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def visualize(
        self,
        key_plot:  list,
        markers:   "list | None"   = None,
        labels:    "list | None"   = None,
        filename:  "str | None"    = None,
        figsize:   Tuple[int, int] = (6, 3),
        title:      "str | None" = None,
        ylim:       "List[float] | None" = None
    ) -> None:
        """
        Plot one or more logged series on a single figure.

        Parameters
        ----------
        key_plot : list of str — keys to plot (must exist in the tracker)
        markers  : optional list of matplotlib format strings, one per key
        labels   : optional legend labels; defaults to `key_plot`
        filename : save path (PNG); if None the figure is shown interactively
        figsize  : matplotlib figure size
        """
        labels = key_plot if labels is None else labels

        plt.figure(figsize=figsize)
        for ik, (k, label) in enumerate(zip(key_plot, labels)):
            if markers is not None:
                plt.plot(self._results[k], markers[ik], label=label)
            else:
                plt.plot(self._results[k], label=label)

        plt.legend(loc='best')
        plt.xlabel("iter")
        if title is not None:
            plt.title(title)
        if ylim is not None:
            plt.ylim(ylim)


        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
        plt.close()

        return None
