# from typing import Any
# import torch

# class Tracker:

#     """
#     The class to track the results during the course of the optimization.
#     """

#     def __init__(self):

#         # the dictionary to store the results
#         self.results = {}

#     def add(self, key: str, val: Any):

#         """Add a key and a value to the dictionary."""

#         # make sure val is in numpy format on cpu, else transfer
#         val  = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val

#         # append key to result
#         if key in self.results:
#             self.results[key].append(val)
#         else:
#             self.results[key] = [val]

#     def __getitem__(self, key: str):

#         """Get an item from the dictionary. (overloaded function)"""

#         return self.results[key]

from typing import Any, Iterator, Tuple
from collections.abc import MutableMapping
import torch
import pickle
import matplotlib.pyplot as plt


class Tracker(MutableMapping):
    """
    The class to track the results during the course of the optimization.
    Behaves like a dictionary.
    """

    def __init__(self):

        self._results = {}

    def add(self, key: str, val: Any):
        """Add a key and a value to the dictionary."""
        val = val.detach().cpu().numpy() if isinstance(val, torch.Tensor) else val

        if key in self._results:
            self._results[key].append(val)
        else:
            self._results[key] = [val]

    # --- Required abstract methods for MutableMapping ---

    def __getitem__(self, key: str):
        return self._results[key]

    def __setitem__(self, key: str, value: Any):
        self._results[key] = value

    def __delitem__(self, key: str):
        del self._results[key]

    def __iter__(self) -> Iterator:
        return iter(self._results)

    def __len__(self) -> int:
        return len(self._results)

    # --- Optional but nice ---

    def __repr__(self):
        return f"{self.__class__.__name__}({self._results})"
    
    # --- pickle helpers ---
    def save(self, filepath: str):
        """Save Tracker to a pickle file."""
        with open(filepath, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath: str) -> "Tracker":
        """Load Tracker from a pickle file."""
        with open(filepath, "rb") as f:
            return pickle.load(f)
        

    def visualize(self, key_plot, markers=None, labels=None, filename: str|None=None, figsize: Tuple[int]=(6,3)):

        labels = key_plot if labels is None else labels

        plt.figure(figsize=figsize)
        for ik, (k, label) in enumerate(zip(key_plot, labels)):
            if markers is not None:
                plt.plot(self._results[k], markers[ik], label=label) 
            else:
                plt.plot(self._results[k], label=k)            
        
        plt.legend(loc='lower left')
        plt.xlabel("iter")
        plt.tight_layout()

        if filename is None:
            plt.show()
        else:
            plt.savefig(filename)
            plt.close()

    

