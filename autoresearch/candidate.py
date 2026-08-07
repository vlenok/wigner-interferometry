"""Agent-editable discrete Wigner transform candidate.

Research agents may edit this file only. The initial implementation delegates to
the frozen, NumPy-only baseline so its paired performance score should be close
to 1.0. Replace the wrapper with an experimental implementation during a run.
"""

import numpy as np

from ._reference import dwv as _baseline_dwv


def dwv(signal1: np.ndarray, signal2: np.ndarray) -> np.ndarray:
    """Return the cross discrete Wigner transform used by the notebooks."""
    return _baseline_dwv(signal1, signal2)
