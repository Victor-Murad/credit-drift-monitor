"""Medição de recursos do processo (memória de pico)."""

from __future__ import annotations

import resource
import sys

_BYTES_POR_MB = 1024 * 1024
_KIB_POR_MB = 1024


def peak_memory_mb() -> float:
    """Pico de memória residente do processo, em MiB.

    `ru_maxrss` vem em bytes no macOS e em KiB no Linux.
    """
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / _BYTES_POR_MB
    return maxrss / _KIB_POR_MB
