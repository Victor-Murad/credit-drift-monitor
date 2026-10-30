"""Medição de recursos do processo (memória de pico)."""

from __future__ import annotations

import sys

# `resource` só existe em Unix; no Windows o pico vem da psapi (ver abaixo).
if sys.platform != "win32":
    import resource

_BYTES_POR_MB = 1024 * 1024
_KIB_POR_MB = 1024


def peak_memory_mb() -> float:
    """Pico de memória residente do processo, em MiB.

    `ru_maxrss` vem em bytes no macOS e em KiB no Linux. No Windows, o
    equivalente é o `PeakWorkingSetSize`, em bytes.
    """
    if sys.platform == "win32":
        return _peak_working_set_bytes() / _BYTES_POR_MB
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return maxrss / _BYTES_POR_MB
    return maxrss / _KIB_POR_MB


def _peak_working_set_bytes() -> int:
    """`PeakWorkingSetSize` do processo atual (Windows, via psapi; só stdlib)."""
    import ctypes
    from ctypes import wintypes

    class _ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.WinDLL("kernel32")
    psapi = ctypes.WinDLL("psapi")
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.POINTER(_ProcessMemoryCounters),
        wintypes.DWORD,
    ]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    ok = psapi.GetProcessMemoryInfo(
        kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb
    )
    if not ok:
        raise ctypes.WinError()
    return int(counters.PeakWorkingSetSize)
