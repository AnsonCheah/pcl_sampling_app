"""Array-module selection: NumPy on the CPU, CuPy on the GPU, with an explicit fallback.

CuPy implements NumPy's ``__array_function__`` protocol, so ``np.linalg.norm``,
``np.einsum``, ``np.sort``, ``np.bincount`` and friends dispatch to the GPU automatically when
handed CuPy arrays.  That is why the matcher is not written twice: only array *creation*
(``arange``, ``zeros``, ``empty``, ``tile``) has to know which module it is talking to,
because there is no array argument to dispatch on.  Everything else is shared code.

**Availability is probed, not assumed.**  ``import cupy`` succeeding proves very little: the
wheel installs fine on a machine with no GPU, with a driver too old for the runtime, or -- as
happened in this repo -- with four compiled ``.pyd`` files missing from a half-written
install, which surfaces as a bogus "circular import" error.  So the probe actually allocates
and reduces on the device once, and any failure at all means CPU.

A caller that asks for ``"cupy"`` and does not get it is **not** an error, but it is never
silent either: :class:`~.match.MatchResult` records the backend that actually ran, so a
benchmark cannot quietly report CPU timings as GPU ones.  That was the concern behind the
original hard failure on unimplemented backends, and it is preserved by reporting rather than
by refusing.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = ["resolve", "cupy_module", "cupy_available", "asnumpy", "BACKENDS"]

BACKENDS = ("numpy", "cupy", "auto")

# Tri-state: None = not yet probed, False = probed and unusable, module = probed and working.
_CUPY: Optional[object] = None


def cupy_module():
    """The CuPy module if it is genuinely usable on this machine, else ``None``.

    Probed once and cached -- the probe launches a kernel, so it is far too expensive to
    repeat per match, and its answer cannot change within a process.
    """
    global _CUPY
    if _CUPY is None:
        try:
            import cupy  # noqa: F401

            if cupy.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            # Actually round-trip through the device. An importable CuPy with a visible
            # device can still fail on the first real allocation (driver/runtime mismatch),
            # and discovering that inside `match` would be much worse than discovering it here.
            if int(cupy.arange(4).sum()) != 6:
                raise RuntimeError("device arithmetic returned the wrong answer")
            _CUPY = cupy
        except Exception:
            _CUPY = False
    return _CUPY or None


def cupy_available() -> bool:
    return cupy_module() is not None


def resolve(backend: str) -> Tuple[object, str]:
    """``(array_module, name_actually_used)``.

    ``"numpy"`` forces the CPU. ``"cupy"`` and ``"auto"`` both prefer the GPU and fall back to
    NumPy when it is unavailable; they differ only in intent, and the caller can tell what
    happened from the returned name. An unknown name is a programming error and raises --
    silently treating a typo as "numpy" would be the one genuinely misleading outcome.
    """
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; expected one of {BACKENDS}")
    if backend == "numpy":
        return np, "numpy"
    cp = cupy_module()
    return (cp, "cupy") if cp is not None else (np, "numpy")


def asnumpy(a):
    """Bring an array back to the host, whichever module produced it."""
    cp = _CUPY if _CUPY else None
    if cp is not None and isinstance(a, cp.ndarray):
        return cp.asnumpy(a)
    return np.asarray(a)
