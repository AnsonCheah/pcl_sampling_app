"""Code shared by the two PPF packages.

`_backend` (CuPy probing / array marshalling) and `_frames` (local reference frames and
the alpha angle) were byte-identical in `ppf/` and `ppf_saliency/`; `_frames` had already
drifted in its comments alone, with nothing to stop it forking for real.

This directory travels with `registration/ppf/`: copying the vanilla matcher into another
project means copying `ppf/` **and** `_shared/`. Both are reached by relative import, so
neither names the repository.
"""
