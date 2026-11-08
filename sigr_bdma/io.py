"""Turning cycler data into `Branch` objects.

Two entry points:
  branch_from_arrays : you already have one constant-current segment (Q, V, I)
  split_cycler_log   : a raw log (time, current, voltage) -> low-rate CC segments
Cycler exports differ (column names, sign convention, units). Map your columns
explicitly; see examples/04_fit_real_rpt_template.py.
"""
from __future__ import annotations

import numpy as np

from .cell import Branch


def _bin_monotone(Q, V, n):
    """Downsample to n points on a uniform Q grid, after sorting by Q."""
    o = np.argsort(Q)
    Q, V = np.asarray(Q, float)[o], np.asarray(V, float)[o]
    grid = np.linspace(Q[0], Q[-1], n)
    return grid, np.interp(grid, Q, V)


def branch_from_arrays(Q, V, direction, I, n=200, label=""):
    """Q: charge throughput of the segment counted from its START (Ah, positive),
    V: voltage. For a discharge segment the start is the HIGH-voltage end, so the
    coordinate is flipped to count from the low-voltage end as `Branch` expects."""
    Q, V = np.asarray(Q, float), np.asarray(V, float)
    Q = np.abs(Q - Q[0])
    if direction == "discharge":
        Q = Q.max() - Q
    Qg, Vg = _bin_monotone(Q, V, n)
    Qg = Qg - Qg[0]
    Vg[0], Vg[-1] = V.min(), V.max()        # exact cut-off anchors
    Vg = np.maximum.accumulate(Vg)
    return Branch(direction, Qg, Vg, float(abs(I)), label)


def split_cycler_log(t, I, V, target_current, rel_tol=0.1, min_duration=600.0,
                     charge_positive=True, n=200):
    """Find constant-current segments with |I| ~ target_current.
    t in s, I in A, V in V. Returns list of Branch (in time order)."""
    t, I, V = (np.asarray(a, float) for a in (t, I, V))
    if not charge_positive:
        I = -I
    on = np.abs(np.abs(I) - target_current) <= rel_tol * target_current
    sgn = np.sign(I) * on
    out, start = [], None
    for k in range(1, len(t) + 1):
        edge = k == len(t) or sgn[k] != sgn[k - 1]
        if start is None and k - 1 < len(t) and sgn[k - 1] != 0:
            start = k - 1
        if edge and start is not None:
            seg = slice(start, k)
            if t[k - 1] - t[start] >= min_duration:
                Ii = I[seg]
                Q = np.concatenate([[0], np.cumsum(np.abs(Ii[1:]) * np.diff(t[seg]))]) / 3600
                d = "charge" if Ii.mean() > 0 else "discharge"
                out.append(branch_from_arrays(Q, V[seg], d, np.abs(Ii).mean(), n,
                                              label=f"{d}@{t[start]:.0f}s"))
            start = None
    return out
