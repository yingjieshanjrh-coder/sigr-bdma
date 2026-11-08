"""Electrode open-circuit potentials (OCPs).

Graphite and NMC811: empirical fits of Chen et al., J. Electrochem. Soc. 167 (2020) 080534,
as implemented in PyBaMM (parameter set "Chen2020").

Silicon: a direction-dependent OCP (lithiation / delithiation branch). The built-in curves
are PLACEHOLDERS for testing only. For real work load measured branches with
`SiOCP.from_csv` or extract them from a blend-anode half-cell with `extract_si_from_blend`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Graphite / NMC811 (Chen 2020)
# ---------------------------------------------------------------------------
def U_graphite(x):
    x = np.asarray(x, float)
    return (1.9793 * np.exp(-39.3631 * x) + 0.2482
            - 0.0909 * np.tanh(29.8538 * (x - 0.1234))
            - 0.04478 * np.tanh(14.9159 * (x - 0.2769))
            - 0.0205 * np.tanh(30.4444 * (x - 0.6103)))


def U_nmc811(y):
    y = np.asarray(y, float)
    return (-0.8090 * y + 4.4875
            - 0.0428 * np.tanh(18.5138 * (y - 0.5542))
            - 17.7326 * np.tanh(15.7890 * (y - 0.3117))
            + 17.5842 * np.tanh(15.9308 * (y - 0.3120)))


NMC811_RANGE = (0.15, 1.0)


# ---------------------------------------------------------------------------
# Inverse of a monotonically decreasing OCP
# ---------------------------------------------------------------------------
class Inverse:
    """x(U) for a decreasing U(x) on [lo, hi]. Clamps outside the tabulated range."""

    def __init__(self, f: Callable, lo: float = 0.0, hi: float = 1.0, n: int = 2001):
        x = np.linspace(lo, hi, n)
        u = np.minimum.accumulate(np.asarray(f(x), float))
        u = u - np.arange(n) * 1e-10            # strictly decreasing
        self.u, self.x = u[::-1], x[::-1]       # ascending in U

    def __call__(self, U):
        return np.interp(U, self.u, self.x)


# ---------------------------------------------------------------------------
# Silicon OCP with hysteresis
# ---------------------------------------------------------------------------
@dataclass
class SiOCP:
    """Silicon OCP as two branches, x = 0 (empty) .. 1 (fully lithiated)."""
    name: str
    lith: Callable
    delith: Callable

    def mean(self, x):
        return 0.5 * (self.lith(x) + self.delith(x))

    def branch(self, direction: str, hys: float = 1.0) -> Callable:
        """direction 'charge' (cell charge = anode lithiation) or 'discharge'.
        hys scales the half-gap: 1 = full measured hysteresis, 0 = mean curve
        (same form as PyBaMM's hysteresis-state models, U = mean + h * half_gap)."""
        base = self.lith if direction == "charge" else self.delith
        return lambda x: self.mean(x) + hys * (base(x) - self.mean(x))

    @classmethod
    def from_csv(cls, name, path, x_col="x", lith_col="U_lith", delith_col="U_delith"):
        """CSV with a common x grid and both branches."""
        d = np.genfromtxt(path, delimiter=",", names=True)
        xs = d[x_col]
        order = np.argsort(xs)
        xs = xs[order]
        ul, ud = d[lith_col][order], d[delith_col][order]
        return cls(name, lambda x: np.interp(x, xs, ul), lambda x: np.interp(x, xs, ud))

    @classmethod
    def from_branch_arrays(cls, name, x_lith, U_lith, x_delith, U_delith):
        o1, o2 = np.argsort(x_lith), np.argsort(x_delith)
        xl, ul = np.asarray(x_lith)[o1], np.asarray(U_lith)[o1]
        xd, ud = np.asarray(x_delith)[o2], np.asarray(U_delith)[o2]
        return cls(name, lambda x: np.interp(x, xl, ul), lambda x: np.interp(x, xd, ud))


def placeholder_si(name="placeholder-A", shift=0.0, gap_scale=1.0) -> SiOCP:
    """Smooth monotone test curves (~0.06 V gap when empty, ~0.2 V mid, ~0.16 V full).
    NOT measured data."""
    def lith(x):
        x = np.clip(x, 0, 1)
        return 0.04 + 0.25 * (1 - x) ** 1.5 + 0.55 * (1 - x) ** 8 + shift

    def delith0(x):
        x = np.clip(x, 0, 1)
        return 0.20 + 0.30 * (1 - x) ** 1.2 + 0.40 * (1 - x) ** 8 + shift

    def delith(x):
        return lith(x) + gap_scale * (delith0(x) - lith(x))

    return SiOCP(name, lith, delith)


def mix_si(a: SiOCP, b: SiOCP, w: float) -> SiOCP:
    """Pointwise interpolation between two candidate Si OCPs (w=0 -> a, w=1 -> b).
    Used to put the (large, literature-documented) uncertainty of the Si OCP itself
    into the posterior instead of fixing one curve."""
    return SiOCP(f"{a.name}*{1 - w:.2f}+{b.name}*{w:.2f}",
                 lambda x: (1 - w) * a.lith(x) + w * b.lith(x),
                 lambda x: (1 - w) * a.delith(x) + w * b.delith(x))


# ---------------------------------------------------------------------------
# Si OCP from a blend-anode half-cell (idea: Rehm et al., J. Power Sources 2026)
# ---------------------------------------------------------------------------
def _pav_decreasing(y, w=None):
    """Pool-adjacent-violators: least-squares non-increasing fit to y."""
    y = np.asarray(y, float)
    w = np.ones_like(y) if w is None else np.asarray(w, float)
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] < vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; cnt[-2] += cnt[-1]; vals[-2] = v
            vals.pop(); wts.pop(); cnt.pop()
    return np.repeat(vals, cnt)


def extract_si_from_blend(U_blend_lith, q_blend_lith, U_blend_delith, q_blend_delith,
                          si_fraction, U_gr=U_graphite, max_gr_slope=4.0,
                          name="Si-from-blend"):
    """Derive Si lithiation/delithiation branches from blend half-cell data.

    q_blend_*: normalised blend lithiation degree (0 empty .. 1 full) at potentials U_blend_*.
    Blend balance at a shared potential U:
        q_blend(U) = (1 - g) * x_gr(U) + g * x_si(U)  ->  x_si(U) = (q_blend - (1-g) x_gr) / g
    Points on graphite plateaus (|dx_gr/dU| > max_gr_slope per V) are ill-conditioned
    (errors amplified by (1-g)/g * slope) and are dropped; the rest is made monotone by
    isotonic regression. Graphite is taken as hysteresis-free. Returns a SiOCP.
    """
    g = si_fraction
    xgr = Inverse(U_gr)
    out = []
    for U, q in ((U_blend_lith, q_blend_lith), (U_blend_delith, q_blend_delith)):
        U, q = np.asarray(U, float), np.asarray(q, float)
        o = np.argsort(U)
        U, q = U[o], q[o]
        xg = xgr(U)
        slope = np.abs(np.gradient(xg, U))
        keep = slope < max_gr_slope
        xs = (q[keep] - (1 - g) * xg[keep]) / g
        xs = np.clip(_pav_decreasing(xs), 0, 1)     # x_si must decrease with U
        out.append((xs[::-1], U[keep][::-1]))       # ascending x
    (xl, ul), (xd, ud) = out
    return SiOCP.from_branch_arrays(name, xl, ul, xd, ud)
