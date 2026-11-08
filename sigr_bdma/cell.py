"""Full-cell forward model: NMC | (graphite + Si) with branch-aware Si OCP and ohmic term.

State/parameters (dict):
  Q_Gr, Q_Si : graphite / silicon capacity in the negative electrode (Ah)
  Q_P        : positive-electrode capacity, stoichiometry 0..1 (Ah)
  Q_Li       : cyclable lithium (Ah)
  R          : effective series resistance during the low-rate test (ohm)
  w          : weight between two candidate Si OCPs (only if two are supplied)
  hys        : hysteresis scale, 1 = full measured gap, 0 = mean curve

A measured low-rate branch is described in its own coordinate:
  Q = charge throughput counted from the LOW-voltage end of that branch,
  V = measured terminal voltage, I = |current| (A).
Anchoring each branch at its own low-voltage end avoids assumptions about
rest/relaxation between branches.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ocp import Inverse, NMC811_RANGE, U_graphite, U_nmc811, mix_si


@dataclass
class Branch:
    direction: str          # "charge" or "discharge"
    Q: np.ndarray           # Ah, from low-voltage end, ascending
    V: np.ndarray           # V, measured
    I: float = 0.0          # A, magnitude of the (constant) current
    label: str = ""

    @property
    def span(self):
        return float(self.Q[-1] - self.Q[0])


class Cell:
    UN_GRID = np.linspace(0.005, 1.5, 2500)   # negative-electrode potential sweep (V vs Li)

    def __init__(self, si_ocps, U_pos=U_nmc811, pos_range=NMC811_RANGE, U_gr=U_graphite,
                 n_inverse=1201):
        if not isinstance(si_ocps, (list, tuple)):
            si_ocps = [si_ocps]
        if len(si_ocps) > 2:
            raise ValueError("use at most two candidate Si OCPs (mixture weight w)")
        self.si_ocps = list(si_ocps)
        self.U_pos, self.pos_range = U_pos, pos_range
        self.x_gr = Inverse(U_gr)(self.UN_GRID)          # fixed, precompute
        self.n_inverse = n_inverse
        self._cache = {}

    # -- silicon lithiation at each grid potential -------------------------------
    def _x_si(self, direction, w, hys):
        key = (direction, round(w, 10), round(hys, 10))   # fine enough for finite differences
        if key not in self._cache:
            si = self.si_ocps[0] if len(self.si_ocps) == 1 else mix_si(*self.si_ocps, w)
            inv = Inverse(si.branch(direction, hys), n=self.n_inverse)
            if len(self._cache) > 4096:
                self._cache.clear()
            self._cache[key] = inv(self.UN_GRID)
        return self._cache[key]

    # -- curves -------------------------------------------------------------------
    def ocv_curve(self, p, direction):
        """(q_n, OCV, V_meas) ordered by increasing NE lithium."""
        w, hys, R = p.get("w", 0.0), p.get("hys", 1.0), p.get("R", 0.0)
        q_n = p["Q_Gr"] * self.x_gr + p["Q_Si"] * self._x_si(direction, w, hys)
        y = (p["Q_Li"] - q_n) / p["Q_P"]
        ocv = self.U_pos(np.clip(y, *self.pos_range)) - self.UN_GRID
        q_n, ocv = q_n[::-1], ocv[::-1]
        ocv = np.maximum.accumulate(ocv) + np.arange(len(ocv)) * 1e-9
        sign = 1.0 if direction == "charge" else -1.0
        return q_n, ocv, ocv + sign * p.get("_I", 0.0) * R

    def predict(self, p, br: Branch):
        """Model voltage at the branch's Q points, and the model's capacity span.
        Returns (None, None) if the model cannot reach the branch's voltage range."""
        q, _, V = self.ocv_curve({**p, "_I": br.I}, br.direction)
        v_lo, v_hi = br.V[0], br.V[-1]
        if V[0] > v_lo or V[-1] < v_hi:
            return None, None
        q_lo = np.interp(v_lo, V, q)
        q_hi = np.interp(v_hi, V, q)
        Vp = np.interp(br.Q - br.Q[0], q - q_lo, V, right=np.nan)
        return Vp, q_hi - q_lo

    def simulate_branch(self, p, direction, I=0.0, window=(2.5, 4.2), n=200, noise=0.0,
                        rng=None, label=""):
        """Synthetic measured branch between the voltage cut-offs `window`."""
        q, _, V = self.ocv_curve({**p, "_I": I}, direction)
        if V[0] > window[0] or V[-1] < window[1]:
            raise ValueError("cell does not span the requested window")
        q_lo, q_hi = np.interp(window[0], V, q), np.interp(window[1], V, q)
        Qs = np.linspace(0, q_hi - q_lo, n)
        Vs = np.interp(Qs, q - q_lo, V)
        if noise:
            rng = rng or np.random.default_rng()
            Vs = Vs + rng.normal(0, noise, n)
            Vs[0], Vs[-1] = window          # keep exact anchors at the cut-offs
        return Branch(direction, Qs, Vs, I, label)
