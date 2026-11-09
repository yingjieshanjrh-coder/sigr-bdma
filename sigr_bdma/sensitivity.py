"""OCP-sensitivity tools: fit a cell with every combination of candidate component OCPs,
summarise the spread of each degradation mode, and profile a single mode.

Typical use::

    from sigr_bdma.sensitivity import ocp_ensemble, ensemble_summary
    res = ocp_ensemble(graphites, silicons, fresh, aged, R_fresh, R_aged)
    print(ensemble_summary(res))

`graphites` maps a name to a graphite OCP callable U(x), `silicons` maps a name to a
`SiOCP`. `fresh` / `aged` are `Branch` objects of the same direction.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from .cell import Branch, Cell
from .inference import CAPACITY_NAMES, Param, Problem, default_params
from .modes import MODE_NAMES, modes_from_capacities
from .ocp import NMC811_RANGE, U_nmc811

DEFAULT_STARTS = [dict(Q_Gr=qg, Q_Si=qs, Q_P=qp, Q_Li=ql)
                  for qg, qs in ((4.3, 0.6), (3.8, 1.0))
                  for qp in (5.6, 6.6, 7.7) for ql in (5.2, 6.2, 7.0)]


def scaled_starts(nominal_ah, base=None):
    """Rescale the default starting points (written for a 5 Ah cell) to another cell size."""
    k = nominal_ah / 5.0
    return [{n: v * k for n, v in g.items()} for g in (base or DEFAULT_STARTS)]


def crop_branch(br: Branch, v_lo=None, v_hi=None) -> Branch:
    """Restrict a branch to v_lo <= V <= v_hi (the model anchors at the new end points)."""
    v_lo = br.V[0] if v_lo is None else max(v_lo, br.V[0])
    v_hi = br.V[-1] if v_hi is None else min(v_hi, br.V[-1])
    q_lo, q_hi = np.interp(v_lo, br.V, br.Q), np.interp(v_hi, br.V, br.Q)
    keep = (br.Q > q_lo) & (br.Q < q_hi)
    Q = np.concatenate([[q_lo], br.Q[keep], [q_hi]])
    V = np.concatenate([[v_lo], br.V[keep], [v_hi]])
    return Branch(br.direction, Q - q_lo, V, br.I, br.label)


def _rms(prob, z):
    rr = prob._branch_residuals(z)
    if any(r is None for r in rr):
        return np.inf
    return float(np.sqrt(np.mean(np.concatenate([r[0] for r in rr]) ** 2)))


def best_fit(cell, branches, R, starts):
    """Multi-start MAP fit with R fixed. Returns (rms_V, capacities[4]) or None."""
    branches = branches if isinstance(branches, (list, tuple)) else [branches]
    best = None
    for g in starts:
        prob = Problem(cell, branches, default_params(g, fit_R=False), fixed={"R": R})
        try:
            z = prob.fit_map()
        except Exception:
            continue
        rms = _rms(prob, z)
        if np.isfinite(rms) and (best is None or rms < best[0]):
            best = (rms, np.asarray(z[:4], float))
    return best


def fit_pair(cell, fresh, aged, R_fresh, R_aged, starts=None):
    """Fresh fit from `starts`, aged fit started from the fresh optimum (and scaled copies)."""
    b0 = best_fit(cell, fresh, R_fresh, starts or DEFAULT_STARTS)
    if b0 is None:
        return None
    s1 = [dict(zip(CAPACITY_NAMES, b0[1] * f)) for f in (1.0, 0.92, 0.85)]
    b1 = best_fit(cell, aged, R_aged, s1)
    if b1 is None:
        return None
    return b0, b1


def ocp_ensemble(graphites, silicons, fresh, aged, R_fresh, R_aged, U_pos=U_nmc811,
                 pos_range=NMC811_RANGE, starts=None, verbose=False):
    """Fit every graphite x silicon combination. Returns one row per combination with the
    fit RMS (mV) at both states, the fresh Si capacity share and the five modes (%)."""
    rows = []
    for (gn, gf), (sn, si) in itertools.product(graphites.items(), silicons.items()):
        cell = Cell(si, U_pos=U_pos, pos_range=pos_range, U_gr=gf)
        out = fit_pair(cell, fresh, aged, R_fresh, R_aged, starts)
        if out is None:
            continue
        (r0, z0), (r1, z1) = out
        m = modes_from_capacities(z0, z1)
        row = dict(graphite=gn, silicon=sn, rms0_mV=1e3 * r0, rms1_mV=1e3 * r1,
                   Si_share0=z0[1] / (z0[0] + z0[1]))
        row.update({n: 100 * v for n, v in zip(MODE_NAMES, m)})
        row.update({f"{n}_0": v for n, v in zip(CAPACITY_NAMES, z0)})
        row.update({f"{n}_1": v for n, v in zip(CAPACITY_NAMES, z1)})
        rows.append(row)
        if verbose:
            print(f"{gn:12s} {sn:14s} rms {row['rms0_mV']:5.1f}/{row['rms1_mV']:5.1f} mV  "
                  + "  ".join(f"{n}={row[n]:5.1f}" for n in MODE_NAMES))
    return pd.DataFrame(rows)


def plausible(res, rms_max=10.0, min_mode=-2.0, max_si=95.0):
    """Combinations with mean RMS <= rms_max and no degenerate mode."""
    r = res.assign(rms=0.5 * (res.rms0_mV + res.rms1_mV))
    ok = ((r[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > min_mode) & (r.LAM_Si < max_si)
          & (r.rms <= rms_max))
    return r[ok]


def spread_stats(values):
    """Range, inter-quartile range and standard deviation of a set of mode values (pp)."""
    v = np.asarray(values, float)
    if len(v) == 0:
        return dict(n=0, range=np.nan, iqr=np.nan, sd=np.nan)
    q1, q3 = np.percentile(v, [25, 75])
    return dict(n=len(v), range=float(v.max() - v.min()), iqr=float(q3 - q1),
                sd=float(v.std(ddof=1)) if len(v) > 1 else 0.0)


def rarefied_range(values, k, n_draw=2000, seed=0):
    """Expected max-min range of k combinations drawn without replacement: removes the
    dependence of the range on how many combinations happen to be plausible."""
    v = np.asarray(values, float)
    if len(v) < k:
        return np.nan
    rng = np.random.default_rng(seed)
    d = np.array([np.ptp(rng.choice(v, k, replace=False)) for _ in range(n_draw)])
    return float(d.mean())


def ensemble_summary(res, rms_max=10.0, k=None):
    """One row per mode: n, range, IQR, SD (and rarefied range to k combinations)."""
    p = plausible(res, rms_max)
    rows = []
    for n in MODE_NAMES:
        st = spread_stats(p[n])
        if k:
            st[f"range_k{k}"] = rarefied_range(p[n], k)
        rows.append(dict(mode=n, **st))
    return pd.DataFrame(rows)


def profile_capacity(cell, aged, R_aged, z_aged, name="Q_Gr", values=None, starts_scale=(1.0,)):
    """Profile of the aged-state fit RMS over one capacity: `name` is fixed at each value
    and the other three capacities are refitted. Returns DataFrame(value, rms_mV, caps...)."""
    z_aged = dict(zip(CAPACITY_NAMES, np.asarray(z_aged, float)))
    if values is None:
        values = z_aged[name] * np.linspace(0.6, 1.25, 27)
    others = [n for n in CAPACITY_NAMES if n != name]
    rows = []
    prev = None
    for v in values:
        best = None
        guesses = [prev] if prev is not None else []
        guesses += [{n: z_aged[n] * f for n in others} for f in starts_scale]
        for g in guesses:
            ps = [Param(n, g[n], 0.02 * g[n] + 1e-4, 3.0 * g[n]) for n in others]
            prob = Problem(cell, [aged], ps, fixed={"R": R_aged, name: float(v)})
            try:
                z = prob.fit_map()
            except Exception:
                continue
            rms = _rms(prob, z)
            if np.isfinite(rms) and (best is None or rms < best[0]):
                best = (rms, dict(zip(others, z[:3])))
        if best is None:
            continue
        prev = best[1]
        rows.append(dict(value=float(v), rms_mV=1e3 * best[0], **{name: float(v)}, **best[1]))
    return pd.DataFrame(rows)
