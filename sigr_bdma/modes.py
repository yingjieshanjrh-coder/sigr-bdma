"""Degradation modes from posterior samples, single comparisons and RPT trajectories."""
from __future__ import annotations

import numpy as np

MODE_NAMES = ["LAM_Gr", "LAM_Si", "LAM_NE", "LAM_P", "LLI"]


def modes_from_capacities(cap_ref, cap):
    """cap arrays (..., 4) ordered Q_Gr, Q_Si, Q_P, Q_Li. Returns (..., 5) fractional losses."""
    cap_ref, cap = np.asarray(cap_ref, float), np.asarray(cap, float)
    ne_ref = cap_ref[..., 0] + cap_ref[..., 1]
    ne = cap[..., 0] + cap[..., 1]
    return np.stack([1 - cap[..., 0] / cap_ref[..., 0],
                     1 - cap[..., 1] / cap_ref[..., 1],
                     1 - ne / ne_ref,
                     1 - cap[..., 2] / cap_ref[..., 2],
                     1 - cap[..., 3] / cap_ref[..., 3]], axis=-1)


def paired_modes(post_ref, post, seed=0):
    """Mode samples from two independent posteriors (reference state, aged state)."""
    rng = np.random.default_rng(seed)
    a, b = post_ref.capacities(), post.capacities()
    n = min(len(a), len(b))
    return modes_from_capacities(a[rng.permutation(len(a))[:n]], b[rng.permutation(len(b))[:n]])


def summarize_modes(samples, truth=None, pct=True):
    k = 100 if pct else 1
    lines = [f"{'mode':7s} {'mean':>8s} {'sd':>7s} {'2.5%':>8s} {'97.5%':>8s}"
             + ("  truth" if truth is not None else "")]
    for i, n in enumerate(MODE_NAMES):
        s = samples[:, i] * k
        lo, hi = np.percentile(s, [2.5, 97.5])
        t = f"  {truth[i] * k:6.1f}" if truth is not None else ""
        lines.append(f"{n:7s} {s.mean():8.2f} {s.std():7.2f} {lo:8.2f} {hi:8.2f}{t}")
    return "\n".join(lines)


def trajectory(posteriors, seed=0):
    """List of Posterior objects (one per RPT, first = reference).
    Returns dict mode -> array (n_rpt, 3) with [median, 2.5%, 97.5%]."""
    out = {m: [] for m in MODE_NAMES}
    for post in posteriors:
        s = paired_modes(posteriors[0], post, seed)
        for i, m in enumerate(MODE_NAMES):
            out[m].append(np.percentile(s[:, i], [50, 2.5, 97.5]))
    return {m: np.array(v) for m, v in out.items()}
