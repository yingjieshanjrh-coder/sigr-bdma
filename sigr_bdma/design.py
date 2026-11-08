"""Experimental design for degradation-mode analysis (DMA).

Fisher-information estimate of the posterior uncertainty of each capacity for a given
measurement protocol (which branches, which voltage window, current, noise), evaluated
at an assumed true state. Answers questions like "is LAM_Si identifiable from a
3.0-4.2 V discharge?" before anyone runs a test.
"""
from __future__ import annotations

import numpy as np

from .cell import Cell
from .inference import CAPACITY_NAMES, Problem, default_params


def fisher_covariance(problem: Problem, x_true, noise=1e-3, rel_step=1e-4, prior_sd=None):
    """Gauss-Newton covariance of the parameters (log-sigma excluded)."""
    def whitened(x):
        r = []
        for br, rr in zip(problem.branches, problem._branch_residuals(x)):
            if rr is None:
                raise ValueError("model cannot reproduce the window at x_true")
            res, dq = rr
            r.append(np.append(res / noise, dq / problem.cap_sd))
        return np.concatenate(r)

    x_true = np.asarray(x_true, float)
    r0 = whitened(x_true)
    J = np.empty((len(r0), len(x_true)))
    for i in range(len(x_true)):
        h = rel_step * max(abs(x_true[i]), 1e-2)
        xp, xm = x_true.copy(), x_true.copy()
        xp[i] += h; xm[i] -= h
        J[:, i] = (whitened(xp) - whitened(xm)) / (2 * h)
    F = J.T @ J
    if prior_sd is not None:
        F += np.diag(1.0 / np.asarray(prior_sd) ** 2)
    return np.linalg.pinv(F)


def evaluate_design(si_ocp, truth: dict, branches=("charge", "discharge"), window=(2.5, 4.2),
                    I=0.0, noise=1e-3, n=200, fit_R=False, cap_sd=0.01):
    """Returns dict name -> relative sd (sd / value) and the correlation matrix."""
    cell = Cell(si_ocp)
    brs = [cell.simulate_branch({**truth, "R": truth.get("R", 0.0)}, d, I, window, n)
           for d in branches]
    params = default_params(truth, fit_R=fit_R)
    prob = Problem(cell, brs, params, fixed={"R": truth.get("R", 0.0)} if not fit_R else None,
                   cap_sd=cap_sd)
    x = np.array([truth[p.name] for p in params])
    prior_sd = [np.inf if p.name != "R" else 0.05 for p in params]
    prior_sd = [1e6 if not np.isfinite(s) else s for s in prior_sd]
    C = fisher_covariance(prob, x, noise, prior_sd=prior_sd)
    sd = np.sqrt(np.diag(C))
    rel = {p.name: sd[i] / max(abs(x[i]), 1e-12) for i, p in enumerate(params)}
    corr = C / np.outer(sd, sd)
    return rel, corr, [p.name for p in params]
