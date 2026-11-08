"""Bayesian inference of electrode capacities from one or more measured branches.

Posterior over p = (Q_Gr, Q_Si, Q_P, Q_Li, [R], [w], [hys], log_sigma per branch):
  likelihood : Gaussian voltage residuals per branch (own noise level) +
               Gaussian on each branch's capacity span (sd = cap_sd)
  priors     : weakly-informative, see `default_params`
Tools: MAP (bounded least squares), affine-invariant ensemble MCMC, Laplace evidence
(for comparing / averaging over candidate Si OCPs).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .cell import Branch, Cell

CAPACITY_NAMES = ["Q_Gr", "Q_Si", "Q_P", "Q_Li"]


@dataclass
class Param:
    name: str
    init: float
    lower: float
    upper: float
    logprior: Callable = field(default=lambda v: 0.0)


def _halfnormal(scale):
    return lambda v: -0.5 * (v / scale) ** 2


def _normal(mu, sd):
    return lambda v: -0.5 * ((v - mu) / sd) ** 2


def default_params(guess: dict, fit_R=True, fit_w=False, fit_hys=False, R_scale=0.05):
    """guess: rough capacities in Ah (e.g. from datasheet / fresh-cell fit)."""
    ps = [Param(n, guess[n], 0.02 * guess[n] + 1e-4, 3.0 * guess[n]) for n in CAPACITY_NAMES]
    if fit_R:
        ps.append(Param("R", guess.get("R", 0.02), 0.0, 0.5, _halfnormal(R_scale)))
    if fit_w:
        ps.append(Param("w", 0.5, 0.0, 1.0))
    if fit_hys:
        ps.append(Param("hys", 1.0, 0.0, 2.0, _normal(1.0, 0.3)))
    return ps


class Problem:
    def __init__(self, cell: Cell, branches, params, fixed=None, cap_sd=0.01,
                 v_scale=1e-3, noise="iid"):
        """noise: "iid" (independent Gaussian residuals) or "ar1" (first-order
        autoregressive residuals, a simple model-discrepancy term: systematic misfit
        shows up as phi -> 1 and widens the posterior instead of being read as signal)."""
        if noise not in ("iid", "ar1"):
            raise ValueError("noise must be 'iid' or 'ar1'")
        self.noise = noise
        self.cell = cell
        self.branches = list(branches)
        self.params = list(params)
        self.names = [p.name for p in self.params]
        self.fixed = dict(fixed or {})
        self.cap_sd = cap_sd
        self.v_scale = v_scale
        self.nb = len(self.branches)
        self.lo = np.array([p.lower for p in self.params])
        self.hi = np.array([p.upper for p in self.params])

    # -- helpers ------------------------------------------------------------------
    def as_dict(self, x):
        d = dict(self.fixed)
        d.update(zip(self.names, x[: len(self.names)]))
        return d

    def _branch_residuals(self, x):
        p = self.as_dict(x)
        out = []
        for br in self.branches:
            Vp, span = self.cell.predict(p, br)
            if Vp is None:
                out.append(None)
                continue
            r = np.where(np.isnan(Vp), 0.3, Vp - br.V)
            out.append((r, span - br.span))
        return out

    # -- least squares ------------------------------------------------------------
    def ls_residuals(self, x):
        res = []
        for br, rr in zip(self.branches, self._branch_residuals(x)):
            if rr is None:
                res.append(np.full(len(br.V) + 1, 300.0))
            else:
                r, dq = rr
                res.append(np.append(r / self.v_scale, dq / self.cap_sd))
        return np.concatenate(res)

    def fit_map(self, x0=None):
        x0 = np.array([p.init for p in self.params]) if x0 is None else np.asarray(x0)
        x0 = np.clip(x0, self.lo + 1e-9, self.hi - 1e-9)
        sol = least_squares(self.ls_residuals, x0, bounds=(self.lo, self.hi),
                            x_scale="jac", diff_step=1e-4)
        x = sol.x
        sig, phi = [], []
        for rr in self._branch_residuals(x):
            if rr is None:
                sig.append(np.log(1e-2)); phi.append(0.5)
                continue
            r = rr[0]
            ph = float(np.clip(np.corrcoef(r[:-1], r[1:])[0, 1], 0.0, 0.98)) if len(r) > 3 else 0.0
            innov = r[1:] - ph * r[:-1] if self.noise == "ar1" else r
            sig.append(np.log(max(np.std(innov), 1e-5)))
            phi.append(ph)
        return np.concatenate([x, sig, phi if self.noise == "ar1" else []])

    # -- posterior ----------------------------------------------------------------
    def split(self, z):
        k = len(self.names)
        x, ls = z[:k], z[k:k + self.nb]
        phi = z[k + self.nb:k + 2 * self.nb] if self.noise == "ar1" else np.zeros(self.nb)
        return x, ls, phi

    def log_post(self, z):
        x, ls, phi = self.split(z)
        if (np.any(x < self.lo) or np.any(x > self.hi) or np.any(ls < -12) or np.any(ls > 0)
                or np.any(phi < 0) or np.any(phi > 0.995)):
            return -np.inf
        lp = sum(p.logprior(v) for p, v in zip(self.params, x))
        for rr, s, ph in zip(self._branch_residuals(x), ls, phi):
            if rr is None:
                return -np.inf
            r, dq = rr
            if ph > 0:
                e = np.empty_like(r)
                e[0] = r[0] * np.sqrt(1 - ph ** 2)
                e[1:] = r[1:] - ph * r[:-1]
                lp += 0.5 * np.log(1 - ph ** 2)
            else:
                e = r
            lp += -0.5 * np.sum(e ** 2) * np.exp(-2 * s) - len(r) * s
            lp += -0.5 * (dq / self.cap_sd) ** 2
        return lp

    def sample(self, z0, n_walkers=32, n_steps=1500, burn=600, seed=0, progress=False):
        chain, acc = ensemble_mcmc(self.log_post, z0, n_walkers, n_steps, seed=seed,
                                   progress=progress)
        flat = chain[burn:].reshape(-1, chain.shape[-1])
        return Posterior(self, flat, acc, chain)

    def laplace_log_evidence(self, z_map, h=1e-4):
        """Laplace approximation of log p(data | model). Priors on capacities are flat
        within bounds, so the prior volume term is included explicitly."""
        f0 = self.log_post(z_map)
        H = -numerical_hessian(self.log_post, z_map, h)
        sign, logdet = np.linalg.slogdet(H)
        if sign <= 0:
            return -np.inf
        d = len(z_map)
        log_vol = 0.0  # normalise flat priors on bounded params
        for p in self.params:
            if p.name in CAPACITY_NAMES or p.name == "w":
                log_vol -= np.log(p.upper - p.lower)
        log_vol -= self.nb * np.log(12.0)                          # log-sigma in (-12, 0)
        if self.noise == "ar1":
            log_vol -= self.nb * np.log(0.995)                     # phi in (0, 0.995)
        return f0 + log_vol + 0.5 * d * np.log(2 * np.pi) - 0.5 * logdet


@dataclass
class Posterior:
    problem: Problem
    samples: np.ndarray
    acceptance: float
    chain: np.ndarray

    def get(self, name):
        if name in self.problem.names:
            return self.samples[:, self.problem.names.index(name)]
        return np.full(len(self.samples), self.problem.fixed[name])

    def capacities(self):
        return np.column_stack([self.get(n) for n in CAPACITY_NAMES])

    def summary(self):
        rows = []
        for i, n in enumerate(self.problem.names):
            s = self.samples[:, i]
            rows.append((n, s.mean(), s.std(), *np.percentile(s, [2.5, 97.5])))
        return rows

    def correlation(self):
        return np.corrcoef(self.samples[:, : len(self.problem.names)].T)


# ---------------------------------------------------------------------------
def ensemble_mcmc(logp, z0, n_walkers=32, n_steps=1500, a=2.0, seed=0, init_scale=1e-3,
                  progress=False):
    """Goodman & Weare (2010) affine-invariant stretch move (same algorithm as emcee)."""
    rng = np.random.default_rng(seed)
    nd = len(z0)
    w = np.asarray(z0) + init_scale * (np.abs(z0) + 1e-2) * rng.standard_normal((n_walkers, nd))
    lp = np.array([logp(x) for x in w])
    bad = ~np.isfinite(lp)
    while bad.any():                         # re-draw walkers that start outside support
        w[bad] = z0 + 0.5 * init_scale * (np.abs(z0) + 1e-2) * rng.standard_normal((bad.sum(), nd))
        lp[bad] = [logp(x) for x in w[bad]]
        bad = ~np.isfinite(lp)
    chain = np.empty((n_steps, n_walkers, nd))
    half, acc = n_walkers // 2, 0
    groups = (np.arange(half), np.arange(half, n_walkers))
    for t in range(n_steps):
        for k in (0, 1):
            idx, other = groups[k], w[groups[1 - k]]
            z = ((a - 1) * rng.random(len(idx)) + 1) ** 2 / a
            prop = other[rng.integers(len(other), size=len(idx))]
            prop = prop + z[:, None] * (w[idx] - prop)
            lpp = np.array([logp(x) for x in prop])
            ok = np.log(rng.random(len(idx))) < (nd - 1) * np.log(z) + lpp - lp[idx]
            w[idx[ok]], lp[idx[ok]] = prop[ok], lpp[ok]
            acc += ok.sum()
        chain[t] = w
        if progress and (t + 1) % max(n_steps // 10, 1) == 0:
            print(f"    mcmc {t + 1}/{n_steps}")
    return chain, acc / (n_steps * n_walkers)


def numerical_hessian(f, x, h=1e-4):
    x = np.asarray(x, float)
    n = len(x)
    H = np.empty((n, n))
    step = h * np.maximum(np.abs(x), 1e-2)
    f0 = f(x)
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n); ei[i] = step[i]
            ej = np.zeros(n); ej[j] = step[j]
            if i == j:
                v = (f(x + ei) - 2 * f0 + f(x - ei)) / step[i] ** 2
            else:
                v = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                    4 * step[i] * step[j])
            H[i, j] = H[j, i] = v
    return H
