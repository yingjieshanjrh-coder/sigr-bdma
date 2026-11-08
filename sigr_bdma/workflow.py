"""Convenience wrappers: MAP + MCMC for one state, and for a series of RPTs."""
from __future__ import annotations

from .inference import Problem, default_params


def fit_state(cell, branches, guess, fit_R=True, fit_w=False, fit_hys=False, fixed=None,
              noise="iid", n_walkers=32, n_steps=1500, burn=600, seed=0, verbose=True):
    params = default_params(guess, fit_R=fit_R, fit_w=fit_w, fit_hys=fit_hys)
    fixed = dict(fixed or {})
    if not fit_R:
        fixed.setdefault("R", 0.0)
    prob = Problem(cell, branches, params, fixed=fixed, noise=noise)
    z0 = prob.fit_map()
    post = prob.sample(z0, n_walkers, n_steps, burn, seed)
    post.z_map = z0
    if verbose:
        print("    MAP: " + ", ".join(f"{n}={v:.4g}" for n, v in zip(prob.names, z0)),
              f"| acc={post.acceptance:.2f}")
    return post


def fit_series(cell, branch_sets, guess, **kw):
    """branch_sets: list (one per RPT) of lists of Branch. Each fit starts from the
    previous MAP. Returns list of Posterior."""
    posts, g = [], dict(guess)
    for i, brs in enumerate(branch_sets):
        post = fit_state(cell, brs, g, seed=i, **kw)
        g.update({n: v for n, v in zip(post.problem.names, post.z_map)})
        posts.append(post)
    return posts
