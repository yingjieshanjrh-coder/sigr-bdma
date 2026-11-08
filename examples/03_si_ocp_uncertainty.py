"""Propagating Si-OCP uncertainty into the degradation modes.

Literature Si OCPs differ substantially, and the chosen curve changes the estimated
modes (Rehm et al., J. Power Sources 2026). Here the TRUE Si OCP lies between two
candidate curves (w_true = 0.35). Compared:
  fixed-A / fixed-B : pick one literature curve (common practice)
  mixture           : infer the weight w jointly with the capacities (this work)
  BMA (Laplace)     : average fixed-A and fixed-B by their Laplace evidence (this work)
"""
import numpy as np

import sigr_bdma as s

FRESH = dict(Q_Gr=5.0, Q_Si=1.0, Q_P=5.6, Q_Li=5.5, R=0.030)
LOSS = dict(Q_Gr=0.05, Q_Si=0.25, Q_P=0.03, Q_Li=0.10)
AGED = {k: v * (1 - LOSS.get(k, 0)) for k, v in FRESH.items()}
AGED["R"] = 0.045
W_TRUE = 0.35
TRUE = np.array([0.05, 0.25, 1 - (AGED["Q_Gr"] + AGED["Q_Si"]) / 6.0, 0.03, 0.10])


def main(n_steps=1500):
    A = s.placeholder_si("A")
    B = s.placeholder_si("B", shift=0.04, gap_scale=0.6)
    truth_cell = s.Cell(s.mix_si(A, B, W_TRUE))
    rng = np.random.default_rng(3)
    D = {st: [truth_cell.simulate_branch(state, d, 0.5, noise=1e-3, rng=rng)
              for d in ("charge", "discharge")]
         for st, state in (("fresh", FRESH), ("aged", AGED))}
    guess = {k: v * 0.95 for k, v in FRESH.items()}

    out, evid = {}, {}
    for name, cell, fit_w in (("fixed-A", s.Cell(A), False), ("fixed-B", s.Cell(B), False),
                              ("mixture", s.Cell([A, B]), True)):
        print(f"== {name} ==")
        posts = [s.fit_state(cell, D[st], guess, fit_w=fit_w, n_steps=n_steps)
                 for st in ("fresh", "aged")]
        out[name] = s.paired_modes(*posts)
        if fit_w:
            for p, st in zip(posts, ("fresh", "aged")):
                w = p.get("w")
                print(f"    w ({st}) = {w.mean():.3f} ± {w.std():.3f}   (true {W_TRUE})")
        else:
            evid[name] = sum(p.problem.laplace_log_evidence(p.z_map) for p in posts)

    # Bayesian model averaging of the two fixed-curve results
    le = np.array([evid["fixed-A"], evid["fixed-B"]])
    wts = np.exp(le - le.max()); wts /= wts.sum()
    n = min(len(out["fixed-A"]), len(out["fixed-B"]))
    rng = np.random.default_rng(0)
    pick = rng.random(n) < wts[0]
    out["BMA(A,B)"] = np.where(pick[:, None], out["fixed-A"][:n], out["fixed-B"][:n])
    print(f"\nLaplace log-evidence A={le[0]:.1f}, B={le[1]:.1f} -> BMA weights {np.round(wts, 3)}")

    print("\nDegradation modes, % (mean ± sd)")
    print(f"{'mode':7s} {'truth':>6s} " + "".join(f"{k:>18s}" for k in out))
    for i, m in enumerate(s.MODE_NAMES):
        print(f"{m:7s} {100 * TRUE[i]:6.1f} " + "".join(
            f"{100 * v[:, i].mean():10.1f} ± {100 * v[:, i].std():4.1f}" for v in out.values()))
    return out


if __name__ == "__main__":
    main()
