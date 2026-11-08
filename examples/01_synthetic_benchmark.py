"""Synthetic benchmark with known degradation.

Truth: NMC811 | Gr+Si cell, branch-dependent Si OCP, ohmic drop at C/10, fresh vs aged.
Compared methods
  A  this work     : charge + discharge jointly, branch-aware Si OCP, R inferred
  B1 conventional  : discharge only, mean (hysteresis-free) Si OCP, no IR correction
  B2 best practice : discharge only, delithiation Si OCP, R known (IR-corrected),
                     i.e. the protocol recommended in recent work; deterministic fit
                     -> we still sample it to show its uncertainty
"""
import numpy as np

import sigr_bdma as s

FRESH = dict(Q_Gr=5.0, Q_Si=1.0, Q_P=5.6, Q_Li=5.5, R=0.030)
AGED_MODES = dict(Q_Gr=0.05, Q_Si=0.25, Q_P=0.03, Q_Li=0.10)
AGED = {k: v * (1 - AGED_MODES.get(k, 0)) for k, v in FRESH.items()}
AGED["R"] = 0.045
I_C10 = 0.5
TRUE = np.array([0.05, 0.25,
                 1 - (AGED["Q_Gr"] + AGED["Q_Si"]) / (FRESH["Q_Gr"] + FRESH["Q_Si"]),
                 0.03, 0.10])


def data(cell, state, seed):
    rng = np.random.default_rng(seed)
    return {d: cell.simulate_branch(state, d, I_C10, noise=1e-3, rng=rng)
            for d in ("charge", "discharge")}


def main(n_steps=1500):
    si = s.placeholder_si()
    truth_cell = s.Cell(si)
    D = {"fresh": data(truth_cell, FRESH, 1), "aged": data(truth_cell, AGED, 2)}
    guess = {k: v * 0.95 for k, v in FRESH.items()}

    configs = {
        "A (this work)": dict(cell=s.Cell(si), dirs=("charge", "discharge"), fit_R=True,
                              fixed={}),
        "B1 conventional": dict(cell=s.Cell(si), dirs=("discharge",), fit_R=False,
                                fixed={"hys": 0.0, "R": 0.0}),
        "B2 best practice": dict(cell=s.Cell(si), dirs=("discharge",), fit_R=False,
                                 fixed={"R": "true"}),
    }
    results = {}
    for name, c in configs.items():
        print(f"== {name} ==")
        posts = []
        for st, state in (("fresh", FRESH), ("aged", AGED)):
            fixed = {k: (state[k] if v == "true" else v) for k, v in c["fixed"].items()}
            brs = [D[st][d] for d in c["dirs"]]
            posts.append(s.fit_state(c["cell"], brs, guess, fit_R=c["fit_R"], fixed=fixed,
                                     n_steps=n_steps))
        results[name] = s.paired_modes(*posts)

    print("\nDegradation modes, % (posterior mean ± sd; truth in first column)")
    print(f"{'mode':7s} {'truth':>6s} " + "".join(f"{n:>24s}" for n in results))
    for i, m in enumerate(s.MODE_NAMES):
        row = f"{m:7s} {100 * TRUE[i]:6.1f} "
        for v in results.values():
            row += f"{100 * v[:, i].mean():14.1f} ± {100 * v[:, i].std():5.1f}  "
        print(row)
    return results


if __name__ == "__main__":
    main()
