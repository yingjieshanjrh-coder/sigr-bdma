"""Does the fragility of the graphite/silicon split depend on how much silicon the anode holds?

Synthetic C/10 discharge curves (fresh and aged) are generated with one OCP pair as the
"true" model, for fresh silicon capacity shares between 5 and 35 % of the negative
electrode. Ageing: LAM_Gr 10 %, LAM_Si 40 %, LAM_P 8 %, LLI 15 %. The curves are then
fitted with every other pair in the library (the true pair is excluded, as for real data).
Reported per share: spread of each mode over the plausible fits and the error of the
lowest-RMS fit.

usage: python 17_si_fraction.py "<TUM DMA repo>"
"""
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.inference import CAPACITY_NAMES
from sigr_bdma.ocp_library import load_tum_ocps
from sigr_bdma.sensitivity import ocp_ensemble, plausible

MODES = s.MODE_NAMES
TRUE_LOSS = dict(LAM_Gr=10.0, LAM_Si=40.0, LAM_P=8.0, LLI=15.0)
SHARES = (0.05, 0.10, 0.15, 0.25, 0.35)
TRUTHS = (("Wetjen", "Rehm2026"), ("Ai2020", "Verbrugge2016"))


def library(tum):
    lib = load_tum_ocps(tum)
    grs = {"Ai2020": s.graphite_ai2020}
    grs.update({n: d["delith"] for n, d in lib["graphite"].items() if "delith" in d})
    sis = {"Verbrugge2016": s.si_verbrugge2016()}
    sis.update({n: s.SiOCP(n, d["lith"], d["delith"]) for n, d in lib["silicon"].items()
                if "lith" in d and "delith" in d})
    return grs, sis


def true_caps(share, q_ne=5.6, q_p=5.9, q_li=5.6):
    """Fresh capacities (Ah) close to the LG M50T fit, with Si share `share` of Q_NE."""
    fresh = np.array([q_ne * (1 - share), q_ne * share, q_p, q_li])
    loss = np.array([TRUE_LOSS["LAM_Gr"], TRUE_LOSS["LAM_Si"], TRUE_LOSS["LAM_P"],
                     TRUE_LOSS["LLI"]]) / 100
    return fresh, fresh * (1 - loss)


def main(tum, seed=0):
    grs, sis = library(tum)
    rng = np.random.default_rng(seed)
    rows, summ = [], []
    for tg, ts in TRUTHS:
        tcell = s.Cell(sis[ts], U_gr=grs[tg])
        for share in SHARES:
            z0, z1 = true_caps(share)
            try:
                fr = tcell.simulate_branch({**dict(zip(CAPACITY_NAMES, z0)), "R": 0.026},
                                           "discharge", I=0.5, noise=1e-3, rng=rng)
                ag = tcell.simulate_branch({**dict(zip(CAPACITY_NAMES, z1)), "R": 0.031},
                                           "discharge", I=0.5, noise=1e-3, rng=rng)
            except ValueError as e:
                print(f"truth {tg}/{ts} share {share}: {e}")
                continue
            g_lib = dict(grs)
            s_lib = dict(sis)
            res = ocp_ensemble(g_lib, s_lib, fr, ag, 0.026, 0.031)
            res = res[~((res.graphite == tg) & (res.silicon == ts))]   # truth not in library
            res["truth"], res["share"] = f"{tg}/{ts}", share
            rows.append(res)
            p = plausible(res)
            best = res.assign(rms=0.5 * (res.rms0_mV + res.rms1_mV)).sort_values("rms").iloc[0]
            row = dict(truth=f"{tg}/{ts}", share=share, n=len(p), best_rms=best.rms)
            for m in MODES:
                t = TRUE_LOSS.get(m, np.nan)
                if m == "LAM_NE":
                    t = 100 * (1 - (z1[0] + z1[1]) / (z0[0] + z0[1]))
                row[f"{m}_range"] = np.ptp(p[m]) if len(p) else np.nan
                row[f"{m}_maxerr"] = np.max(np.abs(p[m] - t)) if len(p) else np.nan
                row[f"{m}_besterr"] = best[m] - t
            summ.append(row)
            print(f"{tg}/{ts} share {share:.2f}: n={len(p):2d} best {best.rms:.1f} mV  "
                  + "  ".join(f"{m} range {row[m + '_range']:.1f}" for m in MODES))
    pd.concat(rows).to_csv("results/17_si_fraction_all.csv", index=False)
    summ = pd.DataFrame(summ)
    summ.to_csv("results/17_si_fraction_summary.csv", index=False)
    print(summ.round(1).to_string(index=False))
    return summ


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    main(sys.argv[1])
