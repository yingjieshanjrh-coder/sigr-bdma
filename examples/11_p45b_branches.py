"""Second cell chemistry, both branches: Molicel P45B (NCA | graphite-SiOx).

Data: pOCV charge AND discharge (C/30) of one P45B cell at 9 check-ups over 800 EFC,
shipped as test data with the TUM-EES DegradationModeAnalysis repository (MIT;
Rehm et al., J. Power Sources 670 (2026) 239418), together with the authors' DMA results.

For every combination of graphite OCP (4 with both branches) and Si OCP (10 with both
branches), CU1 and CU9 are fitted three ways:
  charge-only, discharge-only, and joint (both branches, branch-aware graphite and Si OCPs).
Question: does the joint fit narrow the OCP-induced spread of the anode split?
The NCA OCP is the P45B delithiation curve for both branches; R is fixed at 15 mOhm
(C/30, 0.15 A -> ~2 mV IR drop; no pulse data in this file).

usage: python 11_p45b_branches.py "<TUM DMA repo>"
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import tum_p45b_branches, tum_p45b_published
from sigr_bdma.inference import Problem, default_params
from sigr_bdma.ocp_library import _read_tum_mat, load_tum_ocps

R_FIXED = 0.015
STARTS = [dict(Q_Gr=qg, Q_Si=qs, Q_P=qp, Q_Li=ql)
          for qg, qs in ((4.0, 0.6), (3.5, 1.0)) for qp in (4.9, 5.8) for ql in (4.6, 5.6)]


def fit(cell, brs, starts):
    best = None
    for g in starts:
        prob = Problem(cell, brs, default_params(g, fit_R=False), fixed={"R": R_FIXED})
        try:
            z = prob.fit_map()
        except Exception:
            continue
        rr = prob._branch_residuals(z)
        if any(r is None for r in rr):
            continue
        rms = float(np.sqrt(np.mean(np.concatenate([r[0] for r in rr]) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, z[:4])
    return best


def main(tum):
    lib = load_tum_ocps(tum)
    xc, vc = _read_tum_mat(os.path.join(tum, "input_data/NCA/P45B_Cathode_Delithiation_0C03.mat"))
    upos = lambda y: np.interp(y, xc, vc)
    cus = tum_p45b_branches(tum)
    first, last = cus[0], cus[-1]
    pub = tum_p45b_published(tum)
    grs = {n: d for n, d in lib["graphite"].items() if "lith" in d and "delith" in d}
    sis = {"Verbrugge2016": s.si_verbrugge2016()}
    sis.update({n: s.SiOCP(n, d["lith"], d["delith"]) for n, d in lib["silicon"].items()
                if "lith" in d and "delith" in d})
    configs = {"charge": ["charge"], "discharge": ["discharge"], "joint": ["charge", "discharge"]}
    rows = []
    for (gn, g), (sn, si) in itertools.product(grs.items(), sis.items()):
        cell = s.Cell(si, U_pos=upos, pos_range=(0.0, 1.0),
                      U_gr={"charge": g["lith"], "discharge": g["delith"]})
        for cname, dirs in configs.items():
            b0 = fit(cell, [first[d] for d in dirs], STARTS)
            if b0 is None:
                continue
            st1 = [dict(zip(s.inference.CAPACITY_NAMES, b0[1] * f)) for f in (1.0, 0.9, 0.8)]
            b1 = fit(cell, [last[d] for d in dirs], st1)
            if b1 is None:
                continue
            m = s.modes.modes_from_capacities(b0[1], b1[1])
            row = dict(graphite=gn, silicon=sn, config=cname, rms0_mV=1e3 * b0[0], rms1_mV=1e3 * b1[0])
            row.update({n: 100 * v for n, v in zip(s.MODE_NAMES, m)})
            rows.append(row)
        print(f"{gn:9s} {sn:14s} done", flush=True)
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/11_p45b_branches.csv", index=False)

    res["rms"] = 0.5 * (res.rms0_mV + res.rms1_mV)
    ok = (res[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (res.LAM_Si < 95)
    print("\npublished (charge-direction DMA, CU9):",
          "  ".join(f"{k}={v:.1f}" for k, v in pub.items()))
    for thr in (6.0, 8.0, 10.0):
        print(f"\nplausible fits with mean RMS <= {thr:.0f} mV: range over OCP choices (spread, pp)")
        for cname in configs:
            g = res[ok & (res.config == cname) & (res.rms <= thr)]
            if len(g) < 2:
                print(f"  {cname:9s} n={len(g)}"); continue
            print(f"  {cname:9s} n={len(g):2d} " + "  ".join(
                f"{m} {g[m].min():5.1f}-{g[m].max():5.1f} ({g[m].max() - g[m].min():4.1f})"
                for m in s.MODE_NAMES))
    return res


if __name__ == "__main__":
    main(sys.argv[1])
