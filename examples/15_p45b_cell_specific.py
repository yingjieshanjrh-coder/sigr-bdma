"""Reference with cell-specific OCPs: Molicel P45B.

The TUM-EES collection contains lithiation/delithiation OCPs of the blend anode harvested from
P45B cells (input_data/silicon-graphite/P45B_Anode_*_0C03.mat) and the P45B NCA delithiation
curve. Fitting CU1 and CU9 with these cell-specific curves (blend treated as one material)
gives reference values for LLI, LAM_NE and LAM_PE. They are compared with the spread obtained
from the 40 literature graphite/silicon pairs (example 11) and with the published values.
The blend curve cannot resolve LAM_Gr and LAM_Si.

usage: python 15_p45b_cell_specific.py "<TUM DMA repo>"
"""
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import tum_p45b_branches, tum_p45b_published
from sigr_bdma.inference import Param, Problem
from sigr_bdma.ocp_library import _read_tum_mat

R_FIXED = 0.015
MODES3 = ["LAM_NE", "LAM_P", "LLI"]


def blend_cell(tum):
    d = os.path.join(tum, "input_data")
    xl, vl = _read_tum_mat(os.path.join(d, "silicon-graphite", "P45B_Anode_Lithiation_0C03.mat"))
    xd, vd = _read_tum_mat(os.path.join(d, "silicon-graphite", "P45B_Anode_Delithiation_0C03.mat"))
    xc, vc = _read_tum_mat(os.path.join(d, "NCA", "P45B_Cathode_Delithiation_0C03.mat"))
    dummy_si = s.SiOCP("none", lambda x: 0.1 + 0 * np.asarray(x), lambda x: 0.1 + 0 * np.asarray(x))
    return s.Cell(dummy_si, U_pos=lambda y: np.interp(y, xc, vc), pos_range=(0.0, 1.0),
                  U_gr={"charge": lambda x: np.interp(x, xl, vl),
                        "discharge": lambda x: np.interp(x, xd, vd)})


def fit(cell, brs, starts):
    best = None
    for g in starts:
        params = [Param("Q_Gr", g["Q_Gr"], 0.5, 15.0), Param("Q_P", g["Q_P"], 0.5, 15.0),
                  Param("Q_Li", g["Q_Li"], 0.5, 15.0)]
        prob = Problem(cell, brs, params, fixed={"Q_Si": 1e-6, "R": R_FIXED})
        try:
            z = prob.fit_map()
        except Exception:
            continue
        rr = prob._branch_residuals(z)
        if any(r is None for r in rr):
            continue
        rms = float(np.sqrt(np.mean(np.concatenate([r[0] for r in rr]) ** 2)))
        if best is None or rms < best[0]:
            best = (rms, z[:3])
    return best


def main(tum):
    cell = blend_cell(tum)
    cus = tum_p45b_branches(tum)
    starts = [dict(Q_Gr=qn, Q_P=qp, Q_Li=ql) for qn in (4.6, 5.2) for qp in (4.8, 5.6) for ql in (4.6, 5.4)]
    rows = []
    for cname, dirs in {"charge": ["charge"], "discharge": ["discharge"],
                        "joint": ["charge", "discharge"]}.items():
        a = fit(cell, [cus[0][d] for d in dirs], starts)
        b = fit(cell, [cus[-1][d] for d in dirs],
                [dict(Q_Gr=a[1][0] * k, Q_P=a[1][1] * k, Q_Li=a[1][2] * k) for k in (1, .9, .8)])
        q0, q1 = a[1], b[1]
        rows.append(dict(config=cname, rms0_mV=1e3 * a[0], rms1_mV=1e3 * b[0],
                         LAM_NE=100 * (1 - q1[0] / q0[0]), LAM_P=100 * (1 - q1[1] / q0[1]),
                         LLI=100 * (1 - q1[2] / q0[2])))
    ref = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    ref.to_csv("results/15_p45b_cell_specific.csv", index=False)
    print("cell-specific OCPs (P45B blend anode + P45B NCA), CU1 -> CU9:")
    print(ref.round(2).to_string(index=False))

    lit = pd.read_csv("results/11_p45b_branches.csv")
    lit["rms"] = 0.5 * (lit.rms0_mV + lit.rms1_mV)
    lit = lit[(lit[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (lit.LAM_Si < 95)]
    print("\nliterature graphite/Si pairs, five best per configuration (range):")
    for cname, g in lit.groupby("config"):
        g = g.nsmallest(5, "rms")
        r = ref[ref.config == cname].iloc[0]
        print(f"  {cname:9s} " + "  ".join(
            f"{m} {g[m].min():5.1f}-{g[m].max():5.1f} (ref {r[m]:5.1f})" for m in MODES3))
    pub = tum_p45b_published(tum)
    print("\npublished (TUM, charge direction): " + "  ".join(f"{m}={pub[m]:.1f}" for m in MODES3))
    return ref


if __name__ == "__main__":
    main(sys.argv[1])
