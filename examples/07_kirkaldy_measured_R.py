"""LG M50T cell series with the ohmic resistance MEASURED from GITT pulses (fixed per RPT)
instead of inferred, compared with the inferred-R fit and the published modes.

Why: with discharge-only data, an inferred R absorbs all polarisation (it grows from 26 to
~70 mOhm over life for cell D) and trades off against electrode capacities. The measured
instantaneous R (26 -> 32 mOhm) is an independent constraint.

usage: python 07_kirkaldy_measured_R.py "<inner 'Battery Degradation Data' folder>" [cells]
"""
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import (KIRKALDY_COLMAP, KIRKALDY_TEMPS, interpolate_resistance,
                                kirkaldy2022_branches, kirkaldy2022_published_modes,
                                kirkaldy2022_resistance)

GUESS = dict(Q_Gr=4.27, Q_Si=0.65, Q_P=7.7, Q_Li=7.0)


def fit_cell(root, cell, model, n_steps=800, burn=300):
    idx, sets = kirkaldy2022_branches(root, cell)
    R = interpolate_resistance(kirkaldy2022_resistance(root, cell), idx)
    posts, g = [], dict(GUESS)
    for k, brs in zip(idx, sets):
        p = s.fit_state(model, brs, g, fit_R=False, fixed={"R": R[k]}, n_steps=n_steps,
                        burn=burn, seed=k, verbose=False)
        g.update(zip(p.problem.names, p.z_map))
        posts.append(p)
    return idx, posts, R


def main(root, cells="ABCDEFGH"):
    model = s.Cell(s.si_verbrugge2016(), U_gr=s.graphite_ai2020)
    rows = []
    for cell in cells:
        try:
            idx, posts, R = fit_cell(root, cell, model)
        except (IndexError, ValueError) as e:
            print(f"cell {cell}: skipped ({e})"); continue
        pub = kirkaldy2022_published_modes(root, cell)
        modes = s.paired_modes(posts[0], posts[-1])
        r = posts[-1].problem._branch_residuals(posts[-1].z_map)[0][0]
        row = dict(cell=cell, T=KIRKALDY_TEMPS[cell], rpt=idx[-1], R0_mOhm=1e3 * R[idx[0]],
                   Rend_mOhm=1e3 * R[idx[-1]], rms_mV=1e3 * np.sqrt(np.mean(r ** 2)))
        for i, m in enumerate(s.MODE_NAMES):
            lo, med, hi = np.percentile(100 * modes[:, i], [2.5, 50, 97.5])
            p = 100 * pub[KIRKALDY_COLMAP[m]].loc[idx[-1]]
            row.update({m: med, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_pub": p,
                        f"{m}_cov": bool(lo <= p <= hi)})
        rows.append(row)
        print(f"cell {cell} ({row['T']} degC) R {row['R0_mOhm']:.1f}->{row['Rend_mOhm']:.1f} mOhm, "
              f"rms {row['rms_mV']:.1f} mV | " + "  ".join(
                  f"{m} {row[m]:.1f} [{row[m + '_lo']:.1f},{row[m + '_hi']:.1f}] pub {row[m + '_pub']:.1f}"
                  for m in s.MODE_NAMES))
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/07_kirkaldy_measured_R.csv", index=False)
    print("coverage: " + "  ".join(f"{m}={res[m + '_cov'].mean():.2f}" for m in s.MODE_NAMES))
    return res


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["ABCDEFGH"]))
