"""All eight 0-100 % SoC cells of Kirkaldy et al. (LG M50T, 10/25/40 degC), discharge-only.

Two noise models are compared on every fit:
  iid : independent residuals (standard, ignores model discrepancy)
  ar1 : autoregressive residuals (model-discrepancy aware)
For each, we report the end-of-test modes with 95 % intervals and whether the
published least-squares value falls inside the interval ("covered").

usage: python 06_kirkaldy_all_cells.py "<inner 'Battery Degradation Data' folder>"
"""
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import (KIRKALDY_COLMAP, KIRKALDY_TEMPS, kirkaldy2022_branches,
                                kirkaldy2022_published_modes)

GUESS = dict(Q_Gr=4.27, Q_Si=0.65, Q_P=7.7, Q_Li=7.0, R=0.026)


def run(root, cells="ABCDEFGH", n_steps=800, burn=300):
    model = s.Cell(s.si_verbrugge2016(), U_gr=s.graphite_ai2020)
    rows = []
    for cell in cells:
        idx, sets = kirkaldy2022_branches(root, cell)
        if len(sets) < 2:
            print(f"cell {cell}: not enough RPT files staged, skipped"); continue
        pub = kirkaldy2022_published_modes(root, cell)
        for noise in ("iid", "ar1"):
            print(f"== cell {cell} ({KIRKALDY_TEMPS[cell]} degC), noise={noise}, RPTs {idx}")
            posts = s.fit_series(model, sets, GUESS, fit_R=True, noise=noise,
                                 n_steps=n_steps, burn=burn, verbose=False)
            last = posts[-1]
            r = last.problem._branch_residuals(last.z_map)[0][0]
            modes = s.paired_modes(posts[0], last)
            row = dict(cell=cell, T=KIRKALDY_TEMPS[cell], noise=noise, rpt=idx[-1],
                       rms_mV=1e3 * np.sqrt(np.mean(r ** 2)))
            if noise == "ar1":
                row["phi"] = float(np.median(last.samples[:, -1]))
            for i, m in enumerate(s.MODE_NAMES):
                lo, med, hi = np.percentile(100 * modes[:, i], [2.5, 50, 97.5])
                p = 100 * pub[KIRKALDY_COLMAP[m]].loc[idx[-1]]
                row.update({m: med, f"{m}_lo": lo, f"{m}_hi": hi, f"{m}_pub": p,
                            f"{m}_cov": bool(lo <= p <= hi)})
            rows.append(row)
            print("   " + "  ".join(f"{m}={row[m]:.1f}[{row[m + '_lo']:.1f},{row[m + '_hi']:.1f}]"
                                    f"/pub {row[m + '_pub']:.1f}" for m in s.MODE_NAMES))
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv("results/06_kirkaldy_all_cells.csv", index=False)
    print("\ncoverage of published values by 95 % interval (fraction of cells):")
    for noise in ("iid", "ar1"):
        sub = res[res.noise == noise]
        print(f"  {noise}: " + "  ".join(f"{m}={sub[m + '_cov'].mean():.2f}" for m in s.MODE_NAMES)
              + f"   median interval width LAM_Si={np.median(sub['LAM_Si_hi'] - sub['LAM_Si_lo']):.1f} pp")
    return res


if __name__ == "__main__":
    run(sys.argv[1], *(sys.argv[2:3] or ["ABCDEFGH"]))
