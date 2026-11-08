"""Real-data case study: LG M50T (NMC811 | graphite-SiOx), Kirkaldy et al.,
ACS Appl. Energy Mater. 2022, data: Zenodo 10.5281/zenodo.7235857 (not redistributed).

The dataset provides C/10 DISCHARGE curves at each RPT (no C/10 charge), so this is a
discharge-only fit: delithiation-branch Si OCP, ohmic term inferred with a prior.
OCPs: graphite Ai et al. 2020 (measured), Si Verbrugge et al. 2016 (delith. branch),
NMC811 Chen et al. 2020. Results are compared with the authors' own (least-squares)
degradation modes shipped with the dataset.

usage: python 05_kirkaldy_lgm50t.py "<path to 'Battery Degradation Data'>" [cell] [expt]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s


def load_branches(root, cell="D", soc="0-100% SoC"):
    d = os.path.join(root, "Timeseries Data from RPTs", soc, "0.1C Voltage Curves", f"Cell {cell}")
    files = glob.glob(os.path.join(d, "*0.1C discharge data.csv"))
    files.sort(key=lambda f: int(re.search(r"RPT(\d+)", f).group(1)))
    out = []
    for f in files:
        df = pd.read_csv(f)
        I = abs(df["Current (mA)"].mean()) / 1000
        out.append([s.branch_from_arrays(df["Charge (mA.h)"] / 1000, df["Voltage (V)"],
                                         "discharge", I, n=300,
                                         label=re.search(r"RPT\d+", f).group(0))])
    return out


def load_published(root, cell="D", soc="0-100% SoC"):
    f = glob.glob(os.path.join(root, "Extracted Data - Degradation Modes", soc, f"cell {cell}",
                               "*DM data*.csv"))[0]
    return pd.read_csv(f, index_col=0)


def main(root, cell="D", n_steps=1000, burn=400):
    sets = load_branches(root, cell)
    print(f"cell {cell}: {len(sets)} RPTs")
    model = s.Cell(s.si_verbrugge2016(), U_gr=s.graphite_ai2020)
    guess = dict(Q_Gr=4.27, Q_Si=0.65, Q_P=7.7, Q_Li=7.0, R=0.026)   # from a multi-start fit of RPT0
    posts = s.fit_series(model, sets, guess, fit_R=True, n_steps=n_steps, burn=burn)
    rms = []
    for p, brs in zip(posts, sets):
        r = p.problem._branch_residuals(p.z_map)[0][0]
        rms.append(np.sqrt(np.mean(r ** 2)) * 1e3)
    traj = s.trajectory(posts)
    pub = load_published(root, cell)
    colmap = {"LAM_Gr": "LAM NE_Gr", "LAM_Si": "LAM NE_Si", "LAM_NE": "LAM NE_tot",
              "LAM_P": "LAM PE", "LLI": "LLI"}
    rows = []
    for i in range(len(posts)):
        row = {"RPT": i, "rms_mV": rms[i], "R_ohm": float(np.median(posts[i].get("R")))}
        for m, c in colmap.items():
            med, lo, hi = 100 * traj[m][i]
            row[f"{m}"] = med
            row[f"{m}_lo"] = lo
            row[f"{m}_hi"] = hi
            row[f"{m}_published"] = 100 * pub[c].iloc[i] if i < len(pub) else np.nan
        rows.append(row)
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv(f"results/05_kirkaldy_cell{cell}.csv", index=False)
    show = ["RPT", "rms_mV", "R_ohm"] + [k for m in colmap for k in (m, f"{m}_lo", f"{m}_hi",
                                                                      f"{m}_published")]
    pd.set_option("display.width", 250, "display.precision", 2)
    print(res[show].to_string(index=False))
    return res


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["D"]))
