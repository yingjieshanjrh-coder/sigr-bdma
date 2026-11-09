"""Non-ohmic polarisation from GITT relaxation (LG M50T, Kirkaldy et al.).

For each 0.5C GITT pulse: ohmic resistance from the voltage step 0.15 s after the current is
switched off, and the non-ohmic overpotential from the subsequent relaxation to the end of
the rest (expressed per ampere). Compares the first and last GITT of a cell.

usage: python 14_gitt_relaxation.py "<inner 'Battery Degradation Data'>" [cell]
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd


def relaxation(path):
    df = pd.read_csv(path)
    t, I, V = df["Time (s)"].values, df["Current (mA)"].values / 1000, df["Voltage (V)"].values
    on = I < -0.5
    rows = []
    for e in np.where(~on[1:] & on[:-1])[0]:
        after = np.where(t >= t[e] + 0.15)[0][0]
        rest = np.where((t > t[e]) & (np.abs(I) < 0.01))[0]
        gaps = np.where(np.diff(rest) > 1)[0]
        blk = rest[: gaps[0] + 1] if len(gaps) else rest
        if len(blk) < 10:
            continue
        rows.append(dict(V_end_pulse=V[e], R_ohm_mOhm=1e3 * (V[after] - V[e]) / -I[e],
                         relax_mV=1e3 * (V[blk[-1]] - V[after]),
                         relax_per_A_mOhm=1e3 * (V[blk[-1]] - V[after]) / -I[e],
                         rest_s=t[blk[-1]] - t[e]))
    return pd.DataFrame(rows)


def main(root, cell="D"):
    d = os.path.join(root, "Timeseries Data from RPTs", "0-100% SoC", "GITT Voltage Curves", f"Cell {cell}")
    files = sorted(glob.glob(os.path.join(d, "*5-pulse GITT*.csv")),
                   key=lambda f: int(re.search(r"RPT(\d+)", f).group(1)))
    files = [f for f in files if "25-pulse" not in f]
    out = []
    for f in (files[0], files[-1]):
        r = relaxation(f)
        r.insert(0, "rpt", int(re.search(r"RPT(\d+)", f).group(1)))
        out.append(r)
    res = pd.concat(out)
    os.makedirs("results", exist_ok=True)
    res.to_csv(f"results/14_gitt_relaxation_cell{cell}.csv", index=False)
    print(res.round(1).to_string(index=False))
    return res


if __name__ == "__main__":
    main(sys.argv[1], *(sys.argv[2:3] or ["D"]))
