"""TEMPLATE: fit a series of reference performance tests (RPTs) from real cycler data.

Written for datasets like Kirkaldy et al. (LG M50T, Zenodo record 7235857), which contain
C/10 discharge AND charge at every RPT. Column names and file layout below are
ASSUMPTIONS - open one file, check the headers, and edit COLS / the file loop.

Steps
 1. Si OCP: measured branches (SiOCP.from_csv) or extracted from a blend-anode
    half-cell (extract_si_from_blend). Optionally give TWO candidates and fit_w=True.
 2. For each RPT: cut out the C/10 charge and discharge segments -> Branch objects.
 3. fit_series -> one posterior per RPT -> trajectory of degradation modes with bands.

Note: the graphite and NMC811 OCPs shipped here are the Chen2020 fits for the LG M50.
For other cells, supply matching half-cell OCPs.
"""
import glob
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s

COLS = dict(t="Time [s]", I="Current [A]", V="Voltage [V]")   # <- EDIT to match the files
C10_CURRENT = 0.5                                              # A, ~C/10 for a 5 Ah cell
CHARGE_POSITIVE = True                                         # sign convention of the file


def load_rpt(path):
    df = pd.read_csv(path)
    brs = s.split_cycler_log(df[COLS["t"]], df[COLS["I"]], df[COLS["V"]], C10_CURRENT,
                             charge_positive=CHARGE_POSITIVE)
    ch = [b for b in brs if b.direction == "charge"]
    dis = [b for b in brs if b.direction == "discharge"]
    if not ch or not dis:
        raise ValueError(f"{path}: could not find both C/10 branches")
    return [ch[-1], dis[-1]]


def main(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern}")
    si = s.placeholder_si()            # <- REPLACE with measured / extracted Si OCP
    cell = s.Cell(si)
    branch_sets = [load_rpt(f) for f in files]
    guess = dict(Q_Gr=4.6, Q_Si=0.9, Q_P=5.4, Q_Li=5.2, R=0.03)   # rough LG M50(T) values
    posts = s.fit_series(cell, branch_sets, guess, fit_R=True)
    traj = s.trajectory(posts)
    for m, arr in traj.items():
        print(m, np.round(100 * arr[:, 0], 2))
    np.savez("trajectory.npz", **traj)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/rpt_*.csv")
