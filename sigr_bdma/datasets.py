"""Loaders for public datasets (data are not redistributed; point to your download)."""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

from .io import branch_from_arrays


def kirkaldy2022_branches(root, cell, soc="0-100% SoC", n=300):
    """LG M50T dataset of Kirkaldy et al. (Zenodo 10.5281/zenodo.7235857).
    root = the inner 'Battery Degradation Data' folder. Returns (rpt_indices, branch_sets);
    the dataset holds C/10 DISCHARGE curves only."""
    d = os.path.join(root, "Timeseries Data from RPTs", soc, "0.1C Voltage Curves", f"Cell {cell}")
    files = glob.glob(os.path.join(d, "*0.1C discharge data.csv"))
    files.sort(key=lambda f: int(re.search(r"RPT(\d+)", f).group(1)))
    idx, sets = [], []
    for f in files:
        df = pd.read_csv(f)
        I = abs(df["Current (mA)"].mean()) / 1000
        k = int(re.search(r"RPT(\d+)", f).group(1))
        idx.append(k)
        sets.append([branch_from_arrays(df["Charge (mA.h)"] / 1000, df["Voltage (V)"],
                                        "discharge", I, n=n, label=f"RPT{k}")])
    return idx, sets


def kirkaldy2022_published_modes(root, cell, soc="0-100% SoC"):
    """Authors' least-squares degradation modes (fractions), indexed by RPT."""
    f = glob.glob(os.path.join(root, "Extracted Data - Degradation Modes", soc, f"cell {cell}",
                               "*DM data*.csv"))[0]
    return pd.read_csv(f, index_col=0)


KIRKALDY_COLMAP = {"LAM_Gr": "LAM NE_Gr", "LAM_Si": "LAM NE_Si", "LAM_NE": "LAM NE_tot",
                   "LAM_P": "LAM PE", "LLI": "LLI"}
KIRKALDY_TEMPS = {"A": 10, "B": 10, "C": 10, "D": 25, "E": 25, "F": 40, "G": 40, "H": 40}
