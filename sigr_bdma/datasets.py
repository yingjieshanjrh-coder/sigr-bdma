"""Loaders for public datasets (data are not redistributed; point to your download)."""
from __future__ import annotations

import glob
import os
import re

import pandas as pd

import numpy as np

from .io import branch_from_arrays, pulse_resistance


def _cell_dir(base, cell):
    for name in (f"Cell {cell}", f"cell {cell}"):
        if os.path.isdir(os.path.join(base, name)):
            return os.path.join(base, name)
    return os.path.join(base, f"Cell {cell}")


def kirkaldy2022_branches(root, cell, soc="0-100% SoC", n=300):
    """LG M50T dataset of Kirkaldy et al. (Zenodo 10.5281/zenodo.7235857).
    root = the inner 'Battery Degradation Data' folder. Returns (rpt_indices, branch_sets);
    the dataset holds C/10 DISCHARGE curves only."""
    d = _cell_dir(os.path.join(root, "Timeseries Data from RPTs", soc, "0.1C Voltage Curves"), cell)
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
KIRKALDY_TEMPS_030 = {"A": 10, "B": 10, "J": 10, "D": 25, "E": 25, "F": 25, "K": 40, "L": 40, "M": 40}


def kirkaldy2022_resistance(root, cell, soc="0-100% SoC"):
    """Median instantaneous resistance (ohm) per RPT from the 5-pulse GITT files
    (0.5C pulses from rest). Returns dict rpt -> R. RPTs without GITT are not included;
    see `interpolate_resistance`."""
    d = _cell_dir(os.path.join(root, "Timeseries Data from RPTs", soc, "GITT Voltage Curves"), cell)
    out = {}
    for f in glob.glob(os.path.join(d, "*5-pulse GITT*.csv")):
        if "25-pulse" in f:
            continue
        df = pd.read_csv(f)
        p = pulse_resistance(df["Time (s)"], df["Current (mA)"] / 1000, df["Voltage (V)"])
        p = p[p[:, 2] > 0]
        if len(p):
            out[int(re.search(r"RPT(\d+)", f).group(1))] = float(np.median(p[:, 2]))
    return dict(sorted(out.items()))


def interpolate_resistance(R_by_rpt, rpts):
    k = np.array(list(R_by_rpt)); v = np.array(list(R_by_rpt.values()))
    return {r: float(np.interp(r, k, v)) for r in rpts}


# ---------------------------------------------------------------------------
# Molicel P45B test data of the TUM-EES DegradationModeAnalysis repository
# (input_data/test_data/P45B_serial23_aging_data_table.mat, a MATLAB table object).
# ---------------------------------------------------------------------------
def _matlab_table_objects(path):
    """Decode the MCOS object store of a MATLAB table saved with -v7 (no MATLAB needed).
    Returns the list of object-metadata cells."""
    import io

    import scipy.io as sio
    from scipy.io.matlab._mio5 import MatFile5Reader
    raw = sio.loadmat(path)["__function_workspace__"].tobytes()
    hdr = b"MATLAB 5.0 MAT-file".ljust(116, b" ") + b"\x00" * 8 + b"\x00\x01IM"
    r = MatFile5Reader(io.BytesIO(hdr + raw[8:]), struct_as_record=False, squeeze_me=False)
    r.mat_stream.seek(0)
    r.initialize_read()
    r.mat_stream.seek(128)
    h, _ = r.read_var_header()
    top = r.read_var_array(h, process=True)[0, 0]
    return list(np.asarray(top.MCOS[0]["_ObjectMetadata"]).ravel())


def tum_p45b_branches(repo_path, n=300):
    """List (one per check-up) of {"charge": Branch, "discharge": Branch} for the P45B
    cell (C/30 pOCV in both directions, 9 check-ups, 0-800 EFC)."""
    cells = _matlab_table_objects(os.path.join(
        repo_path, "input_data", "test_data", "P45B_serial23_aging_data_table.mat"))
    curves = []
    for c in cells:
        a = np.asarray(c)
        if a.dtype != object or a.size != 1:
            continue
        o = a.ravel()[0]
        if not hasattr(o, "_fieldnames") or "data" not in o._fieldnames:
            continue
        U, I, Ah = (np.asarray(x).ravel() for x in o.data.ravel())
        curves.append((U, I, Ah))
    ch = [c for c in curves if c[1].mean() > 0]
    dch = [c for c in curves if c[1].mean() < 0]
    out = []
    for (Uc, Ic, Ac), (Ud, Id, Ad) in zip(ch, dch):
        out.append({"charge": branch_from_arrays(np.abs(Ac), Uc, "charge", abs(Ic.mean()), n),
                    "discharge": branch_from_arrays(np.abs(Ad), Ud, "discharge", abs(Id.mean()), n)})
    return out


def tum_p45b_published(repo_path, cu=9):
    """Degradation modes (%) reported by the TUM-EES framework for the P45B test cell
    (charge-direction fit) at check-up `cu` relative to CU1."""
    import scipy.io as sio
    d = sio.loadmat(os.path.join(repo_path, "input_data", "test_data", "results",
                                 "P45B_parameters.mat"), squeeze_me=True, struct_as_record=False)
    c = getattr(d["data"], f"CU{cu}")
    return {"LAM_Gr": 100 * float(c.lamAnodeBlend1), "LAM_Si": 100 * float(c.lamAnodeBlend2),
            "LAM_NE": 100 * float(c.lamAnode), "LAM_P": 100 * float(c.lamCathode),
            "LLI": 100 * float(c.li)}
