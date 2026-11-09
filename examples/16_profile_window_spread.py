"""Profile likelihood, voltage-window sensitivity and sample-size-robust spread statistics.

A. Profile of the aged-state fit RMS over Q_Gr (cell D and B, best-fitting OCP pairs):
   how far can LAM_Gr move before the fit gets visibly worse?
B. Voltage-window sensitivity: refit all 50 OCP pairs for cells B, D, E using only
   3.0-4.2 V and only 3.2-4.1 V, which removes the regions with the largest
   polarisation at C/10.
C. Spread statistics that do not depend on the number of plausible combinations:
   inter-quartile range, standard deviation and the range rarefied to k = 5 combinations.

usage: python 16_profile_window_spread.py "<inner 'Battery Degradation Data'>" "<TUM DMA repo>" [A|B|C]
"""
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import interpolate_resistance, kirkaldy2022_branches, kirkaldy2022_resistance
from sigr_bdma.ocp_library import load_tum_ocps
from sigr_bdma.sensitivity import (crop_branch, ensemble_summary, fit_pair, ocp_ensemble,
                                   plausible, profile_capacity, rarefied_range, spread_stats)

MODES = s.MODE_NAMES


def library(tum):
    lib = load_tum_ocps(tum)
    grs = {"Ai2020": s.graphite_ai2020}
    grs.update({n: d["delith"] for n, d in lib["graphite"].items() if "delith" in d})
    sis = {"Verbrugge2016": s.si_verbrugge2016()}
    sis.update({n: s.SiOCP(n, d["lith"], d["delith"]) for n, d in lib["silicon"].items()
                if "lith" in d and "delith" in d})
    return grs, sis


def cell_data(root, cell, soc="0-100% SoC"):
    idx, sets = kirkaldy2022_branches(root, cell, soc)
    R = interpolate_resistance(kirkaldy2022_resistance(root, cell, soc), idx)
    return sets[0][0], sets[-1][0], R[idx[0]], R[idx[-1]]


# --------------------------------------------------------------------------- A
def profiles(root, tum, cells=("D", "B"), n_pairs=3):
    grs, sis = library(tum)
    allres = pd.read_csv("results/08_ocp_sensitivity_all.csv")
    rows = []
    for c in cells:
        fr, ag, R0, R1 = cell_data(root, c)
        best = plausible(allres[allres.cell == c]).sort_values("rms").head(n_pairs)
        for _, t in best.iterrows():
            cell = s.Cell(sis[t.silicon], U_gr=grs[t.graphite])
            (r0, z0), (r1, z1) = fit_pair(cell, fr, ag, R0, R1)
            grid = z0[0] * (1 - np.linspace(-0.20, 0.40, 31))      # LAM_Gr from -20 to 40 %
            prof = profile_capacity(cell, ag, R1, z1, "Q_Gr", grid, starts_scale=(1.0, 0.9, 1.1))
            for _, p in prof.iterrows():
                lam = 1 - np.array([p.Q_Gr, p.Q_Si, p.Q_P, p.Q_Li]) / z0
                rows.append(dict(cell=c, graphite=t.graphite, silicon=t.silicon,
                                 rms_best_mV=1e3 * r1, LAM_Gr=100 * lam[0], LAM_Si=100 * lam[1],
                                 LAM_P=100 * lam[2], LLI=100 * lam[3], rms_mV=p.rms_mV))
            print(f"cell {c} {t.graphite:10s} {t.silicon:14s} best aged RMS {1e3 * r1:.2f} mV")
    prof = pd.DataFrame(rows)
    prof.to_csv("results/16a_profile_QGr.csv", index=False)
    out = []
    for (c, g, si), d in prof.groupby(["cell", "graphite", "silicon"]):
        rmin = d.rms_mV.min()
        for dv in (0.25, 0.5, 1.0):
            ok = d[d.rms_mV <= rmin + dv]
            out.append(dict(cell=c, graphite=g, silicon=si, rms_min_mV=rmin, delta_mV=dv,
                            LAM_Gr_lo=ok.LAM_Gr.min(), LAM_Gr_hi=ok.LAM_Gr.max(),
                            LLI_lo=ok.LLI.min(), LLI_hi=ok.LLI.max(),
                            LAM_Si_lo=ok.LAM_Si.min(), LAM_Si_hi=ok.LAM_Si.max()))
    out = pd.DataFrame(out)
    out.to_csv("results/16a_profile_summary.csv", index=False)
    print(out.round(2).to_string(index=False))
    return out


# --------------------------------------------------------------------------- B
WINDOWS = {"2.5-4.2": (None, None), "3.0-4.2": (3.0, None), "3.2-4.1": (3.2, 4.1)}


def windows(root, tum, cells=("B", "D", "E")):
    grs, sis = library(tum)
    allres = []
    for c in cells:
        fr, ag, R0, R1 = cell_data(root, c)
        for wn, (lo, hi) in WINDOWS.items():
            if wn == "2.5-4.2":
                r = pd.read_csv("results/08_ocp_sensitivity_all.csv")
                r = r[r.cell == c].copy()
            else:
                r = ocp_ensemble(grs, sis, crop_branch(fr, lo, hi), crop_branch(ag, lo, hi), R0, R1)
            r["cell"], r["window"] = c, wn
            allres.append(r)
            summ = ensemble_summary(r)
            print(f"cell {c} window {wn}: n={int(summ.n.iloc[0])}  " + "  ".join(
                f"{m}={v:.1f}" for m, v in zip(summ["mode"], summ["range"])))
    res = pd.concat(allres, ignore_index=True)
    res.to_csv("results/16b_window.csv", index=False)
    rows = []
    for (c, wn), d in res.groupby(["cell", "window"]):
        p = plausible(d)
        row = dict(cell=c, window=wn, n=len(p), best_rms=float(0.5 * (d.rms0_mV + d.rms1_mV).min()))
        for m in MODES:
            row[f"{m}_range"] = np.ptp(p[m]) if len(p) else np.nan
            row[f"{m}_median"] = p[m].median() if len(p) else np.nan
        rows.append(row)
    summ = pd.DataFrame(rows)
    summ.to_csv("results/16b_window_summary.csv", index=False)
    print(summ.round(1).to_string(index=False))
    return summ


# --------------------------------------------------------------------------- C
def robust_spread(k=5):
    rows = []
    for tag, f in (("0-100", "results/08_ocp_sensitivity_all.csv"),
                   ("0-30", "results/08_ocp_sensitivity_030_all.csv")):
        res = pd.read_csv(f)
        for c, g in res.groupby("cell"):
            p = plausible(g)
            for m in MODES:
                st = spread_stats(p[m])
                rows.append(dict(soc=tag, cell=c, mode=m, **st,
                                 range_k5=rarefied_range(p[m], k)))
    d = pd.DataFrame(rows)
    d.to_csv("results/16c_robust_spread.csv", index=False)
    med = d.groupby(["soc", "mode"])[["range", "iqr", "sd", "range_k5"]].median()
    med = med.reindex(pd.MultiIndex.from_product([["0-100", "0-30"], MODES]))
    print("median over cells (pp); range_k5 only for cells with >= 5 plausible combinations")
    print(med.round(1).to_string())
    med.round(2).to_csv("results/16c_robust_spread_median.csv")
    return med


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    what = sys.argv[3] if len(sys.argv) > 3 else "ABC"
    if "C" in what:
        robust_spread()
    if "A" in what:
        profiles(sys.argv[1], sys.argv[2])
    if "B" in what:
        windows(sys.argv[1], sys.argv[2])
