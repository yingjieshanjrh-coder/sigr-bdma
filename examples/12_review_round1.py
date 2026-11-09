"""Robustness analyses for the OCP-sensitivity study.

A. Known-truth OCP-mismatch test (cell D capacities): synthetic C/10 discharge curves at
   RPT0/RPT15 are generated with one OCP combination ("truth", 1 mV noise, measured R) and
   fitted with all 50 combinations. Reports the error in each mode and whether the
   lowest-RMS fit recovers the true combination.
B. Leave-one-curve-out: spread of each mode when each graphite or Si curve is removed.
C. Rank correlation between fit RMS and LAM_Gr across combinations, per cell.
D. Full RPT0..RPT15 trajectory of LAM_Gr and LAM_Si for cell D, for the plausible combinations.

usage: python 12_review_round1.py "<inner 'Battery Degradation Data'>" "<TUM DMA repo>"
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import sigr_bdma as s
from sigr_bdma.datasets import (interpolate_resistance, kirkaldy2022_branches,
                                kirkaldy2022_resistance)
from sigr_bdma.inference import CAPACITY_NAMES, Problem, default_params
from sigr_bdma.ocp_library import load_tum_ocps

MODES = s.MODE_NAMES
STARTS = [dict(Q_Gr=qg, Q_Si=qs, Q_P=qp, Q_Li=ql)
          for qg, qs in ((4.3, 0.6), (3.8, 1.0)) for qp in (5.6, 7.7) for ql in (5.2, 7.0)]


def library(tum):
    lib = load_tum_ocps(tum)
    grs = {"Ai2020": s.graphite_ai2020}
    grs.update({n: d["delith"] for n, d in lib["graphite"].items() if "delith" in d})
    sis = {"Verbrugge2016": s.si_verbrugge2016()}
    sis.update({n: s.SiOCP(n, d["lith"], d["delith"]) for n, d in lib["silicon"].items()
                if "lith" in d and "delith" in d})
    return grs, sis


def best_fit(cell, br, R, starts):
    best = None
    for g in starts:
        p = Problem(cell, [br], default_params(g, fit_R=False), fixed={"R": R})
        try:
            z = p.fit_map()
        except Exception:
            continue
        rr = p._branch_residuals(z)[0]
        if rr is None:
            continue
        rms = float(np.sqrt(np.mean(rr[0] ** 2)))
        if best is None or rms < best[0]:
            best = (rms, z[:4])
    return best


def plausible(r, thr=10.0):
    r = r.assign(rms=0.5 * (r.rms0_mV + r.rms1_mV))
    return r[(r[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (r.LAM_Si < 95) & (r.rms <= thr)]


# --------------------------------------------------------------------------- A
def known_truth(root, tum, n_truth=8, seed=0):
    grs, sis = library(tum)
    res = pd.read_csv("results/08_ocp_sensitivity_all.csv")
    d = plausible(res[res.cell == "D"]).sort_values("rms").head(n_truth)
    idx, sets = kirkaldy2022_branches(root, "D")
    R = interpolate_resistance(kirkaldy2022_resistance(root, "D"), idx)
    R0, R1 = R[idx[0]], R[idx[-1]]
    rng = np.random.default_rng(seed)
    rows = []
    for _, t in d.iterrows():
        tcell = s.Cell(sis[t.silicon], U_gr=grs[t.graphite])
        # truth capacities: the real-data fit of cell D with this combination
        f0 = best_fit(tcell, sets[0][0], R0, STARTS)
        f1 = best_fit(tcell, sets[-1][0], R1, [dict(zip(CAPACITY_NAMES, f0[1] * k)) for k in (1, .9)])
        th0, th1 = dict(zip(CAPACITY_NAMES, f0[1])), dict(zip(CAPACITY_NAMES, f1[1]))
        try:
            b0 = tcell.simulate_branch({**th0, "R": R0}, "discharge", 0.5, (2.5, 4.17), 300, 1e-3, rng)
            b1 = tcell.simulate_branch({**th1, "R": R1}, "discharge", 0.5, (2.5, 4.17), 300, 1e-3, rng)
        except ValueError:
            continue
        true_modes = s.modes.modes_from_capacities(f0[1], f1[1]) * 100
        for (gn, gf), (sn, si) in itertools.product(grs.items(), sis.items()):
            cell = s.Cell(si, U_gr=gf)
            a = best_fit(cell, b0, R0, STARTS)
            if a is None:
                continue
            b = best_fit(cell, b1, R1, [dict(zip(CAPACITY_NAMES, a[1] * k)) for k in (1, .9, .8)])
            if b is None:
                continue
            m = s.modes.modes_from_capacities(a[1], b[1]) * 100
            row = dict(truth=f"{t.graphite}|{t.silicon}", fit=f"{gn}|{sn}",
                       is_truth=(gn == t.graphite and sn == t.silicon),
                       rms=0.5e3 * (a[0] + b[0]))
            row.update({f"err_{k}": m[i] - true_modes[i] for i, k in enumerate(MODES)})
            rows.append(row)
        print(f"  truth {t.graphite}|{t.silicon} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv("results/12a_known_truth.csv", index=False)
    print("\n[A] known-truth OCP-mismatch test (cell D capacities, 1 mV noise)")
    sel = []
    for tr, g in out.groupby("truth"):
        g = g.sort_values("rms")
        top = g.iloc[0]
        near = g[g.rms <= g.rms.min() + 1.0]
        sel.append(dict(truth=tr, best_is_truth=bool(top.is_truth), rank_truth=int(np.where(g.is_truth.values)[0][0]) + 1,
                        n_within_1mV=len(near),
                        gr_err_range=near.err_LAM_Gr.max() - near.err_LAM_Gr.min(),
                        max_abs_gr_err=near.err_LAM_Gr.abs().max(),
                        max_abs_lli_err=near.err_LLI.abs().max()))
    sel = pd.DataFrame(sel)
    print(sel.round(2).to_string(index=False))
    print(f"  lowest-RMS fit is the true combination in {sel.best_is_truth.mean():.0%} of cases;"
          f" among fits within 1 mV of the best, |LAM_Gr error| up to {sel.max_abs_gr_err.median():.1f} pp"
          f" (median over truths) vs |LLI error| up to {sel.max_abs_lli_err.median():.1f} pp")
    sel.to_csv("results/12a_known_truth_summary.csv", index=False)
    return sel


# --------------------------------------------------------------------------- B, C
def leave_one_out_and_correlation():
    res = pd.read_csv("results/08_ocp_sensitivity_all.csv")
    ok = plausible(res)
    rows = []
    for kind in ("graphite", "silicon"):
        for name in sorted(res[kind].unique()):
            sub = ok[ok[kind] != name]
            sp = sub.groupby("cell")[MODES].agg(lambda x: x.max() - x.min()).median()
            rows.append(dict(removed=f"{kind}:{name}", **{m: sp[m] for m in MODES}))
    full = ok.groupby("cell")[MODES].agg(lambda x: x.max() - x.min()).median()
    rows.insert(0, dict(removed="none", **{m: full[m] for m in MODES}))
    loo = pd.DataFrame(rows)
    loo.to_csv("results/12b_leave_one_out.csv", index=False)
    print("\n[B] median spread over cells (pp) with one curve removed")
    print(loo.round(1).to_string(index=False))

    cr = []
    allr = res.assign(rms=0.5 * (res.rms0_mV + res.rms1_mV))
    allr = allr[(allr[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (allr.LAM_Si < 95)]
    for c, g in allr.groupby("cell"):
        g = g[g.rms <= 15]
        rho, p = spearmanr(g.rms, g.LAM_Gr)
        cr.append(dict(cell=c, n=len(g), spearman_rho=rho, p=p))
    cr = pd.DataFrame(cr)
    cr.to_csv("results/12c_rms_vs_lamgr.csv", index=False)
    print("\n[C] Spearman correlation between fit RMS and LAM_Gr (fits <= 15 mV)")
    print(cr.round(3).to_string(index=False))
    return loo, cr


# --------------------------------------------------------------------------- D
def trajectory_cell_d(root, tum):
    grs, sis = library(tum)
    res = pd.read_csv("results/08_ocp_sensitivity_all.csv")
    d = plausible(res[res.cell == "D"])
    idx, sets = kirkaldy2022_branches(root, "D")
    R = interpolate_resistance(kirkaldy2022_resistance(root, "D"), idx)
    rows = []
    for _, t in d.iterrows():
        cell = s.Cell(sis[t.silicon], U_gr=grs[t.graphite])
        caps = []
        st = STARTS
        for k, brs in zip(idx, sets):
            f = best_fit(cell, brs[0], R[k], st)
            if f is None:
                caps = None
                break
            caps.append(f[1])
            st = [dict(zip(CAPACITY_NAMES, f[1] * q)) for q in (1.0, 0.97)]
        if caps is None:
            continue
        for k, c in zip(idx, caps):
            m = s.modes.modes_from_capacities(caps[0], c) * 100
            rows.append(dict(combo=f"{t.graphite}|{t.silicon}", rpt=k, **dict(zip(MODES, m))))
    tr = pd.DataFrame(rows)
    tr.to_csv("results/12d_trajectory_cellD.csv", index=False)
    last = tr[tr.rpt == idx[-1]]
    print(f"\n[D] cell D trajectories for {tr.combo.nunique()} plausible combinations; "
          f"RPT{idx[-1]} LAM_Gr range {last.LAM_Gr.min():.1f}-{last.LAM_Gr.max():.1f} %")
    return tr


if __name__ == "__main__":
    root, tum = sys.argv[1], sys.argv[2]
    os.makedirs("results", exist_ok=True)
    leave_one_out_and_correlation()
    trajectory_cell_d(root, tum)
    known_truth(root, tum)


# --------------------------------------------------------------------------- E
def multimodality_check(root, tum, combos=(("Wetjen", "Kunz"), ("Wetjen", "Verbrugge2016"),
                                           ("Hossain", "Kunz"))):
    """Same OCP combination, RPT15 fitted from two starting paths (directly from the RPT0
    optimum vs. sequentially through all RPTs): do near-equal RMS solutions exist with
    different graphite loss?"""
    grs, sis = library(tum)
    idx, sets = kirkaldy2022_branches(root, "D")
    R = interpolate_resistance(kirkaldy2022_resistance(root, "D"), idx)
    rows = []
    for gname, sname in combos:
        cell = s.Cell(sis[sname], U_gr=grs[gname])
        f0 = best_fit(cell, sets[0][0], R[idx[0]], STARTS)
        direct = best_fit(cell, sets[-1][0], R[idx[-1]],
                          [dict(zip(CAPACITY_NAMES, f0[1] * k)) for k in (1, .92, .85)])
        st, seq = STARTS, None
        for k, b in zip(idx, sets):
            seq = best_fit(cell, b[0], R[k], st)
            st = [dict(zip(CAPACITY_NAMES, seq[1] * q)) for q in (1.0, 0.97)]
        for path, f in (("direct", direct), ("sequential", seq)):
            m = s.modes.modes_from_capacities(f0[1], f[1]) * 100
            rows.append(dict(combo=f"{gname}|{sname}", path=path, rms_mV=1e3 * f[0],
                             **dict(zip(MODES, m))))
    out = pd.DataFrame(rows)
    out.to_csv("results/12e_multimodality_cellD.csv", index=False)
    print("\n[E] same OCP, two optimisation paths (cell D, RPT15)")
    print(out.round(2).to_string(index=False))
    return out


if __name__ == "__main__" and len(sys.argv) > 3 and sys.argv[3] == "--multimodality":
    multimodality_check(sys.argv[1], sys.argv[2])
