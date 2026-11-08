"""Robustness checks for the OCP-sensitivity study.

1. Plausibility threshold: median spread of each mode for RMS cut-offs 8/10/12/15 mV.
2. OCP spread vs noise-only Bayesian 95 % interval (example 07) per mode.
"""
import pandas as pd

MODES = ["LAM_Gr", "LAM_Si", "LAM_NE", "LAM_P", "LLI"]


def plausible(r):
    r = r.assign(rms=0.5 * (r.rms0_mV + r.rms1_mV))
    return r[(r[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (r.LAM_Si < 95)]


def threshold_table(path, label):
    ok = plausible(pd.read_csv(path))
    print(f"\n[{label}] median spread (pp) over cells with >= 3 plausible fits")
    print(f"{'RMS cut':>8s} {'cells':>5s} " + " ".join(f"{m:>7s}" for m in MODES) + "  Gr/LLI")
    for thr in (8, 10, 12, 15):
        g = ok[ok.rms <= thr]
        n = g.groupby("cell").size()
        sp = g.groupby("cell")[MODES].agg(lambda s: s.max() - s.min())[n >= 3].median()
        print(f"{thr:6d} mV {(n >= 3).sum():5d} " + " ".join(f"{sp[m]:7.1f}" for m in MODES)
              + f"  {sp['LAM_Gr'] / sp['LLI']:6.1f}")


def noise_vs_ocp():
    b = pd.read_csv("results/07_kirkaldy_measured_R.csv")
    s = pd.read_csv("results/08_ocp_sensitivity_summary.csv")
    m = b.merge(s, on="cell")
    print("\n[0-100 % SoC] OCP-induced spread vs noise-only 95 % interval (medians over cells)")
    for x in ["LAM_Gr", "LAM_Si", "LAM_NE", "LAM_P", "LLI"]:
        w = m[f"{x}_hi"] - m[f"{x}_lo"]
        print(f"  {x:7s} interval {w.median():5.1f} pp   OCP spread {m[f'{x}_span'].median():5.1f} pp"
              f"   ratio {(m[f'{x}_span'] / w).median():4.1f}")


if __name__ == "__main__":
    threshold_table("results/08_ocp_sensitivity_all.csv", "0-100 % SoC")
    threshold_table("results/08_ocp_sensitivity_030_all.csv", "0-30 % SoC")
    noise_vs_ocp()
