"""Figures for the OCP-sensitivity study (reads results/08_*.csv)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEMPS = {"A": 10, "B": 10, "C": 10, "D": 25, "E": 25, "F": 40, "G": 40, "H": 40}
TEMPS_030 = {"A": 10, "B": 10, "J": 10, "D": 25, "E": 25, "F": 25, "K": 40, "L": 40, "M": 40}
MODES = [("LLI", "LLI"), ("LAM_P", "LAM$_{PE}$"), ("LAM_Si", "LAM$_{Si}$"),
         ("LAM_NE", "LAM$_{NE}$"), ("LAM_Gr", "LAM$_{Gr}$")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def plausible(r, rms_max=10.0):
    r = r.assign(rms=0.5 * (r.rms0_mV + r.rms1_mV))
    ok = (r[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (r.LAM_Si < 95) & (r.rms <= rms_max)
    return r[ok]


def fig_ranges(res, path, temps=TEMPS):
    cells = sorted(res.cell.unique(), key=lambda c: (temps[c], c))
    fig, axes = plt.subplots(1, len(MODES), figsize=(7.2, 2.9), sharey=True)
    for ax, (m, lab) in zip(axes, MODES):
        for i, c in enumerate(cells):
            p = plausible(res[res.cell == c])
            y = len(cells) - 1 - i
            ax.plot([p[m].min(), p[m].max()], [y, y], color=BLUE, lw=3, solid_capstyle="round")
            ax.plot(res[res.cell == c][f"{m}_pub"].iloc[0], y, "D", ms=4.5, mfc="none", mec=ORANGE, mew=1.3, zorder=3)
        ax.set_title(lab, color=INK)
        ax.set_xlabel("loss at last RPT (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(cells)))
    axes[0].set_yticklabels([f"{c} ({temps[c]} °C)" for c in reversed(cells)])
    h1, = axes[0].plot([], [], color=BLUE, lw=3)
    h2, = axes[0].plot([], [], "D", ms=4.5, mfc="none", mec=ORANGE, mew=1.3)
    fig.legend([h1, h2], ["range over OCP choices (fits ≤ 10 mV)", "published (least squares)"],
               frameon=False, loc="upper center", ncol=2, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=300)
    plt.close(fig)


def fig_rms_vs_gr(res, path, cells=("B", "D")):
    fig, axes = plt.subplots(1, len(cells), figsize=(6.0, 2.6), sharey=True)
    for ax, c in zip(axes, cells):
        r = res[res.cell == c]
        r = r.assign(rms=0.5 * (r.rms0_mV + r.rms1_mV))
        r = r[(r.LAM_Gr > -5) & (r.LAM_Gr < 40) & (r.LAM_Si < 95)]
        ax.scatter(r.rms, r.LAM_Gr, s=18, color=BLUE, edgecolor="white", lw=0.6, zorder=2)
        ax.axhline(r.LAM_Gr_pub.iloc[0], color=ORANGE, lw=1.2, ls="--", label="published")
        ax.set_title(f"cell {c} ({TEMPS[c]} °C)", color=INK)
        ax.set_xlabel("mean fit RMS (mV)")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("LAM$_{Gr}$ at RPT15 (%)")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    res = pd.read_csv("results/08_ocp_sensitivity_all.csv")
    os.makedirs("figures", exist_ok=True)
    fig_ranges(res, "figures/fig_mode_ranges.png")
    fig_rms_vs_gr(res, "figures/fig_rms_vs_lam_gr.png")
    if os.path.exists("results/08_ocp_sensitivity_030_all.csv"):
        r030 = pd.read_csv("results/08_ocp_sensitivity_030_all.csv")
        fig_ranges(r030, "figures/fig_mode_ranges_030.png", TEMPS_030)
    print("figures written to figures/")


def fig_trajectory(path_csv="results/12d_trajectory_cellD.csv", path="figures/fig_trajectory_cellD.png"):
    tr = pd.read_csv(path_csv)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharex=True)
    for ax, (m, lab) in zip(axes, [("LLI", "LLI"), ("LAM_Gr", "LAM$_{Gr}$"), ("LAM_Si", "LAM$_{Si}$")]):
        for _, g in tr.groupby("combo"):
            ax.plot(g.rpt, g[m], color=BLUE, lw=1.0, alpha=0.55)
        ax.set_title(lab, color=INK)
        ax.set_xlabel("RPT")
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("loss relative to RPT0 (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__" and os.path.exists("results/12d_trajectory_cellD.csv"):
    fig_trajectory()


AQUA = "#1baf7a"


def fig_profile(path_csv="results/16a_profile_QGr.csv", path="figures/fig_profile_QGr.png"):
    """Aged-state fit RMS when LAM_Gr is fixed and the other capacities are refitted."""
    d = pd.read_csv(path_csv)
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.6), sharey=True)
    styles = {"Kunz": (BLUE, "-"), "Rehm2026": (ORANGE, "--"), "Verbrugge2016": (AQUA, ":")}
    for ax, c in zip(axes, ("B", "D")):
        g = d[d.cell == c]
        ax.axvspan(-21, 0, color=GRID, alpha=0.6, lw=0)
        ax.text(-10, 13.6, "graphite gain\n(unphysical)", ha="center", va="top", color=MUTED, fontsize=7)
        for si, (col, ls) in styles.items():
            h = g[g.silicon == si].sort_values("LAM_Gr")
            ax.plot(h.LAM_Gr, h.rms_mV, color=col, ls=ls, lw=1.6, label=f"Wetjen Gr + {si} Si")
        ax.set_xlim(-21, 32)
        ax.set_ylim(0, 14)
        ax.set_xlabel("LAM$_{Gr}$ fixed at (%)")
        ax.set_title(f"cell {c} ({TEMPS[c]} °C)", color=INK)
        ax.grid(color=GRID, lw=0.6)
    axes[0].set_ylabel("aged-state fit RMS (mV)")
    axes[0].legend(frameon=False, fontsize=7, loc="lower right")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


if __name__ == "__main__" and os.path.exists("results/16a_profile_QGr.csv"):
    fig_profile()
