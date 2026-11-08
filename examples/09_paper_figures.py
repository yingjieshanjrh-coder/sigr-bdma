"""Figures for the OCP-sensitivity study (reads results/08_*.csv)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TEMPS = {"A": 10, "B": 10, "C": 10, "D": 25, "E": 25, "F": 40, "G": 40, "H": 40}
MODES = [("LLI", "LLI"), ("LAM_P", "LAM$_{PE}$"), ("LAM_Si", "LAM$_{Si}$"),
         ("LAM_NE", "LAM$_{NE}$"), ("LAM_Gr", "LAM$_{Gr}$")]
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans"})


def plausible(r, rms_max=10.0):
    r = r.assign(rms=0.5 * (r.rms0_mV + r.rms1_mV))
    ok = (r[["LAM_Gr", "LAM_Si", "LAM_P"]].min(axis=1) > -2) & (r.LAM_Si < 95) & (r.rms <= rms_max)
    return r[ok]


def fig_ranges(res, path):
    cells = sorted(res.cell.unique())
    fig, axes = plt.subplots(1, len(MODES), figsize=(7.2, 2.9), sharey=True)
    for ax, (m, lab) in zip(axes, MODES):
        for i, c in enumerate(cells):
            p = plausible(res[res.cell == c])
            y = len(cells) - 1 - i
            ax.plot([p[m].min(), p[m].max()], [y, y], color=BLUE, lw=3, solid_capstyle="round")
            ax.plot(res[res.cell == c][f"{m}_pub"].iloc[0], y, "D", ms=4, color=ORANGE,
                    mec="white", mew=0.8, zorder=3)
        ax.set_title(lab, color=INK)
        ax.set_xlabel("loss at RPT15 (%)")
        ax.grid(axis="x", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(cells)))
    axes[0].set_yticklabels([f"{c} ({TEMPS[c]} °C)" for c in reversed(cells)])
    h1, = axes[0].plot([], [], color=BLUE, lw=3)
    h2, = axes[0].plot([], [], "D", ms=4, color=ORANGE)
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
    print("written figures/fig_mode_ranges.png, figures/fig_rms_vs_lam_gr.png")
