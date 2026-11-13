"""Graphical abstract (reads results/12b_leave_one_out.csv and results/16a_profile_QGr.csv).

Left: median spread of each mode over the plausible OCP combinations, 0-100 % SoC cells.
Right: aged-state fit RMS when LAM_Gr is fixed, for one OCP pair and cells B and D.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

BLUE, ORANGE, MUTED, GRID = "#2a78d6", "#eb6834", "#52514e", "#e4e3df"
MODES = [("LLI", "Lithium inventory"), ("LAM_P", "Positive electrode"), ("LAM_Si", "Silicon"),
         ("LAM_NE", "Negative electrode"), ("LAM_Gr", "Graphite")]
PAIR = ("Wetjen", "Rehm2026")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.titlesize": 11.5,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False})


def main(path="figures/GA_ocp_sensitivity.png"):
    loo = pd.read_csv("results/12b_leave_one_out.csv").set_index("removed").loc["none"]
    prof = pd.read_csv("results/16a_profile_QGr.csv")
    prof = prof[(prof.graphite == PAIR[0]) & (prof.silicon == PAIR[1])]

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained",
                               gridspec_kw={"width_ratios": [1, 1.12], "wspace": 0.06})
    spread = [loo[m] for m, _ in MODES]
    a.barh(range(len(MODES)), spread, height=0.55, color=[BLUE, BLUE, ORANGE, ORANGE, ORANGE])
    for i, v in enumerate(spread):
        a.text(v + 0.3, i, f"{v:.1f}", va="center")
    a.set_yticks(range(len(MODES)), [lab for _, lab in MODES], color="black")
    a.invert_yaxis()
    a.set_xlim(0, 17)
    a.set_xticks(range(0, 15, 2))
    a.xaxis.grid(True, color=GRID)
    a.set_axisbelow(True)
    a.text(4.2, 0.5, "robust to OCP choice", va="center")
    a.text(9.6, 2.0, "sensitive to OCP choice", va="center")
    a.set_xlabel("spread over OCP choices (percentage points,\nmedian of 8 cells)")
    a.set_title("Spread of each loss over 50 literature OCP pairs", loc="left")

    b.axvspan(-20, 0, color="#ecebe7", zorder=0)
    for cell, col in (("B", BLUE), ("D", ORANGE)):
        g = prof[prof.cell == cell].sort_values("LAM_Gr")
        b.plot(g.LAM_Gr, g.rms_mV, color=col, lw=2.4)
    b.text(-12, 5.75, "cell B")
    b.text(-12, 7.8, "cell D")
    b.text(-19, 12.2, "graphite gain\n(unphysical)", color=MUTED, fontsize=11, va="center")
    b.set_xlim(-20, 40)
    b.set_ylim(4, 13)
    b.grid(True, color=GRID)
    b.set_axisbelow(True)
    b.set_xlabel("graphite loss fixed in the fit (%)")
    b.set_ylabel("fit residual, RMS (mV)")
    b.set_title("One OCP pair: graphite loss bounded only from above", loc="left")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
