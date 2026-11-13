"""Figure of the OCP library: the five graphite and ten silicon delithiation curves
used in the sensitivity study (examples/08_ocp_sensitivity.py).

usage: python 18_ocp_library_figure.py "<TUM DMA repo>" [output stem]
Writes <stem>.png and <stem>.pdf (default figures/fig_ocp_library) and prints the spread
between the curves at equal lithiation fraction.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sigr_bdma as s
from sigr_bdma.ocp_library import load_tum_ocps

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#8e5bd0", "#c79a00",
          "#d9488a", "#3aa7c9", "#7a7a75", "#a3542a", "#0b0b0b"]
STYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1, 1, 1))]
LABEL = {"Ai2020": "Ai 2020", "Rehm2025": "Rehm 2025", "Rehm2026": "Rehm 2026",
         "Verbrugge2016": "Verbrugge 2016"}
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "font.family": "DejaVu Sans", "pdf.fonttype": 42})


def library(tum):
    lib = load_tum_ocps(tum)
    gr = {"Ai2020": s.graphite_ai2020}
    gr.update({n: d["delith"] for n, d in lib["graphite"].items() if "delith" in d})
    si = {"Verbrugge2016": s.si_verbrugge2016().delith}
    si.update({n: d["delith"] for n, d in lib["silicon"].items() if "lith" in d and "delith" in d})
    return dict(sorted(gr.items())), dict(sorted(si.items()))


def spread(curves, x):
    u = np.array([f(x) for f in curves.values()])
    return 1e3 * (u.max(axis=0) - u.min(axis=0))


def main(tum, stem="figures/fig_ocp_library"):
    gr, si = library(tum)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    x = np.linspace(0.005, 0.995, 600)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, curves, title, ymax in ((axes[0], gr, "graphite", 0.5), (axes[1], si, "silicon", 1.0)):
        for k, (name, f) in enumerate(curves.items()):
            ax.plot(x, f(x), color=COLORS[k % len(COLORS)], ls=STYLES[k % len(STYLES)], lw=1.3,
                    label=LABEL.get(name, name))
        ax.set_title(title, color=INK)
        ax.set_xlabel("lithiation fraction $x$")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, ymax)
        ax.grid(color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=6.5, ncol=2 if len(curves) > 5 else 1, handlelength=2.6)
    axes[0].set_ylabel("delithiation OCP vs Li/Li$^+$ (V)")
    for ax, letter in zip(axes, "ab"):
        ax.text(0, 1.035, f"({letter})", transform=ax.transAxes, fontsize=10, color="black",
                ha="left", va="bottom")
    fig.tight_layout()
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=300)
    plt.close(fig)

    xm = np.linspace(0.1, 0.9, 401)
    for name, curves in (("graphite", gr), ("silicon", si)):
        sp = spread(curves, xm)
        print(f"{name}: {len(curves)} curves {list(curves)}; spread over x = 0.1-0.9: "
              f"median {np.median(sp):.0f} mV, min {sp.min():.0f} mV, max {sp.max():.0f} mV")


if __name__ == "__main__":
    main(*sys.argv[1:3])
