"""Does the positive-electrode OCP choice change the conclusions?

Cells B (10 degC) and D (25 degC), RPT0 vs last RPT, measured R. The 50 graphite/Si pairs are
fitted with three NMC811-type positive OCPs:
  Chen2020      : PyBaMM fit for the LG M50 (used in the main study)
  M50-3E        : LG M50 positive electrode, three-electrode lithiation curve
                  (pybamm-param repository, data/cathode_OCP_3_lit.csv; not redistributed)
  M36-coin      : LG M36 positive electrode, coin-cell lithiation at C/50
                  (TUM-EES collection, input_data/NMC/LGM36_PE_coin_Cathode_Lithiation_0C02.mat)
Reports, for each positive OCP, the spread of each mode over plausible fits and the median.

usage: python 13_cathode_ocp.py "<inner 'Battery Degradation Data'>" "<TUM DMA repo>" "<pybamm-param data dir>"
"""
import importlib.util
import itertools
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import interpolate_resistance, kirkaldy2022_branches, kirkaldy2022_resistance
from sigr_bdma.inference import CAPACITY_NAMES
from sigr_bdma.ocp_library import _read_tum_mat

_spec = importlib.util.spec_from_file_location("r1", os.path.join(os.path.dirname(__file__),
                                                                  "12_review_round1.py"))
r1 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(r1)


def positive_ocps(tum, pbp):
    d = pd.read_csv(os.path.join(pbp, "cathode_OCP_3_lit.csv"))
    c, v = d.iloc[:, 0].values, d.iloc[:, 1].values
    x = (c - c.min()) / (c.max() - c.min())          # lithiation fraction over the measured window
    o = np.argsort(x)
    xm, vm = x[o], np.minimum.accumulate(v[o])
    xc, vc = _read_tum_mat(os.path.join(tum, "input_data/NMC/LGM36_PE_coin_Cathode_Lithiation_0C02.mat"))
    return {"Chen2020": (s.U_nmc811, s.ocp.NMC811_RANGE),
            "M50-3E": (lambda y: np.interp(y, xm, vm), (0.0, 1.0)),
            "M36-coin": (lambda y: np.interp(y, xc, vc), (0.0, 1.0))}


def main(root, tum, pbp, cells="BD"):
    grs, sis = r1.library(tum)
    pos = positive_ocps(tum, pbp)
    starts = [dict(Q_Gr=qg, Q_Si=qs, Q_P=qp, Q_Li=ql)
              for qg, qs in ((4.3, 0.6), (3.8, 1.0)) for qp in (5.2, 6.4, 7.7) for ql in (5.0, 6.0, 7.0)]
    rows = []
    for cell_id in cells:
        idx, sets = kirkaldy2022_branches(root, cell_id)
        R = interpolate_resistance(kirkaldy2022_resistance(root, cell_id), idx)
        for pn, (upos, prange) in pos.items():
            for (gn, gf), (sn, si) in itertools.product(grs.items(), sis.items()):
                model = s.Cell(si, U_pos=upos, pos_range=prange, U_gr=gf)
                a = r1.best_fit(model, sets[0][0], R[idx[0]], starts)
                if a is None:
                    continue
                b = r1.best_fit(model, sets[-1][0], R[idx[-1]],
                                [dict(zip(CAPACITY_NAMES, a[1] * k)) for k in (1, .92, .85)])
                if b is None:
                    continue
                m = s.modes.modes_from_capacities(a[1], b[1]) * 100
                rows.append(dict(cell=cell_id, positive=pn, graphite=gn, silicon=sn,
                                 rms0_mV=1e3 * a[0], rms1_mV=1e3 * b[0], **dict(zip(s.MODE_NAMES, m))))
            print(f"cell {cell_id} {pn} done", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv("results/13_cathode_ocp.csv", index=False)
    ok = r1.plausible(res)
    print("\nplausible fits (mean RMS <= 10 mV): n, best RMS, and range of each mode (%)")
    for (c, p), g in ok.groupby(["cell", "positive"]):
        best = res[(res.cell == c) & (res.positive == p)]
        print(f"cell {c} {p:9s} n={len(g):2d} bestRMS={0.5 * (best.rms0_mV + best.rms1_mV).min():4.1f} mV  "
              + "  ".join(f"{m} {g[m].min():5.1f}-{g[m].max():5.1f}" for m in s.MODE_NAMES))
    return res


if __name__ == "__main__":
    main(*sys.argv[1:4], *(sys.argv[4:5] or ["BD"]))
