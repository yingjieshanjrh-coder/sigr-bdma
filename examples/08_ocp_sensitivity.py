"""How much do the inferred degradation modes depend on the choice of component OCPs?

LG M50T cell (Kirkaldy et al. 2022), C/10 discharge at the first and last RPT,
ohmic R fixed at the GITT-measured value. Every combination of
  graphite delithiation OCP : Ai2020 (PyBaMM) + 4 from the TUM-EES collection
  Si delithiation OCP       : Verbrugge2016 (PyBaMM) + 9 from the TUM-EES collection
is fitted (multi-start MAP). Reports, per combination, fit RMS and the modes, and the
spread of each mode across all combinations and across the well-fitting ones.

usage: python 08_ocp_sensitivity.py "<inner 'Battery Degradation Data'>" "<TUM DMA repo>" [cell]
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

import sigr_bdma as s
from sigr_bdma.datasets import (KIRKALDY_COLMAP, interpolate_resistance, kirkaldy2022_branches,
                                kirkaldy2022_published_modes, kirkaldy2022_resistance)
from sigr_bdma.inference import Problem, default_params
from sigr_bdma.ocp_library import load_tum_ocps

STARTS = [dict(Q_Gr=qg, Q_Si=qs, Q_P=qp, Q_Li=ql)
          for qg, qs in ((4.3, 0.6), (3.8, 1.0)) for qp in (5.6, 6.6, 7.7) for ql in (5.2, 6.2, 7.0)]


def best_map(cell, branch, R, starts):
    best = None
    for g in starts:
        prob = Problem(cell, [branch], default_params(g, fit_R=False), fixed={"R": R})
        try:
            z = prob.fit_map()
        except Exception:
            continue
        rr = prob._branch_residuals(z)[0]
        if rr is None:
            continue
        rms = float(np.sqrt(np.mean(rr[0] ** 2)))
        if best is None or rms < best[0]:
            best = (rms, z[:4], prob)
    return best


def main(root, tum, cell="D"):
    idx, sets = kirkaldy2022_branches(root, cell)
    R = interpolate_resistance(kirkaldy2022_resistance(root, cell), idx)
    first, last = idx[0], idx[-1]
    br0, br1 = sets[0][0], sets[-1][0]
    lib = load_tum_ocps(tum)
    graphites = {"Ai2020": s.graphite_ai2020}
    graphites.update({n: d["delith"] for n, d in lib["graphite"].items() if "delith" in d})
    sis = {"Verbrugge2016": s.si_verbrugge2016()}
    sis.update({n: s.SiOCP(n, d["lith"], d["delith"]) for n, d in lib["silicon"].items()
                if "lith" in d and "delith" in d})
    pub = kirkaldy2022_published_modes(root, cell)

    rows = []
    for (gn, gf), (sn, si) in itertools.product(graphites.items(), sis.items()):
        model = s.Cell(si, U_gr=gf)
        b0 = best_map(model, br0, R[first], STARTS)
        if b0 is None:
            print(f"{gn:10s} {sn:14s} fresh fit failed"); continue
        z0 = b0[1]
        starts1 = [dict(zip(s.inference.CAPACITY_NAMES, z0 * f)) for f in (1.0, 0.92, 0.85)]
        b1 = best_map(model, br1, R[last], starts1)
        if b1 is None:
            print(f"{gn:10s} {sn:14s} aged fit failed"); continue
        m = s.modes.modes_from_capacities(z0, b1[1])
        row = dict(graphite=gn, silicon=sn, rms0_mV=1e3 * b0[0], rms1_mV=1e3 * b1[0],
                   Si_share0=z0[1] / (z0[0] + z0[1]))
        row.update({n: 100 * v for n, v in zip(s.MODE_NAMES, m)})
        rows.append(row)
        print(f"{gn:10s} {sn:14s} rms {row['rms0_mV']:5.1f}/{row['rms1_mV']:5.1f} mV  "
              + "  ".join(f"{n}={row[n]:5.1f}" for n in s.MODE_NAMES))
    res = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    res.to_csv(f"results/08_ocp_sensitivity_cell{cell}.csv", index=False)

    res["rms"] = 0.5 * (res.rms0_mV + res.rms1_mV)
    good = res[res.rms <= 1.25 * res.rms.min()]
    print(f"\n{len(res)} OCP combinations; {len(good)} fit within 25 % of the best RMS "
          f"({res.rms.min():.1f} mV)")
    print(f"{'mode':7s} {'published':>9s} {'all: min':>9s} {'max':>6s} {'good: min':>10s} {'max':>6s}")
    for n in s.MODE_NAMES:
        p = 100 * pub[KIRKALDY_COLMAP[n]].loc[last]
        print(f"{n:7s} {p:9.1f} {res[n].min():9.1f} {res[n].max():6.1f} {good[n].min():10.1f} "
              f"{good[n].max():6.1f}")
    return res


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], *(sys.argv[3:4] or ["D"]))
