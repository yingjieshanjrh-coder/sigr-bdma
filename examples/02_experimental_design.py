"""Which low-rate test makes which degradation mode identifiable?

Fisher-information estimate of the relative posterior sd of each capacity for
combinations of: branches measured, lower voltage cut-off, and whether R is unknown.
Fast (no MCMC). Assumed truth = fresh cell of example 01, C/10, 1 mV noise.
"""
import itertools

import sigr_bdma as s

TRUTH = dict(Q_Gr=5.0, Q_Si=1.0, Q_P=5.6, Q_Li=5.5, R=0.030)


def main():
    si = s.placeholder_si()
    rows = []
    for brs, v_lo, fit_R in itertools.product(
            [("discharge",), ("charge",), ("charge", "discharge")],
            [2.5, 3.0, 3.4], [False, True]):
        rel, corr, names = s.evaluate_design(si, TRUTH, brs, (v_lo, 4.2), I=0.5,
                                             noise=1e-3, fit_R=fit_R)
        rows.append(("+".join(b[:3] for b in brs), v_lo, "fit" if fit_R else "known",
                     rel["Q_Gr"], rel["Q_Si"], rel["Q_P"], rel["Q_Li"],
                     corr[names.index("Q_Gr"), names.index("Q_Si")]))
    print("relative sd (%) of each capacity; corr = correlation(Q_Gr, Q_Si)")
    print(f"{'branches':10s} {'V_lo':>5s} {'R':>6s} {'Q_Gr':>8s} {'Q_Si':>8s} {'Q_P':>8s} "
          f"{'Q_Li':>8s} {'corr':>6s}")
    for r in rows:
        print(f"{r[0]:10s} {r[1]:5.1f} {r[2]:>6s} " + " ".join(f"{100 * v:8.3f}" for v in r[3:7])
              + f" {r[7]:6.2f}")
    return rows


if __name__ == "__main__":
    main()
