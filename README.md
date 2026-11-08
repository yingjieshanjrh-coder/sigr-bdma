# sigr-bdma

**Bayesian degradation-mode analysis for lithium-ion cells with silicon–graphite blend anodes.**

Degradation-mode analysis (DMA) splits capacity fade into loss of lithium inventory (LLI) and loss of active material (LAM) by fitting low-rate full-cell voltage curves with electrode open-circuit potentials (OCPs). Blend anodes make this hard. Silicon and graphite age at different rates. Silicon has a large lithiation/delithiation hysteresis. And the silicon OCP reported in the literature varies a lot, which changes the answer. `sigr-bdma` handles these problems in a fully Bayesian way:

| Feature | What it does |
|---|---|
| Branch-aware Si OCP | separate lithiation/delithiation Si curves; the charge and discharge branches are fitted **jointly** |
| Ohmic term inferred | the series resistance R is a parameter, so no separate pulse test is needed |
| Si-OCP uncertainty | a mixture weight `w` between two candidate Si OCPs is inferred together with the capacities; Laplace-evidence Bayesian model averaging is also available |
| Full posteriors | affine-invariant ensemble MCMC (dependency-free) gives credible intervals and correlations for LAM_Gr, LAM_Si, LAM_NE, LAM_P and LLI |
| Experimental design | a Fisher-information estimate of which protocol (branches, voltage window, R known or not) makes which mode identifiable, before any test is run |
| Si OCP from blend half-cell | derives Si branches from blend-anode and graphite half-cell data (plateau masking plus isotonic regression) |
| Cycler I/O | extracts low-rate CC segments from raw logs; template for RPT series |

## Install
```bash
pip install -e .          # numpy, scipy (pandas/matplotlib optional)
python -m pytest -q tests
```

## Quick start
```python
import sigr_bdma as s
si = s.placeholder_si()                 # replace with measured Si branches!
cell = s.Cell(si)
truth = dict(Q_Gr=5.0, Q_Si=1.0, Q_P=5.6, Q_Li=5.5, R=0.03)
branches = [cell.simulate_branch(truth, d, I=0.5, noise=1e-3) for d in ("charge", "discharge")]
post = s.fit_state(cell, branches, {k: 0.95 * v for k, v in truth.items()})
print(post.summary())
```

## Synthetic results (`results/`, reproducible with `examples/`)
These results are **synthetic** (known ground truth, an idealised model and placeholder Si curves). They show what the method can do. They are not yet validated on real cells.

**01, benchmark.** Fresh to aged; true LAM_Gr 5 %, LAM_Si 25 %, LAM_P 3 %, LLI 10 %.

| mode | truth | this work (both branches, R inferred) | conventional (discharge, mean Si OCP, no IR) | best practice (discharge, delith. Si OCP, R known) |
|---|---|---|---|---|
| LAM_Gr | 5.0 | 4.9 ± 0.2 | 0.3 ± 6.0 | 5.1 ± 0.3 |
| LAM_Si | 25.0 | 25.1 ± 0.3 | 24.4 ± 4.5 | 24.8 ± 0.5 |
| LAM_P | 3.0 | 3.0 ± 0.0 | 4.2 ± 0.7 | 2.9 ± 0.1 |
| LLI | 10.0 | 10.0 ± 0.0 | 9.8 ± 0.2 | 10.0 ± 0.0 |

When the correct branch OCP and the true R are given, a discharge-only fit is already accurate. The joint fit reaches the same accuracy with R unknown, and its uncertainties are tighter.

**02, design.** Fitting both branches cuts the relative uncertainty of Q_Si by about 2–4× compared with a single branch. It also breaks the strong Gr–Si correlation (−0.9 to −0.99 with one branch, down to −0.15 to −0.85 with both). Raising the lower cut-off to 3.4 V makes a charge-only test nearly blind to Si (σ ≈ 4.4 %).

**03, Si-OCP uncertainty.** The true Si curve lies between two candidates. Fixing either candidate biases LAM_Gr or LAM_Si and still reports small error bars. Laplace BMA puts all its weight on one candidate and inherits that bias. The mixture model recovers the weight (w = 0.354 ± 0.010, true 0.35) and the modes.

## Limitations (current version)
- Only the ohmic polarisation is modelled. Diffusion and charge-transfer overpotentials at C/10–C/20 are not, so residuals are correlated and credible intervals on real data will be too narrow unless a discrepancy term is added (planned).
- Graphite is treated as hysteresis-free. Si hysteresis is a fixed gap scaled by `hys`; there is no path-dependent (scanning-curve) model yet.
- The built-in Si curves are placeholders. The graphite and NMC811 OCPs are the Chen2020 fits (LG M50).
- Not yet validated on experimental data. The next step is the LG M50T dataset of Kirkaldy et al. (Zenodo 7235857), which has C/10 charge and discharge at every RPT.

## Related work
- Kirkaldy et al., *ACS Appl. Energy Mater.* 2022: composite-electrode OCV fitting, LG M50T dataset.
- Rehm et al., *J. Power Sources* 2026: DMA for Si–Gr blends; the Si OCP generated from blend and graphite half-cells. Their study shows modes change by more than 2× with the choice of Si OCP.
- "Phantom LAM and LLI", arXiv:2512.19773: resistance and hysteresis bias in voltage-curve DMA (deterministic fits).
- Lin & Khoo, *J. Power Sources*: Fisher-information identifiability for single-material electrodes.
- Chen et al., *J. Electrochem. Soc.* 2020: graphite and NMC811 OCP fits.

## License
MIT License. Copyright (c) 2026 Yingjie Shan.
