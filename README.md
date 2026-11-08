# sigr-bdma

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24047660.svg)](https://doi.org/10.5281/zenodo.24047660)

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

**05, real data: LG M50T (Kirkaldy et al. 2022, cell D, 25 °C, 0–100 % SoC).** The dataset only contains C/10 *discharge* curves, so this is a discharge-only fit (graphite OCP from Ai 2020, Si OCP from Verbrugge 2016, R inferred). Results are in `results/05_kirkaldy_cellD.csv`. The raw data are not redistributed; download them from Zenodo 10.5281/zenodo.7235857.

| at RPT15 | this work (median [95 % CI]) | published (least squares) |
|---|---|---|
| LLI | 16.0 [15.3, 16.7] | 17.1 |
| LAM_P | 8.5 [7.4, 9.7] | 7.7 |
| LAM_NE | 18.5 [17.8, 19.2] | 21.5 |
| LAM_Gr | 14.9 [13.8, 15.9] | 19.8 |
| LAM_Si | 42.5 [40.4, 44.5] | 31.2 |

LLI and LAM_P agree with the published analysis. How the anode loss is split between graphite and Si does **not** agree, and the credible intervals do not cover the published values. The fit residuals are 7–9 mV RMS and systematic. Together this shows that the Gr/Si split is governed by the choice of component OCPs and by model discrepancy, not by measurement noise. The intervals shown above reflect noise only and are therefore too narrow. The next step is to add a model-discrepancy term and Si-OCP mixtures.

**06, real data: all eight 0–100 % SoC cells (10/25/40 °C, RPT 0/3/6/9/12/15).** Results are in `results/06_kirkaldy_all_cells.csv`.
- With independent (iid) noise, LAM_P agrees with the published values for 7 of 8 cells. LLI agrees to within about 1–2.5 pp, but the published value lies inside our 95 % interval for none of the cells.
- LAM_Si comes out **10–15 pp higher than published in every cell** (42–49 % vs 30–42 %). That is a systematic offset, not scatter, and is consistent with the choice of Si/graphite OCP setting the Gr/Si split.
- The fit residuals are 9–10 mV RMS. An AR(1) discrepancy model (`noise="ar1"`) drives φ to its upper bound (≈ 0.994). It widens the LAM_Si intervals (median width 4 → 14 pp) but also produces an artefact, LAM_P ≈ 20 % in every cell. A simple AR(1) term is therefore **not** an adequate discrepancy model for these data. It is kept as an experimental option.

**07, measured resistance (cell D).** We extract the instantaneous resistance from the GITT pulses: it rises from 26.4 to 31.6 mΩ over the test. When R is instead inferred from discharge-only data, it climbs to ~70 mΩ, because it absorbs all polarisation, not just the ohmic part. Fixing R at the measured values barely changes the modes: at RPT15, LAM_Si = 40.1 % [37.8, 42.5] vs 31.2 % published. **The Si offset is therefore not a resistance artefact**; it comes from the component OCPs.

**08, OCP sensitivity (cell D, RPT0→RPT15, measured R).** We fit 50 combinations of graphite OCP (Ai2020 plus 4 from the TUM-EES collection) and Si OCP (Verbrugge2016 plus 9 from the TUM-EES collection), each multi-start MAP. Results are in `results/08_ocp_sensitivity_cellD.csv`. Among the 15 physically plausible fits with RMS ≤ 10 mV:

| mode | range across OCP choices | published |
|---|---|---|
| LLI | 14.5 – 15.7 | 17.1 |
| LAM_P | 8.0 – 10.0 | 7.7 |
| LAM_Si | 37.5 – 46.8 | 31.2 |
| LAM_NE | 7.2 – 19.6 | 21.5 |
| **LAM_Gr** | **0.8 – 16.3** | 19.8 |

LLI and LAM_P are robust to the OCP choice. LAM_Gr is not: it changes by a factor of 20 depending on which graphite curve is used. Fit quality does not settle the question either. The three best fits (6.1–6.3 mV, all with the Wetjen graphite) give LAM_Gr ≈ 2.5 %, while fits only ~2 mV worse give ≈ 16 %. No OCP combination reproduces the published LAM_Si (every fit gives ≥ 35 %). The OCP curves come from the TUM-EES DegradationModeAnalysis repository (MIT; cite Rehm et al. 2026 and the original sources) and are not redistributed here.

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

## Citation
Shan, Y. (2026). *sigr-bdma: Bayesian degradation-mode analysis for Si–graphite blend anodes* (v0.1.0). Zenodo. https://doi.org/10.5281/zenodo.24047660
