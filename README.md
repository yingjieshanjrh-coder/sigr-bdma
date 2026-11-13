# sigr-bdma

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24090400.svg)](https://doi.org/10.5281/zenodo.24090400)

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

OCP-sensitivity range for real data (every graphite x silicon combination, R fixed at the measured value):
```python
from sigr_bdma.sensitivity import ocp_ensemble, ensemble_summary
res = ocp_ensemble(graphites, silicons, fresh_branch, aged_branch, R_fresh, R_aged)
print(ensemble_summary(res, rms_max=10.0))      # n, range, IQR and SD of each mode
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

**08b, all eight cells.** Same 50-combination study for every cell (RPT0 vs RPT15, GITT-measured R). Results are in `results/08_ocp_sensitivity_summary.csv`. Across cells, the median spread due to OCP choice alone is LAM_Gr 12.3 pp, LAM_NE 9.3 pp, LAM_Si 7.8 pp, LAM_P 2.0 pp and LLI 1.2 pp. For the 10 °C and 25 °C cells (A–E), LAM_Gr ranges over 10–15 pp. For the 40 °C cells (F–H), only 2–3 combinations fit within 10 mV, so the spread there is not well sampled. In all eight cells, LLI is 1–4 pp below the published value and LAM_Si is 2–12 pp above it.

**08c, 0–30 % SoC cells (9 cells, RPT0 vs last RPT).** Results are in `results/08_ocp_sensitivity_030_summary.csv` and `figures/fig_mode_ranges_030.png`. The picture is the same: the median spread is LLI 1.4 pp, LAM_P 2.8 pp, LAM_Si 9.7 pp, LAM_NE 16.0 pp and LAM_Gr 19.3 pp. In the strongly degraded cells (LLI 40–54 %), LLI agrees with the published values to within 2 pp and LAM_P to within 5 pp, while LAM_NE is 16–36 pp and LAM_Gr 21–46 pp lower than published.

**10, robustness.** The output is in `results/10_summary_stats.txt`. Whichever RMS cut-off is used (8–15 mV), LAM_Gr spreads 8–35× more than LLI. For LAM_Gr and LAM_NE, the OCP-induced spread is 4–5× the noise-only 95 % interval of a single-OCP Bayesian fit. For LLI and LAM_P it is smaller than that interval.

**11, second chemistry with both branches: Molicel P45B (NCA | graphite–SiOx).** We use the C/30 pOCV charge and discharge curves at 9 check-ups (0–800 EFC) that ship as test data with the TUM-EES DMA repository, decoded from the MATLAB table without MATLAB (`sigr_bdma.datasets.tum_p45b_branches`). For 40 OCP combinations, each with branch-aware graphite and Si, CU1 → CU9 is fitted from charge only, discharge only, and both branches jointly. Results are in `results/11_p45b_branches.csv`.
- LLI is again robust: it spreads by less than 1 pp over the five best fits of each configuration and by at most 1.6 pp over all fits with RMS ≤ 10 mV.
- LAM_Si is poorly constrained from the charge branch alone (spread 21.5 pp over the five best fits, 38 pp over fits with RMS ≤ 10 mV) but much better from the discharge branch (3.5 pp and 8 pp).
- LAM_Gr remains fragile in all three configurations (spread 9–14 pp over the five best fits).
- On these data the joint fit does **not** narrow the spread further than discharge alone. Its residuals are larger (≥ 8.7 mV), which shows that the literature branch-OCP pairs are not mutually consistent for this cell.

**12–13, robustness checks.** Results are in `results/12*`, `results/13_cathode_ocp.csv` and `figures/fig_trajectory_cellD.png`.
- *Known truth*: synthetic curves are generated with one OCP pair and fitted with all 50. When the true pair is in the library, the lowest-RMS fit recovers it every time, with errors below 0.3 pp. When it is not, fits at 6–10 mV RMS give LAM_Gr errors of up to about 11 pp but LLI errors of at most 1.6 pp.
- *Leave one curve out*: removing any single curve leaves the LAM_Gr spread at 7.5 pp or more, still at least 6× the LLI spread.
- *RMS vs LAM_Gr*: the Spearman ρ lies between −0.22 and 0.15 in all eight cells, and none is significant.
- *Multiple optima*: a single OCP pair (Wetjen graphite + Kunz Si) gives LAM_Gr = 2.5 % or 14.5 % depending on the optimisation path, with RMS 7.58 vs 7.81 mV.
- *Trajectories*: across 16 RPTs of cell D, LLI curves almost coincide while LAM_Gr curves diverge. For two OCP pairs LAM_Gr falls by more than 2 pp between consecutive RPTs (unphysical).
- *Positive-electrode OCP*: replacing Chen2020 by an LG M36 coin-cell curve leaves the anode picture unchanged but shifts the median LLI by +2.8 pp (cell D) and +4.0 pp (cell B).

**15, cell-specific reference (P45B).** Fitting CU1 → CU9 with the OCPs measured on electrodes harvested from P45B cells (blend anode lithiation/delithiation and NCA) gives LLI = 17.3–18.3 % across charge, discharge and joint fits. This agrees within 1.5 pp with all literature pairs (17.6–19.3 %) and with the published 18.5 %. LAM_NE does not agree: it is 13.7–22.5 % with the cell-specific curves, 7–18 % with the literature pairs and 9.6 % published. The aged-state fits with the pristine blend curve are also poorer (RMS up to 12 mV). This is consistent with a change in blend-OCP shape as silicon is lost, so the pristine blend curve does not provide a ground truth for the anode losses. Results are in `results/15_p45b_cell_specific.csv`.

**16, profile likelihood, fitting window and spread statistics.** Results are in `results/16*` and `figures/fig_profile_QGr.png`.
- *Profile over Q_Gr* (cells B and D, the three best OCP pairs each): with LAM_Gr fixed anywhere between 0 and 22 % and the other capacities refitted, the aged-state RMS changes by only 1.0–1.8 mV, while LLI moves by at most 1.7 pp. The RMS keeps falling for negative LAM_Gr (an unphysical gain of graphite capacity), and it rises steeply only above 22–28 %. The data therefore give an upper bound on graphite loss, not a value.
- *Fitting window*: refitting cells B, D and E with 3.0–4.2 V instead of 2.5–4.2 V lowers the best RMS from about 6 to 4–5 mV and leaves the pattern unchanged (LLI spread 1.5–1.7 pp, LAM_Gr 13–14 pp). Cropping to 3.2–4.1 V removes the silicon information (LAM_Si spread 67–84 pp) and also degrades LLI (9–12 pp).
- *Spread statistics*: the range is driven partly by a few outlying combinations. Median over the 0–100 % cells, the IQR is 1.7 pp for LAM_Gr and 2.8 pp for LAM_Si against 0.5 pp for LLI; the SD is 4.0, 2.3 and 0.4 pp; the range rarefied to five combinations is 11.4, 7.2 and 0.9 pp.

**17, silicon content (synthetic).** Fresh and aged C/10 curves are generated with a known OCP pair for Si capacity shares of 5–35 % and fitted with the rest of the library (the true pair excluded). LLI spreads by at most 1.2 pp at every share. LAM_Si is unconstrained at 5 % Si (28–31 pp) and becomes well constrained as the share grows (2–4 pp at 35 %). LAM_Gr stays OCP-sensitive at every share (3–25 pp, depending on the true pair). Results are in `results/17_si_fraction_summary.csv`.

## Reproducing the paper
Data: LG M50T from Zenodo 10.5281/zenodo.7235857 (unzip; pass the inner `Battery Degradation Data` folder as `DATA`), and a clone of https://github.com/tum-ees/DegradationModeAnalysis as `TUM`.

| Output | Command |
|---|---|
| Fig. 1 (OCP library) | `python examples/18_ocp_library_figure.py TUM` |
| Tables 1–2, Figs. 2, 3 and 5 | `python examples/08_ocp_sensitivity.py DATA TUM` (and with `ABJDEFKLM "0-30% SoC"`), then `python examples/09_paper_figures.py` |
| Noise-only intervals, thresholds | `python examples/07_kirkaldy_measured_R.py DATA`, then `python examples/10_summary_stats.py` |
| Table 3 (P45B) | `python examples/11_p45b_branches.py TUM` |
| Known truth, leave-one-out, trajectories (Table 4, Fig. 6) | `python examples/12_review_round1.py DATA TUM`, then `python examples/09_paper_figures.py` |
| Cathode OCP | `python examples/13_cathode_ocp.py DATA TUM PBPARAM` (PBPARAM: `pbparam/input/data` of https://github.com/paramm-team/pybamm-param) |
| Cell-specific P45B reference | `python examples/15_p45b_cell_specific.py TUM` |
| Profile, window, spread statistics, Fig. 4 | `python examples/16_profile_window_spread.py DATA TUM`, then `python examples/09_paper_figures.py` |
| Silicon content | `python examples/17_si_fraction.py TUM` |
| Graphical abstract | `python examples/19_graphical_abstract.py` |

## Limitations (current version)
- Only the ohmic polarisation is modelled. Diffusion and charge-transfer overpotentials at C/10 are not, so residuals are correlated (9–10 mV RMS on the LG M50T data) and the noise-only credible intervals are too narrow. The AR(1) option does not fix this (see 06).
- Graphite is treated as hysteresis-free. Si hysteresis is a fixed gap scaled by `hys`; there is no path-dependent (scanning-curve) model.
- The joint charge/discharge fit has been tested on one real cell (P45B, example 11), where it did not outperform a discharge-only fit.
- The `placeholder_si` curves are for tests only. Real analyses use `si_verbrugge2016`, `graphite_ai2020` or the TUM-EES collection (`load_tum_ocps`).

## Related work
- Kirkaldy et al., *ACS Appl. Energy Mater.* 2022: composite-electrode OCV fitting, LG M50T dataset.
- Rehm et al., *J. Power Sources* 2026: DMA for Si–Gr blends; the Si OCP generated from blend and graphite half-cells. Their study shows modes change by more than 2× with the choice of Si OCP.
- "Phantom LAM and LLI", arXiv:2512.19773: resistance and hysteresis bias in voltage-curve DMA (deterministic fits).
- Lin & Khoo, *J. Power Sources*: Fisher-information identifiability for single-material electrodes.
- Chen et al., *J. Electrochem. Soc.* 2020: NMC811 OCP fit.
- Ai et al., *J. Electrochem. Soc.* 2020 (graphite OCP); Verbrugge et al., *J. Electrochem. Soc.* 2016 (Si OCP), both as distributed with PyBaMM.
- TUM-EES DegradationModeAnalysis (Rehm et al. 2026): collection of literature graphite and Si OCPs used in 08.

## License
MIT License. Copyright (c) 2026 Yingjie Shan.

## Citation
Shan, Y. (2026). *sigr-bdma: Bayesian degradation-mode analysis for Si–graphite blend anodes* (v0.3.0). Zenodo. https://doi.org/10.5281/zenodo.24090400
