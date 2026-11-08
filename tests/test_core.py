import numpy as np

import sigr_bdma as s

TRUTH = dict(Q_Gr=5.0, Q_Si=1.0, Q_P=5.6, Q_Li=5.5, R=0.03)


def test_ocps_monotone():
    x = np.linspace(0, 1, 500)
    assert np.all(np.diff(s.U_graphite(x)) < 0)
    assert np.all(np.diff(s.U_nmc811(np.linspace(0.15, 1, 500))) < 0)
    si = s.placeholder_si()
    assert np.all(np.diff(si.lith(x)) <= 0) and np.all(np.diff(si.delith(x)) <= 0)
    assert np.all(si.delith(x) >= si.lith(x))


def test_branch_simulation_and_hysteresis_order():
    cell = s.Cell(s.placeholder_si())
    ch = cell.simulate_branch(TRUTH, "charge", I=0.5)
    dis = cell.simulate_branch(TRUTH, "discharge", I=0.5)
    assert ch.span > 3 and dis.span > 3
    # at the same throughput the charge voltage must be above the discharge voltage
    q = 0.5 * min(ch.span, dis.span)
    assert np.interp(q, ch.Q, ch.V) > np.interp(q, dis.Q, dis.V)


def test_map_recovers_truth_noise_free():
    cell = s.Cell(s.placeholder_si())
    brs = [cell.simulate_branch(TRUTH, d, I=0.5) for d in ("charge", "discharge")]
    prob = s.Problem(cell, brs, s.default_params({k: v * 0.95 for k, v in TRUTH.items()}))
    z = prob.fit_map()
    for i, n in enumerate(prob.names):
        assert abs(z[i] - TRUTH[n]) / TRUTH[n] < 5e-3, n


def test_mixture_weight_gradient_nonzero():
    A, B = s.placeholder_si("A"), s.placeholder_si("B", shift=0.04, gap_scale=0.6)
    cell = s.Cell([A, B])
    p = {**TRUTH, "w": 0.3, "_I": 0.0}
    v1 = cell.ocv_curve(p, "charge")[1]
    v2 = cell.ocv_curve({**p, "w": 0.3 + 1e-6}, "charge")[1]
    assert np.max(np.abs(v2 - v1)) > 0


def test_mcmc_gaussian():
    logp = lambda z: -0.5 * np.sum((z - 1.0) ** 2 / np.array([1.0, 4.0]))
    chain, acc = s.ensemble_mcmc(logp, np.array([1.0, 1.0]), n_steps=3000, init_scale=0.5)
    flat = chain[1000:].reshape(-1, 2)
    assert np.allclose(flat.mean(0), 1.0, atol=0.15)
    assert np.allclose(flat.std(0), [1.0, 2.0], rtol=0.15)
    assert 0.2 < acc < 0.9


def test_extract_si_from_blend_roundtrip():
    si, g = s.placeholder_si(), 0.15
    U = np.linspace(0.06, 0.9, 300)
    from sigr_bdma.ocp import Inverse
    xgr = Inverse(s.U_graphite)(U)
    q_l = (1 - g) * xgr + g * Inverse(si.lith)(U)
    q_d = (1 - g) * xgr + g * Inverse(si.delith)(U)
    est = s.extract_si_from_blend(U, q_l, U, q_d, g)
    x = np.linspace(0.2, 0.8, 20)
    assert np.max(np.abs(est.lith(x) - si.lith(x))) < 0.02
    assert np.max(np.abs(est.delith(x) - si.delith(x))) < 0.02


def test_split_cycler_log():
    cell = s.Cell(s.placeholder_si())
    ch = cell.simulate_branch(TRUTH, "charge", I=0.5, n=400)
    dis = cell.simulate_branch(TRUTH, "discharge", I=0.5, n=400)
    t_ch = ch.Q / 0.5 * 3600
    t_dis = t_ch[-1] + 3600 + (dis.Q[-1] - dis.Q[::-1]) / 0.5 * 3600
    t = np.concatenate([t_ch, [t_ch[-1] + 1800], t_dis])
    I = np.concatenate([np.full(len(t_ch), 0.5), [0.0], np.full(len(t_dis), -0.5)])
    V = np.concatenate([ch.V, [4.1], dis.V[::-1]])
    brs = s.split_cycler_log(t, I, V, 0.5)
    assert [b.direction for b in brs] == ["charge", "discharge"]
    assert abs(brs[0].span - ch.span) < 0.02 and abs(brs[1].span - dis.span) < 0.02
