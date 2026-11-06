"""risk.py 정합성 — 답을 아는 합성 데이터로. 실행: python3 -m pytest test_risk.py"""
import math

import numpy as np
import pandas as pd

import risk


def _normal(n=20_000, sd=0.01, seed=0):
    return pd.Series(np.random.default_rng(seed).normal(0, sd, n))


def test_normal_methods_agree():
    r = _normal()
    true99 = 2.3263 * 0.01
    for f in (risk.var_historical, risk.var_parametric, risk.var_cornish_fisher):
        assert abs(f(r, 0.99) - true99) / true99 < 0.05, f.__name__
    # 정규 ES 99% = σ·φ(z)/(1−α) = 2.665σ
    assert abs(risk.es_parametric(r, 0.99) - 0.02665) < 0.001
    assert abs(risk.es_historical(r, 0.99) - 0.02665) < 0.0015


def test_fat_tails_raise_cornish_fisher():
    r = pd.Series(np.random.default_rng(1).standard_t(3, 20_000) * 0.01)
    assert risk.var_cornish_fisher(r, 0.99) > risk.var_parametric(r, 0.99)


def test_component_var_sums_to_total():
    rng = np.random.default_rng(2)
    R = pd.DataFrame(rng.multivariate_normal([0, 0, 0], [[1, .5, .2], [.5, 1, .3], [.2, .3, 1]], 5000) * 0.01,
                     columns=list("ABC"))
    w = {"A": 0.5, "B": 0.3, "C": 0.2}
    cv = risk.component_var(R, w, 0.99)
    p = risk.portfolio_returns(R, w)
    z = -risk._N.inv_cdf(0.01)
    assert abs(cv["기여 VaR"].sum() - z * p.std()) < 1e-9
    assert abs(cv["기여 비율"].sum() - 1) < 1e-12


def test_chi2_and_kupiec():
    assert abs(risk._chi2_sf(3.841, 1) - 0.05) < 1e-3
    assert abs(risk._chi2_sf(5.991, 2) - 0.05) < 1e-3
    _, p = risk.kupiec_pof(10, 1000, 0.99)   # 정확히 기대치
    assert p > 0.99
    _, p = risk.kupiec_pof(30, 1000, 0.99)   # 3배
    assert p < 0.001


def test_independence_detects_clustering():
    clustered = np.zeros(1000, bool); clustered[100:110] = True
    spread = np.zeros(1000, bool); spread[::100] = True
    assert risk.christoffersen_ind(clustered)[1] < 0.01
    assert risk.christoffersen_ind(spread)[1] > 0.1


def test_backtest_normal_is_calibrated():
    bt = risk.backtest_var(_normal(1500, seed=3), 0.99, 250)
    assert bt.summary.loc["정규(분산-공분산)", "Kupiec p"] > 0.05


def test_monte_carlo_t_is_wider_than_normal():
    rng = np.random.default_rng(4)
    R = pd.DataFrame(rng.normal(0, 0.01, (2000, 2)), columns=["A", "B"])
    w = {"A": .5, "B": .5}
    vn, en = risk.var_monte_carlo(R, w, 0.99, dof=None)
    vt, et = risk.var_monte_carlo(R, w, 0.99, dof=5.0)
    assert et > en and en > vn


def test_stress_replay_and_proxy():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    b = pd.Series(np.linspace(100, 80, len(idx)), idx)
    px = pd.DataFrame({"BENCH": b, "OLD": b * 2})
    px["NEW"] = np.nan
    px.loc["2020-03-01":, "NEW"] = 50.0   # 구간 중간 상장 → 대리
    out = risk.stress_replay(px, {"OLD": 0.5, "NEW": 0.5}, "BENCH",
                             {"x": ("2020-02-03", "2020-03-31")}, pd.Series({"NEW": 2.0}))
    rb = out.loc["x", "벤치"]
    assert out.loc["x", "대리 비중"] == 0.5
    assert math.isclose(out.loc["x", "북"], 0.5 * rb + 0.5 * 2.0 * rb)


def test_basel_light_only_at_99():
    bt = risk.backtest_var(_normal(1000, seed=5), 0.95, 250)
    assert (bt.summary["바젤(250일 환산)"] == "— (99% 전용)").all()


def test_stale_price_falls_back_to_proxy():
    idx = pd.bdate_range("2020-01-01", "2020-04-30")
    b = pd.Series(np.linspace(100, 80, len(idx)), idx)
    px = pd.DataFrame({"BENCH": b, "DEAD": b.copy()})
    px.loc["2020-02-15":, "DEAD"] = np.nan   # 구간 중간에 끊김
    out = risk.stress_replay(px, {"DEAD": 1.0}, "BENCH",
                             {"x": ("2020-02-03", "2020-03-31")}, pd.Series({"DEAD": 1.0}))
    assert out.loc["x", "대리 비중"] == 1.0


def _garch_sim(n=3000, w=2e-6, a=0.08, b=0.90, seed=6):
    rng = np.random.default_rng(seed)
    s2, x = w / (1 - a - b), np.empty(n)
    for t in range(n):
        x[t] = math.sqrt(s2) * rng.standard_normal()
        s2 = w + a * x[t] ** 2 + b * s2
    return pd.Series(x)


def test_garch_recovers_params():
    g = risk.garch_fit(_garch_sim())
    assert abs(g.alpha - 0.08) < 0.04 and abs(g.beta - 0.90) < 0.05
    assert abs(g.z.std() - 1) < 0.05


def test_fhs_fixes_clustering_in_backtest():
    r = _garch_sim(1500, seed=8)
    s = risk.backtest_var(r, 0.99, 250).summary
    assert s.loc["FHS-GARCH", "Kupiec p"] > 0.05
    assert s.loc["FHS-GARCH", "조건부커버리지 p"] > 0.05


def test_simulate_fhs_scales_like_sqrt_h_when_calm():
    g = risk.garch_fit(_normal(3000, seed=9))
    s5 = risk.simulate_fhs(g, 5, 40_000)
    assert abs(s5.std() / (0.01 * math.sqrt(5)) - 1) < 0.05


def test_reaction_day_bmo_amc():
    td = pd.bdate_range("2026-11-02", "2026-11-13")
    bmo = pd.Timestamp("2026-11-05 06:00", tz="America/New_York")
    amc = pd.Timestamp("2026-11-05 16:00", tz="America/New_York")
    fri_amc = pd.Timestamp("2026-11-06 16:05", tz="America/New_York")
    assert risk.reaction_day(bmo, td) == pd.Timestamp("2026-11-05")
    assert risk.reaction_day(amc, td) == pd.Timestamp("2026-11-06")
    assert risk.reaction_day(fri_amc, td) == pd.Timestamp("2026-11-09")


def test_outcome_adds_jump_variance():
    ex = _normal(1000, sd=0.005, seed=10)
    jumps = {"X": (0.2, np.array([-0.15, 0.15, -0.15, 0.15]))}
    o = risk.simulate_outcome(ex, 5, jumps, n=40_000)
    # 분산 가산: Var(total) ≈ Var(base) + 0.2²·0.15²
    assert abs(o.total.var() - (o.base.var() + 0.03 ** 2)) / o.total.var() < 0.1


def test_variance_ratio_detects_mean_reversion():
    e = np.random.default_rng(12).normal(0, 0.01, 5000)
    iid = pd.Series(e)
    mr = pd.Series(e[1:] - 0.5 * e[:-1])   # MA(1) θ=−0.5 → 되돌림
    assert abs(risk.variance_ratio(iid, 5) - 1) < 0.1
    assert risk.variance_ratio(mr, 5) < 0.8


def test_ambiguous_midnight_picks_bigger_move():
    td = pd.bdate_range("2026-11-02", "2026-11-13")
    ts = pd.Timestamp("2026-11-05 00:00", tz="America/New_York")
    abn = pd.Series(0.001, index=td); abn[pd.Timestamp("2026-11-06")] = -0.12
    assert risk.reaction_day(ts, td, abn) == pd.Timestamp("2026-11-06")
    abn[pd.Timestamp("2026-11-05")] = 0.2
    assert risk.reaction_day(ts, td, abn) == pd.Timestamp("2026-11-05")


def test_variance_ratio_unbiased_for_large_h():
    vals = [risk.variance_ratio(pd.Series(np.random.default_rng(s).normal(0, .01, 1000)), 60) for s in range(30)]
    assert abs(np.mean(vals) - 1) < 0.08


def test_deflate_jumps_and_min_samples():
    js = np.array([-0.1, 0.1, -0.05, 0.05])
    d = risk.deflate_jumps(js, 0.02)
    assert abs(d.var() - (js.var() - 0.02 ** 2)) < 1e-12
    o = risk.simulate_outcome(_normal(800, 0.005, 13), 3, {"X": (0.2, np.array([0.3, -0.3]))}, n=5000)
    assert np.allclose(o.base, o.total)   # 표본 2개 → 점프 미적용


def test_nyse_calendar():
    import data
    d = data.trading_days("2026-11-20", "2026-12-04")
    assert pd.Timestamp("2026-11-26") not in d and pd.Timestamp("2026-11-27") in d and len(d) == 10
