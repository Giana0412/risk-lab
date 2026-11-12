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


# ── 성과 기여도 ──────────────────────────────────────────────────────────

def _attr_data(n=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2026-01-01", periods=n)
    R = pd.DataFrame(rng.normal(0.0005, 0.02, (n, 4)), index=idx, columns=["A", "B", "C", "D"])
    secR = pd.DataFrame(rng.normal(0.0003, 0.01, (n, 3)), index=idx, columns=["tech", "fin", "energy"])
    bench = 0.5 * secR["tech"] + 0.3 * secR["fin"] + 0.2 * secR["energy"] + rng.normal(0, 0.001, n)
    return R, secR, bench


def test_drift_weights_buy_and_hold():
    R, _, _ = _attr_data()
    w0 = {"A": 0.5, "B": 0.4}                      # 현금 10%
    W = risk.drift_weights(R, w0)
    assert abs(W.iloc[0]["A"] - 0.5) < 1e-12 and abs(W.iloc[0]["현금"] - 0.1) < 1e-12
    assert np.allclose(W.sum(axis=1), 1.0)
    # 매일 비중×수익을 복리로 이으면 실제 보유가치 변화와 같다
    rp = (W[["A", "B"]] * R[["A", "B"]]).sum(axis=1)
    val = 0.5 * (1 + R["A"]).prod() + 0.4 * (1 + R["B"]).prod() + 0.1
    assert abs((1 + rp).prod() - val) < 1e-12


def test_carino_link_sums_to_period_excess():
    R, _, bench = _attr_data()
    contrib, rp, rb = risk.holding_attribution(R, {"A": 0.3, "B": 0.3, "C": 0.2}, bench)
    excess = (1 + rp).prod() - (1 + rb).prod()
    assert abs(contrib.sum() - excess) < 1e-12


def test_brinson_sums_and_pure_effects():
    R, secR, bench = _attr_data()
    w0 = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    sectors = {"A": "tech", "B": "tech", "C": "fin", "D": "unknown"}
    tab, tot = risk.brinson_attribution(R, w0, sectors, bench, {"tech": 0.5, "fin": 0.3, "energy": 0.2}, secR)
    assert abs(tot["배분"] + tot["선택"] + tot["재구성오차"] - tot["초과"]) < 1e-12
    # 현금이 있어도 합이 맞아야 한다 (현금 배분 = w_cash·(0 − r_b*))
    _, totc = risk.brinson_attribution(R, {"A": 0.3, "C": 0.2}, sectors, bench,
                                       {"tech": 0.5, "fin": 0.3, "energy": 0.2}, secR)
    assert abs(totc["배분"] + totc["선택"] + totc["재구성오차"] - totc["초과"]) < 1e-12
    # 순수 배분: 종목 = 섹터 ETF 그대로면 선택 효과는 0
    R2 = pd.DataFrame({"T": secR["tech"], "F": secR["fin"]})
    tab2, tot2 = risk.brinson_attribution(R2, {"T": 0.8, "F": 0.2}, {"T": "tech", "F": "fin"},
                                          bench, {"tech": 0.5, "fin": 0.3, "energy": 0.2}, secR)
    assert abs(tot2["선택"]) < 1e-12
    assert abs(tot2["배분"] + tot2["재구성오차"] - tot2["초과"]) < 1e-12
    # 순수 선택: 벤치와 같은 섹터 비중이면 배분은 0 (섹터 수익 = ETF 일 때만 성립하지 않으므로 비중만 맞춘다)
    R3 = pd.DataFrame({"T": secR["tech"] + 0.001, "F": secR["fin"], "E": secR["energy"]})
    _, tot3 = risk.brinson_attribution(R3, {"T": 0.5, "F": 0.3, "E": 0.2}, {"T": "tech", "F": "fin", "E": "energy"},
                                       bench, {"tech": 0.5, "fin": 0.3, "energy": 0.2}, secR)
    assert tot3["선택"] > 0 and abs(tot3["배분"]) < 2e-3   # 비중이 흘러가 약간의 배분이 생긴다


def test_brinson_bench_weights_drift_back_from_current():
    """벤치가 섹터 ETF 의 시가총액 가중 합(매수 후 보유)이면, 끝 비중에서 되짚은 재구성 벤치는 정확히 같다."""
    _, secR, _ = _attr_data()
    w_start = pd.Series({"tech": 0.5, "fin": 0.3, "energy": 0.2})
    val = (1 + secR).cumprod() * w_start
    bench = val.sum(axis=1).pct_change().fillna(float((secR.iloc[0] * w_start).sum()))
    w_end = (val.iloc[-1] / val.iloc[-1].sum()).to_dict()
    R = pd.DataFrame({"T": secR["tech"]})
    _, tot = risk.brinson_attribution(R, {"T": 1.0}, {"T": "tech"}, bench, w_end, secR)
    assert abs(tot["재구성오차"]) < 1e-12


def test_factor_attribution_sums_and_recovers_beta():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2026-01-01", periods=80)
    F = pd.DataFrame({"mkt": rng.normal(0, 0.01, 80), "size": rng.normal(0, 0.005, 80)}, index=idx)
    R = pd.DataFrame({"A": 1.5 * F["mkt"] + rng.normal(0, 0.002, 80),
                      "B": 0.5 * F["mkt"] + 1.0 * F["size"] + rng.normal(0, 0.002, 80)})
    bench = F["mkt"]
    betas = pd.DataFrame({"mkt": [1.5, 0.5], "size": [0.0, 1.0]}, index=["A", "B"])
    contrib, act = risk.factor_attribution(R, {"A": 0.5, "B": 0.5}, bench, betas,
                                           pd.Series({"mkt": 1.0, "size": 0.0}), F)
    rp = (risk.drift_weights(R, {"A": .5, "B": .5})[["A", "B"]] * R).sum(axis=1)
    assert abs(contrib.sum() - ((1 + rp).prod() - (1 + bench).prod())) < 1e-12
    assert abs(act["mkt"]) < 0.1 and abs(act["size"] - 0.5) < 0.1


# ── 매매 이력(blotter) ───────────────────────────────────────────────────

def test_blotter_parse_and_rebuild_matches_pnl():
    import blotter
    text = """
✓    Closed    X    AAA US    Long    110                1000    0    10    0    2026/01/07 22:30:00    2026/01/07 22:30:00    110    2026/01/05 22:30:00    100
Open    X    BBB US    Long    55            5500    500    0    10    0    2026/01/07 21:00:00        2026/01/06 22:30:00    100
"""
    lots = blotter.parse_tmsg(text)
    assert list(lots["bbg"]) == ["BBB US", "AAA US"] and lots["shares"].tolist() == [100, 100]
    assert abs(lots.set_index("bbg").loc["AAA US", "entry_px"] - 100) < 1e-9
    days = pd.bdate_range("2026-01-05", "2026-01-08")
    close = pd.DataFrame({"AAA": [101, 104, 108, 109], "BBB": [50, 51, 53, 55]}, index=days, dtype=float)
    opens = pd.DataFrame({"AAA": [100, 102, 110, 108], "BBB": [50, 50, 52, 54]}, index=days, dtype=float)
    bk = blotter.build(lots, close, opens, days, notional=100_000)
    got = bk.lots.set_index("bbg")["pnl_rebuilt"]
    assert abs(got["AAA US"] - 1000) < 1e-9 and abs(got["BBB US"] - 500) < 1e-9
    # 01-07 은 AAA 를 시가에 판다 → 그날 장중 구간에는 AAA 비중이 없다
    pm = pd.Timestamp("2026-01-07 16:00")
    assert bk.W.at[pm, "AAA#0"] == 0 and bk.W.at[pm, "BBB#1"] > 0
    # Σ w·r = 순자산 변화
    port = (bk.W.drop(columns="현금") * bk.R).sum(axis=1)
    assert abs((1 + port).prod() * 100_000 - bk.nav.iloc[-1]) < 1e-6


def test_random_portfolio_test():
    cand = np.array([-0.10, 0.0, 0.10, 0.20, -0.05, 0.05])
    w = {"A": 0.5, "B": 0.3, "ETF": 0.1}                       # 현금 0.1
    own = pd.Series({"A": 0.20, "B": 0.05, "ETF": 0.02})
    pools = {"A": np.array([0, 1, 2, 3]), "B": np.array([4, 5]), "ETF": None}
    rt = risk.random_portfolio_test(w, own, pools, cand, n=50_000, seed=1)
    assert abs(rt.actual - (0.5 * 0.20 + 0.3 * 0.05 + 0.1 * 0.02)) < 1e-12
    # 기대값 = Σ w·(후보 평균) + 고정 종목
    assert abs(rt.sims.mean() - (0.5 * 0.05 + 0.3 * 0.0 + 0.1 * 0.02)) < 2e-3
    # A 는 후보 중 최고, B 는 위쪽 → 실제 이상인 경우는 (A=0.20, B=0.05) 뿐: 1/4 × 1/2
    assert abs(rt.p_value - 0.125) < 0.01
    assert rt.kept == ["ETF"] and rt.replaced == ["A", "B"]
