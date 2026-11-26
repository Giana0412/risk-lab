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
    assert bt.summary.loc["정규(분산-공분산)", "Z2 p"] > 0.05      # i.i.d. 정규 → ES 도 맞다
    assert abs(bt.summary.loc["정규(분산-공분산)", "Z2"]) < 0.5


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
    r = _garch_sim(1750, seed=8)   # GARCH 창 500일 → 검정일 1,250일
    s = risk.backtest_var(r, 0.99, 250, n_sim=0).summary
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


# ── GARCH-t · ES 백테스트 ───────────────────────────────────────────────

def _garch_t_sim(n=3000, w=2e-6, a=0.08, b=0.90, nu=5.0, seed=6):
    """분산 1 로 맞춘 t_ν 혁신의 GARCH(1,1)."""
    rng = np.random.default_rng(seed)
    k = math.sqrt((nu - 2) / nu)
    s2, x = w / (1 - a - b), np.empty(n)
    for t in range(n):
        x[t] = math.sqrt(s2) * rng.standard_t(nu) * k
        s2 = w + a * x[t] ** 2 + b * s2
    return pd.Series(x)


def test_t_quantile_and_es_closed_forms():
    # 표의 값: t_5 99% 3.3649, t_3 97.5% 3.1824, t_10 95% 1.8125
    assert abs(risk.t_ppf(0.99, 5) - 3.36493) < 1e-4
    assert abs(risk.t_ppf(0.025, 3) + 3.18245) < 1e-4
    assert abs(risk.t_ppf(0.95, 10) - 1.81246) < 1e-4
    # ν → ∞ 면 정규: VaR 2.3263, ES 2.6652
    q, es = risk.t_std_var_es(0.99, 1e6)
    assert abs(q - 2.3263) < 1e-3 and abs(es - 2.6652) < 1e-3
    # 분산 1 t_4 의 ES 를 몬테카를로와 대조
    z = np.random.default_rng(0).standard_t(4, 2_000_000) * math.sqrt(2 / 4)
    qs = np.quantile(z, 0.01)
    q, es = risk.t_std_var_es(0.99, 4)
    assert abs(q + qs) / q < 0.01 and abs(es + z[z <= qs].mean()) / es < 0.015


def test_garch_t_recovers_nu_and_params():
    g = risk.garch_t_fit(_garch_t_sim(nu=5.0))
    assert abs(g.nu - 5) < 1.5
    assert abs(g.alpha - 0.08) < 0.04 and abs(g.beta - 0.90) < 0.05
    # 정규 혁신이면 ν 가 크게 나와야 한다 (t 가 정규로 수렴)
    assert risk.garch_t_fit(_garch_sim(seed=7)).nu > 15
    r = _garch_t_sim(nu=5.0)
    assert risk.es_garch_t(r, 0.99) > risk.var_garch_t(r, 0.99) > 0


def test_cornish_fisher_es_matches_normal_when_no_skew():
    r = _normal(50_000, seed=12)
    assert abs(risk.es_cornish_fisher(r, 0.99) - risk.es_parametric(r, 0.99)) < 5e-4


def test_z2_zero_when_correct_and_negative_when_es_too_small():
    alpha, n = 0.99, 200_000
    z, e = -risk._N.inv_cdf(1 - alpha), risk._N.pdf(risk._N.inv_cdf(1 - alpha)) / (1 - alpha)
    v, es = np.full(n, z), np.full(n, e)
    x = np.random.default_rng(13).standard_normal(n)
    assert abs(risk.es_z2(x, v, es, alpha)) < 0.07          # 정규 데이터 + 정규 예측 (표준오차 ≈ 0.023)
    t3 = np.random.default_rng(14).standard_t(3, n) / math.sqrt(3)   # 분산 1 t_3
    assert risk.es_z2(t3, v, es, alpha) < -0.5                # 팻테일 데이터 + 정규 예측
    # McNeil–Frey: 맞는 모형은 기각되지 않고, 정규 ES 는 t 꼬리에서 기각된다
    assert risk.mcneil_frey(x, v, es, np.ones(n))[1] > 0.05
    m, p, k = risk.mcneil_frey(t3, v, es, np.ones(n))
    assert m > 0 and p < 0.01 and k > 1000


def test_es_backtest_garch_t_passes_normal_es_fails():
    """t-GARCH 데이터: GARCH-t 는 커버리지·ES 검정을 통과, 정규 ES(EWMA σ)는 ES 를 과소추정."""
    bt = risk.backtest_var(_garch_t_sim(2000, nu=4.0, seed=10), 0.975, 500, n_sim=1000)
    s = bt.summary
    assert list(bt.es.columns) == list(bt.forecasts.columns) and "GARCH-t" in bt.es
    assert (bt.es >= bt.forecasts - 1e-12).all().all()         # ES ≥ VaR
    g, e = s.loc["GARCH-t"], s.loc["EWMA λ=0.94"]
    assert g["Kupiec p"] > 0.05 and g["Z2 p"] > 0.05 and g["MF p"] > 0.05
    assert e["MF 잔차"] > 0 and e["MF p"] < 0.01 and e["Z2 p"] < 0.05
    assert e["Z2"] < g["Z2"]
    # EWMA 의 ES 는 같은 σ 의 정규 ES 그대로다
    k = risk._N.pdf(risk._N.inv_cdf(0.025)) / 0.025 / -risk._N.inv_cdf(0.025)
    assert np.allclose(bt.es["EWMA λ=0.94"], k * bt.forecasts["EWMA λ=0.94"])


def test_garch_window_keeps_test_days_paired():
    """GARCH 계열만 긴 창을 쓰고, 검정일은 모든 방법이 같다 (max(window, garch_window) 부터)."""
    r = _normal(800, seed=15)
    b5 = risk.backtest_var(r, 0.99, 250, n_sim=0)                     # 기본 garch_window=500
    b2 = risk.backtest_var(r, 0.99, 250, n_sim=0, garch_window=None)
    assert b5.forecasts.index.equals(r.index[500:]) and b2.forecasts.index.equals(r.index[250:])
    assert not b5.forecasts.isna().any().any() and (b5.summary["표본일"] == 300).all()
    other = list(risk.METHODS)                                         # GARCH 가 아닌 방법은 그대로
    pd.testing.assert_frame_equal(b5.forecasts[other], b2.forecasts.loc[r.index[500:], other])
    assert not np.allclose(b5.forecasts["GARCH-t"], b2.forecasts.loc[r.index[500:], "GARCH-t"])


def _garch_t_sim_sigma(n, nu, seed, w=2e-6, a=0.08, b=0.90):
    """_garch_t_sim 과 같고, 참 σ_t 경로(n+1개, 마지막은 다음 날)도 돌려준다."""
    rng = np.random.default_rng(seed)
    k = math.sqrt((nu - 2) / nu)
    s2, x, sg = w / (1 - a - b), np.empty(n), np.empty(n + 1)
    for t in range(n):
        sg[t] = math.sqrt(s2)
        x[t] = sg[t] * rng.standard_t(nu) * k
        s2 = w + a * x[t] ** 2 + b * s2
    sg[n] = math.sqrt(s2)
    return pd.Series(x), sg


def test_garch_t_longer_window_is_closer_to_nominal():
    """t-GARCH 에서 250일 적합은 σ 를 낮게 잡아 99% VaR 를 너무 자주 넘는다. 500일이 명목에 더 가깝다.
    위반을 세는 대신 참 σ·ν 로 '그날 VaR 를 넘을 참 확률'을 평균한다 — 같은 정보를 훨씬 적은 잡음으로 본다.
    (시드 10개: 250일 평균 1.28%, 500일 1.15%, 모든 시드에서 500일이 더 낮다.)"""
    nu = 5.0
    x, sg = _garch_t_sim_sigma(10_500, nu, seed=0)
    k = math.sqrt((nu - 2) / nu)
    p = {w: np.mean([risk.t_cdf(-risk.var_garch_t(None, 0.99, risk.garch_t_fit(x.iloc[e - w:e])) / (sg[e] * k), nu)
                     for e in range(500, 10_500, 200)]) for w in (250, 500)}
    assert p[250] > 0.0115                  # 과소 커버 (명목 1%)
    assert p[500] < p[250] and abs(p[500] - 0.01) < abs(p[250] - 0.01)
    assert 0.009 < p[500] < 0.0135


# ── HAR / HAR-X (KAN-26) ────────────────────────────────────────────────

def _har_sim(n, beta, sd=0.5, seed=0):
    """log r²_{t+1} = β0 + β1·log r²_t + β5·log mean₅ r² + β22·log mean₂₂ r² + ε (참 HAR, 로그 공간)."""
    rng = np.random.default_rng(seed)
    x = np.empty(n)
    x[:66] = rng.normal(0, 0.01, 66)
    for t in range(66, n):
        r2 = x[t - 22:t] ** 2
        y = beta[0] + beta[1] * math.log(r2[-1]) + beta[2] * math.log(r2[-5:].mean()) \
            + beta[3] * math.log(r2.mean()) + rng.normal(0, sd)
        x[t] = rng.choice([-1.0, 1.0]) * math.exp(y / 2)
    return pd.Series(x, index=pd.bdate_range("2000-01-03", periods=n))


def test_har_ols_recovers_coefficients():
    beta = (-1.5, 0.1, 0.3, 0.4)
    g = risk.har_fit(_har_sim(6000, beta), x=False, window=6000)
    assert g.cols == risk.HAR_COLS
    assert np.allclose(g.b[1:], beta[1:], atol=0.06), g.b
    assert abs(g.b[1:].sum() - sum(beta[1:])) < 0.04
    # σ̂ 배율 c 로 표준화 잔차의 제곱평균이 1 이 된다 (그래서 정규·t 변형이 의미가 있다)
    assert abs((g.z ** 2).mean() - 1) < 1e-9
    v_f, e_f = risk.har_var_es(g, 0.99, "fhs")
    v_n, e_n = risk.har_var_es(g, 0.99, "normal")
    v_t, e_t = risk.har_var_es(g, 0.99, "t")
    assert e_f > v_f > 0 and e_n > v_n > 0 and e_t > v_t > 0
    assert abs(v_n / g.sigma_next - 2.3263) < 1e-3


def test_project_next_uses_year_ago_quarter_or_91_days():
    rx = pd.DatetimeIndex(["2025-01-28", "2025-04-29", "2025-07-30", "2025-10-29"])
    assert risk.project_next(rx[:0]) is None
    assert risk.project_next(rx[:3]) == pd.Timestamp("2025-07-30") + pd.Timedelta(days=91)   # 4건 미만
    assert risk.project_next(rx) == pd.Timestamp("2025-01-28") + pd.Timedelta(days=364)      # 4분기 전 + 364
    # 1년 전 같은 분기 + 364 가 마지막 발표 + 30일 이내면 (분기 건너뜀 등) 마지막 + 91일
    odd = pd.DatetimeIndex(["2024-11-20", "2025-04-29", "2025-07-30", "2025-10-29"])
    assert risk.project_next(odd) == pd.Timestamp("2025-10-29") + pd.Timedelta(days=91)


def _quarterly_stamps(start="2018-01-25", n=36, hour=16):
    """해마다 같은 분기 날짜(± 1일)에 장 마감 후 발표 — 반응일은 다음 거래일. 실제 대형주처럼 규칙적이다."""
    rng = np.random.default_rng(0)
    s = pd.Timestamp(start)
    return [pd.Timestamp((s + pd.DateOffset(months=3 * q) + pd.Timedelta(days=int(rng.integers(-1, 2)))).date())
            + pd.Timedelta(hours=hour) for q in range(n)]


def test_earnings_exposure_uses_only_past_announcements():
    sess = pd.bdate_range("2018-01-02", "2026-06-30")
    st_ = _quarterly_stamps()
    w = {"A": 0.25, "B": 0.0, "C": 0.5}
    full = risk.earnings_exposure({"A": st_, "C": st_[2:]}, w, sess)
    assert set(np.unique(full)) <= {0.0, 0.25, 0.5, 0.75}
    # 실제 반응일(발표 다음 거래일)에는 4분기 기록이 쌓인 뒤 거의 항상 노출이 잡힌다
    rx = risk.reaction_dates(st_, sess)
    hit = [full[d] >= 0.25 for d in rx[6:] - pd.offsets.BDay(1) if d in full.index]
    assert np.mean(hit) > 0.9
    # 첫 발표 반응 전에는 노출 0 (과거 기록이 없으면 투영도 없다)
    assert (full[:rx[0] - pd.Timedelta(days=1)] == 0).all()
    # s0 이후의 발표 기록을 지우거나 엉뚱한 미래 발표를 넣어도 s0 까지의 값은 같다
    s0 = pd.Timestamp("2022-03-15")
    cut = {k: [x for x in v if x.normalize() <= s0] + [pd.Timestamp("2022-03-16 08:00"), pd.Timestamp("2023-01-03 08:00")]
           for k, v in {"A": st_, "C": st_[2:]}.items()}
    part = risk.earnings_exposure(cut, w, sess)
    pd.testing.assert_series_equal(full[:s0], part[:s0])
    assert not full[s0 + pd.Timedelta(days=1):].equals(part[s0 + pd.Timedelta(days=1):])


def test_harx_picks_up_scheduled_earnings_variance():
    """분기 반응일에 분산이 16배인 자료: HAR-X 의 실적 계수가 양수이고, 반응일 전날의 σ̂ 가 HAR 보다 크다."""
    sess = pd.bdate_range("2017-01-02", periods=2200)
    st_ = _quarterly_stamps("2017-01-26", 36)
    rx = risk.reaction_dates(st_, sess)
    rng = np.random.default_rng(3)
    sd = pd.Series(0.01, index=sess)
    sd[sd.index.isin(rx)] = 0.04
    r = pd.Series(rng.standard_normal(len(sess)), index=sess) * sd
    earn = risk.earnings_exposure({"A": st_}, {"A": 1.0}, sess)
    gx = risk.har_fit(r.iloc[:2000], earn, window=1500)
    assert gx.b[gx.cols.index("earn") + 1] > 0.3
    bt = risk.backtest_var(r, 0.99, 250, n_sim=0, earn=earn)
    pre = [d for d in rx if d in bt.forecasts.index]
    assert (bt.forecasts.loc[pre, "HAR-X"] > bt.forecasts.loc[pre, "HAR"]).mean() > 0.8


def test_har_backtest_has_no_lookahead_and_keeps_pairing():
    r = _garch_sim(1100, seed=21)
    r.index = pd.bdate_range("2019-01-01", periods=len(r))
    earn = risk.earnings_exposure({"A": _quarterly_stamps("2019-01-24", 20)}, {"A": 0.3}, r.index)
    b = risk.backtest_var(r, 0.99, 250, n_sim=200, earn=earn)
    # 검정일은 모든 방법이 같다: max(window, garch_window) = 500 부터, 빈칸 없음
    assert {"HAR", "HAR-X"} <= set(b.forecasts.columns) and list(b.es.columns) == list(b.forecasts.columns)
    assert b.forecasts.index.equals(r.index[500:]) and not b.forecasts.isna().any().any()
    assert (b.summary["표본일"] == 600).all() and b.summary.loc["HAR-X", "Z2 p"] == b.summary.loc["HAR-X", "Z2 p"]
    assert (b.es[["HAR", "HAR-X"]] >= b.forecasts[["HAR", "HAR-X"]] - 1e-12).all().all()
    # 실적 노출을 넣어도 다른 방법의 예측은 그대로다
    b0 = risk.backtest_var(r, 0.99, 250, n_sim=0)
    other = [m for m in b.forecasts.columns if m != "HAR-X"]
    pd.testing.assert_frame_equal(b.forecasts[other], b0.forecasts[other])
    # k 번째 날부터 수익률을 바꿔도 그날까지의 예측(그날 아침 = k−1 까지의 자료)은 변하지 않는다
    k = 800
    r2 = r.copy()
    r2.iloc[k:] = r2.iloc[k:] * 3 + 0.01
    b2 = risk.backtest_var(r2, 0.99, 250, n_sim=0, earn=earn)
    upto = r.index[k]
    pd.testing.assert_frame_equal(b.forecasts.loc[:upto], b2.forecasts.loc[:upto])
    pd.testing.assert_frame_equal(b.es.loc[:upto], b2.es.loc[:upto])
    assert not np.allclose(b.forecasts.loc[upto:, "HAR-X"].iloc[2:], b2.forecasts.loc[upto:, "HAR-X"].iloc[2:])
    # har_fit 을 그날까지 자르면 백테스트와 같은 σ̂ — 단 계수는 refit 일에만 새로 적합하므로 refit 일에서 비교
    d = r.index[500 + 20 * 10]
    g = risk.har_fit(r.loc[:d].iloc[:-1], earn, window=500)
    assert abs(g.sigma_next - b.sigma.loc[d, "HAR-X"]) < 1e-12


def test_liquidity_skips_us_holiday_row():
    """KAN-29: 환율이 거래한 미국 휴장일(추수감사절)에 미국 종목 줄이 NaN 이어도
    마지막 거래일 종가와 자기 거래일 20일 평균 거래량을 써야 한다."""
    idx = pd.bdate_range("2026-10-26", "2026-11-26")  # 마지막 날 = 2026-11-26 (미국 휴장)
    close = pd.DataFrame({"AAA": 100.0, "BBB": 50.0, "KRW=X": 1340.0}, index=idx)
    volume = pd.DataFrame({"AAA": 1_000.0, "BBB": 4_000.0, "KRW=X": 0.0}, index=idx)
    volume.iloc[-22, 0] = 1_000_000.0  # 휴장 줄 앞 21번째 거래일 — 20일 창에 들어오면 평균이 튄다
    close.loc[idx[-1], ["AAA", "BBB"]] = np.nan
    volume.loc[idx[-1], ["AAA", "BBB"]] = np.nan
    close.loc[idx[-2], "AAA"] = 101.0

    px, adv = risk.liquidity_inputs(close, volume, ["AAA", "BBB"])
    assert px["AAA"] == 101.0 and px["BBB"] == 50.0
    assert adv["AAA"] == 1_000.0  # 휴장 줄을 빼고 실제 20거래일만 — 21거래일 전 거래량은 창 밖
    assert adv["BBB"] == 4_000.0

    # 예전처럼 마지막 줄을 그대로 쓰면 청산일이 전부 NaN 이었다
    stale = risk.liquidity({"AAA": 0.6, "BBB": 0.4}, close[["AAA", "BBB"]].iloc[-1], adv, 1e6)
    assert stale["청산일"].isna().all()
    lq = risk.liquidity({"AAA": 0.6, "BBB": 0.4}, px, adv, 1e6, 0.2)
    assert lq["청산일"].notna().all()
    assert abs(lq.loc["AAA", "청산일"] - 600_000 / (101_000 * 0.2)) < 1e-9
    assert lq["청산일"].idxmax() == "AAA"


def test_liquidity_missing_volume_is_nan_not_inf():
    lq = risk.liquidity({"A": 1.0, "B": 0.0}, pd.Series({"A": 10.0, "B": 10.0}),
                        pd.Series({"A": float("nan"), "B": 0.0}), 1e6)
    assert lq["청산일"].isna().all()
