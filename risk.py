"""실무 리스크 지표 — 순수 계산만. 데이터 수집·화면은 여기 없다.

이 파일만 import 하면 브리핑(market_metrics.py)에서도 같은 숫자를 낼 수 있게
pandas·numpy 밖의 의존성은 없다 (scipy 없이 정규분포·카이제곱·t 분포는 math 로 푼다).

규약
- 수익률은 **일간 단순수익률** (소수, 0.01 = 1%).
- VaR·ES 는 **손실을 양수**로 낸다 (VaR 0.023 = "하루에 2.3% 넘게 잃을 확률이 α").
- h일 VaR 는 1일 값에 √h 를 곱한다 — i.i.d. 가정이다. 꼬리가 뭉치는(변동성 군집)
  구간에서는 과소추정한다. 백테스트(kupiec/christoffersen)가 그걸 잡는다.
- 현금 = 1 − Σw, 수익률 0 으로 본다 (레버리지 금지 제약이라 Σw ≤ 1).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

_N = NormalDist()


# ── 포트폴리오 수익률 ────────────────────────────────────────────────────

def portfolio_returns(rets: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """고정 비중(매일 리밸런스 가정) 포트폴리오 일간 수익률."""
    w = pd.Series(weights, dtype=float)
    w = w[w.index.isin(rets.columns)]
    return (rets[w.index].fillna(0.0) * w).sum(axis=1)


# ── VaR / ES — 방법별 ───────────────────────────────────────────────────

def var_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """역사적 시뮬레이션. 분포 가정 없음 — 대신 창 안에 없던 사건은 모른다."""
    return float(-np.quantile(r.dropna(), 1 - alpha))


def es_historical(r: pd.Series, alpha: float = 0.99) -> float:
    """기대손실(ES/CVaR): VaR 를 넘은 날들의 평균 손실. 바젤 FRTB 는 97.5% ES 를 쓴다."""
    r = r.dropna()
    q = np.quantile(r, 1 - alpha)
    tail = r[r <= q]
    return float(-tail.mean()) if len(tail) else float("nan")


def var_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    """분산-공분산(정규). 가장 흔하고, 팻테일 때문에 99% 에서 가장 자주 틀린다."""
    r = r.dropna()
    return float(-(r.mean() + _N.inv_cdf(1 - alpha) * r.std()))


def es_parametric(r: pd.Series, alpha: float = 0.99) -> float:
    r = r.dropna()
    z = _N.inv_cdf(1 - alpha)
    return float(-(r.mean() - r.std() * _N.pdf(z) / (1 - alpha)))


def var_cornish_fisher(r: pd.Series, alpha: float = 0.99) -> float:
    """수정 VaR: 왜도·첨도로 정규 분위수를 보정한다 (Favre & Galeano 2002)."""
    r = r.dropna()
    z = _N.inv_cdf(1 - alpha)
    s, k = float(r.skew()), float(r.kurt())  # pandas kurt 는 초과첨도
    return float(-(r.mean() + _cf(z, s, k) * r.std()))


def _cf(z, s, k):
    """코니시-피셔 변환: 정규 분위수 z → 왜도 s·초과첨도 k 로 보정한 분위수."""
    return (z + (z**2 - 1) * s / 6 + (z**3 - 3 * z) * k / 24
            - (2 * z**3 - 5 * z) * s**2 / 36)


def _tail_z(alpha: float, k: int = 400) -> np.ndarray:
    """(0, 1−α) 를 k 등분한 중점의 정규 분위수 — 꼬리 평균을 수치적분할 때 쓴다."""
    p = 1 - alpha
    return np.array([_N.inv_cdf(p * (i + 0.5) / k) for i in range(k)])


def es_cornish_fisher(r: pd.Series, alpha: float = 0.99) -> float:
    """코니시-피셔 VaR 와 짝이 맞는 ES: ES = (1/(1−α))∫₀^{1−α} VaR_CF(u) du 를 중점법으로 적분.
    왜도·첨도가 아주 크면 CF 변환이 단조가 아니게 되어 이 값도 믿기 어렵다(VaR 와 같은 한계)."""
    r = r.dropna()
    s, k = float(r.skew()), float(r.kurt())
    return float(-(r.mean() + _cf(_tail_z(alpha), s, k).mean() * r.std()))


def ewma_vol(r: pd.Series, lam: float = 0.94) -> pd.Series:
    """RiskMetrics(1996) EWMA 변동성. σ²_t = λσ²_{t-1} + (1−λ)r²_{t-1}.
    값 t 는 t 일 **시작 시점**의 예측치다 (t 일 수익률을 쓰지 않는다)."""
    r = r.dropna()
    var = np.empty(len(r))
    var[0] = float(r.iloc[: min(20, len(r))].var())
    x = r.to_numpy()
    for t in range(1, len(r)):
        var[t] = lam * var[t - 1] + (1 - lam) * x[t - 1] ** 2
    return pd.Series(np.sqrt(var), index=r.index)


def var_ewma(r: pd.Series, alpha: float = 0.99, lam: float = 0.94) -> float:
    """내일 VaR — 오늘까지의 EWMA 로 한 스텝 앞을 예측한다. 평균은 0 으로 본다(실무 관행)."""
    r = r.dropna()
    s2 = lam * ewma_vol(r, lam).iloc[-1] ** 2 + (1 - lam) * r.iloc[-1] ** 2
    return float(-_N.inv_cdf(1 - alpha) * math.sqrt(s2))


def var_monte_carlo(rets: pd.DataFrame, weights: dict[str, float], alpha: float = 0.99,
                    n: int = 20_000, seed: int = 7, dof: float | None = 5.0) -> tuple[float, float]:
    """다변량 몬테카를로. dof 가 있으면 다변량 t (팻테일), None 이면 정규.
    (VaR, ES) 를 함께 낸다 — 같은 시뮬레이션에서 나와야 비교가 된다."""
    w = pd.Series(weights, dtype=float)
    w = w[w.index.isin(rets.columns)]
    R = rets[w.index].dropna()
    mu, cov = R.mean().to_numpy(), R.cov().to_numpy()
    rng = np.random.default_rng(seed)
    L = np.linalg.cholesky(cov + 1e-12 * np.eye(len(mu)))
    z = rng.standard_normal((n, len(mu))) @ L.T
    if dof:
        # t 를 공분산이 같게 스케일: X = Z·sqrt((ν−2)/χ²_ν)
        chi = rng.chisquare(dof, size=(n, 1))
        z = z * np.sqrt((dof - 2) / chi)
    pnl = (mu + z) @ w.to_numpy()
    q = np.quantile(pnl, 1 - alpha)
    return float(-q), float(-pnl[pnl <= q].mean())


METHODS = {
    "역사적": var_historical,
    "정규(분산-공분산)": var_parametric,
    "코니시-피셔": var_cornish_fisher,
    "EWMA λ=0.94": var_ewma,
}


def var_table(r: pd.Series, alphas=(0.95, 0.99), horizon: int = 1) -> pd.DataFrame:
    """한 수익률 시계열에 대해 방법 × 신뢰수준 VaR 표 (+ ES 두 개)."""
    rows = []
    h = math.sqrt(horizon)
    for name, f in METHODS.items():
        rows.append({"방법": name, **{f"VaR {a:.0%}": f(r, a) * h for a in alphas}})
    rows.append({"방법": "ES 역사적", **{f"VaR {a:.0%}": es_historical(r, a) * h for a in alphas}})
    rows.append({"방법": "ES 정규", **{f"VaR {a:.0%}": es_parametric(r, a) * h for a in alphas}})
    return pd.DataFrame(rows).set_index("방법")


# ── 리스크 분해 ──────────────────────────────────────────────────────────

def component_var(rets: pd.DataFrame, weights: dict[str, float], alpha: float = 0.99) -> pd.DataFrame:
    """정규 VaR 의 오일러 분해. 기여도의 합 = 포트폴리오 VaR (평균 무시).

    한계 VaR_i = z·(Σw)_i/σ_p — "이 종목 비중을 1%p 늘리면 VaR 가 얼마 느나".
    기여 VaR_i = w_i·한계 VaR_i. 비중보다 기여 비율이 크면 그 종목이 리스크를 끌고 있다.
    """
    w = pd.Series(weights, dtype=float)
    w = w[w.index.isin(rets.columns)]
    cov = rets[w.index].dropna().cov()
    sig_p = float(np.sqrt(w @ cov @ w))
    z = -_N.inv_cdf(1 - alpha)
    mvar = z * (cov @ w) / sig_p
    cvar = w * mvar
    return pd.DataFrame({
        "비중": w,
        "한계 VaR": mvar,
        "기여 VaR": cvar,
        "기여 비율": cvar / cvar.sum(),
        "개별 변동성(연)": np.sqrt(np.diag(cov)) * math.sqrt(252),
    }).sort_values("기여 VaR", ascending=False)


def concentration(weights: dict[str, float]) -> dict[str, float]:
    w = np.array([v for v in weights.values() if v > 0])
    s = w / w.sum()
    hhi = float((s**2).sum())
    return {"HHI": hhi, "유효 종목 수": 1 / hhi, "최대 비중": float(w.max()), "현금": 1 - float(w.sum())}


# ── VaR 백테스트 ────────────────────────────────────────────────────────

def _chi2_sf(x: float, df: int) -> float:
    """카이제곱 꼬리확률 — df 1, 2 만 필요해서 닫힌 꼴로 푼다."""
    if x <= 0:
        return 1.0
    if df == 1:
        return math.erfc(math.sqrt(x / 2))
    if df == 2:
        return math.exp(-x / 2)
    raise ValueError(df)


def _xlogy(x, y):
    return 0.0 if x == 0 else x * math.log(y)


def kupiec_pof(breaches: int, n: int, alpha: float) -> tuple[float, float]:
    """Kupiec(1995) 비율 검정: 위반 횟수가 1−α 에 맞나. (LR, p값)."""
    p = 1 - alpha
    phat = breaches / n if n else 0
    lr = -2 * ((_xlogy(n - breaches, 1 - p) + _xlogy(breaches, p))
               - (_xlogy(n - breaches, 1 - phat) + _xlogy(breaches, phat)))
    return lr, _chi2_sf(lr, 1)


def christoffersen_ind(hits: np.ndarray) -> tuple[float, float]:
    """Christoffersen(1998) 독립성 검정: 위반이 뭉쳐서 오나. (LR, p값)."""
    h = hits.astype(int)
    a, b = h[:-1], h[1:]
    n00 = int(((a == 0) & (b == 0)).sum()); n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum()); n11 = int(((a == 1) & (b == 1)).sum())
    p01 = n01 / (n00 + n01) if n00 + n01 else 0
    p11 = n11 / (n10 + n11) if n10 + n11 else 0
    p = (n01 + n11) / max(1, n00 + n01 + n10 + n11)
    l0 = _xlogy(n00 + n10, 1 - p) + _xlogy(n01 + n11, p)
    l1 = _xlogy(n00, 1 - p01) + _xlogy(n01, p01) + _xlogy(n10, 1 - p11) + _xlogy(n11, p11)
    lr = -2 * (l0 - l1)
    return lr, _chi2_sf(lr, 1)


@dataclass
class Backtest:
    forecasts: pd.DataFrame   # 날짜 × 방법, 그날 아침의 1일 VaR 예측 (양수)
    realized: pd.Series       # 그날 실현 수익률
    summary: pd.DataFrame
    es: pd.DataFrame | None = None     # 날짜 × 방법, 그날 아침의 1일 ES 예측 (양수)
    sigma: pd.DataFrame | None = None  # 날짜 × 방법, 예측 분포의 σ (McNeil–Frey 표준화용)


# ── ES 백테스트 ─────────────────────────────────────────────────────────
#
# VaR 검정은 '선을 넘은 횟수'만 본다. 넘은 날 **얼마나** 잃었는지는 ES 가 책임진다.
# 두 검정 모두 한쪽(ES 과소추정) 방향이다 — 리스크 관리에서 문제가 되는 쪽.
#
# Acerbi–Székely(2014) Z2 = Σ_t X_t·I_t / (T·(1−α)·ES_t) + 1.  X_t 는 수익률(손실 음수),
#   I_t 는 VaR 초과. 예측이 맞으면 E[Z2] = 0, ES 를 과소추정하면(또는 초과가 너무 잦으면) 음수.
#   p값은 **각 방법 자신의 예측분포에서** 실현값을 n_sim 번 다시 뽑아 Z2 귀무분포를 만들어 구한다
#   (논문의 권장 방식. 공표 임계값 −0.70(5%)·−1.8(0.01%)은 97.5%·T=250 전용이라 쓰지 않는다).
#   역사적 방법처럼 이산 분포면 귀무 하의 Z2 평균이 0 이 아닐 수 있는데, 시뮬레이션이 그걸 그대로 반영한다.
# McNeil–Frey(2000) 초과 잔차: VaR 초과일의 (손실 − ES)/σ 는 평균 0 이어야 한다.
#   중심화한 잔차를 복원추출하는 스튜던트화 부트스트랩(Efron–Tibshirani 1993 16.4)으로 H1: 평균 > 0.

# 각 방법의 ES 가 무엇인지 — 자연스러운 ES 가 없는 방법은 여기 적힌 대로 대신한다.
ES_BASIS = {
    "역사적": "창 안 VaR 이하 손실의 평균",
    "정규(분산-공분산)": "정규 ES (창 평균·표준편차)",
    "코니시-피셔": "CF 분위수를 꼬리 구간에서 적분",
    "EWMA λ=0.94": "정규 ES, σ = EWMA (평균 0)",
    "FHS-GARCH": "VaR 이하 표준화 잔차 평균 × σ_GARCH",
    "GARCH-t": "t_ν 닫힌 꼴 × σ_GARCH",
    "HAR": "VaR 이하 표준화 잔차 평균 × σ_HAR",
    "HAR-X": "VaR 이하 표준화 잔차 평균 × σ_HAR-X",
}


def es_z2(real, var, es, alpha: float) -> float:
    """Acerbi–Székely Z2. real 수익률, var·es 는 양수 손실 예측."""
    x, v, e = (np.asarray(a, float) for a in (real, var, es))
    return float(np.where(x < -v, x / e, 0.0).sum() / (len(x) * (1 - alpha)) + 1)


def _z2_many(draws: np.ndarray, v: np.ndarray, e: np.ndarray, alpha: float) -> np.ndarray:
    """draws (날짜, 시뮬레이션) 각 열의 Z2."""
    return np.where(draws < -v[:, None], draws / e[:, None], 0.0).sum(axis=0) / (len(v) * (1 - alpha)) + 1


def mcneil_frey(real, var, es, sigma, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, int]:
    """McNeil–Frey 초과 잔차 검정. (잔차 평균, 한쪽 p값, 초과 수). 초과가 2회 미만이면 p = NaN."""
    x, v, e, s = (np.asarray(a, float) for a in (real, var, es, sigma))
    hit = x < -v
    res = ((-x - e) / s)[hit]
    k = len(res)
    if k < 2:
        return (float(res.mean()) if k else float("nan")), float("nan"), k
    sd = res.std(ddof=1)
    if sd == 0:
        return float(res.mean()), float("nan"), k
    t = res.mean() / (sd / math.sqrt(k))
    c = res - res.mean()
    b = c[np.random.default_rng(seed).integers(0, k, (n_boot, k))]
    with np.errstate(divide="ignore", invalid="ignore"):
        tb = b.mean(axis=1) / (b.std(axis=1, ddof=1) / math.sqrt(k))
    return float(res.mean()), float((1 + (tb >= t).sum()) / (n_boot + 1)), k


GARCH_WINDOW = 500   # GARCH 계열 모수 추정 창 (backtest_var 설명 참고)


def backtest_var(r: pd.Series, alpha: float = 0.99, window: int = 250, refit: int = 20,
                 n_sim: int = 2000, seed: int = 0, garch_window: int | None = GARCH_WINDOW,
                 earn: pd.Series | None = None) -> Backtest:
    """롤링 창으로 매일 '어제까지의 데이터'로 오늘 VaR·ES 를 예측하고 위반을 센다.
    바젤 신호등: 250일 99% 기준 위반 0–4 녹색 · 5–9 황색 · 10+ 적색.
    n_sim: Z2 p값용 귀무 시뮬레이션 횟수 (0 이면 Z2 p 생략). seed 로 재현된다.

    garch_window: GARCH 계열(FHS-GARCH·GARCH-t) 모수 추정에 쓰는 창. None 이면 window 와 같다.
      250일로는 GARCH-t 의 σ 가 4–5% 낮게, ν 가 들쭉날쭉하게 추정돼(소표본 효과) 모형이 맞아도
      99% 에서 과소 커버한다 — 500일이면 명목 수준에 가깝다. 나머지 방법은 window 그대로 쓴다.
      FHS-GARCH 도 긴 창을 쓴다: 99% 분위수를 잔차 250개(꼬리 2–3점) 대신 500개에서 잡는 편이 낫다.
      t-GARCH 모의자료 20개(2,500일, ν=5)에서 99% Kupiec 기각률 — GARCH-t 15%→5%,
      FHS-GARCH 40%→20% (FHS 만 250일로 두면 50%). 97.5% 는 FHS 20%→0%.
    검정 기간은 모든 방법이 같다: 예측은 max(window, garch_window) 번째 날부터 시작하고, 그 앞의
      날은 어느 방법도 채점하지 않는다. 그래서 방법 간 비교는 같은 날짜끼리의 짝 비교로 남는다
      (대신 garch_window 가 길면 같은 데이터에서 검정일 수가 그만큼 줄어든다).
    HAR·HAR-X 도 garch_window 창으로 적합한다(계수 7–8개, 66일 지연 피처). 계수는 GARCH 와 같이 refit 일마다,
      σ̂·배율 c·표준화 잔차는 매일 갱신한다. earn: earnings_exposure(h=1) 를 r 날짜에 맞춘 것 — 값 s 는
      s+1 일 예측에 쓰인다. 없으면 HAR-X 는 실적 항 없이(레버리지·lv66 만) 돈다."""
    r = r.dropna()
    x = r.to_numpy()
    gw = int(garch_window or window)
    start = max(window, gw)
    n = len(r) - start
    idx = r.index[start:]
    names = list(METHODS) + ["FHS-GARCH", "GARCH-t", "HAR", "HAR-X"]
    V, E, S = ({m: np.empty(n) for m in names} for _ in range(3))
    ew = ewma_vol(r).to_numpy()            # ew[t] 는 t-1 까지로 만든 예측
    zq = -_N.inv_cdf(1 - alpha)
    zes = _N.pdf(zq) / (1 - alpha)         # 정규 ES 배수
    tz = _tail_z(alpha)
    # 귀무분포 시뮬레이션에 쓸 그날의 예측분포 모수
    mu_w, sd_w, sk_w, ku_w, mu_g, mu_t, nu_t = (np.empty(n) for _ in range(7))
    fz = np.empty((n, gw))                 # FHS 가 그날 쓴 표준화 잔차
    g = gt = None
    # HAR 계열: 피처는 한 번에 만든다 (행 s 는 s 까지의 자료만 쓴다). 적합 행은 _har_rows 가 고른다.
    F, hy, hf2, hfs = _har_design(r, earn, 1)
    HA = {m: np.column_stack([np.ones(len(r)), F[c].to_numpy()]) for m, c in (("HAR", HAR_COLS), ("HAR-X", HARX_COLS))}
    hb = dict.fromkeys(HA)
    hz = {m: np.full((n, gw), np.nan) for m in HA}   # 그날의 표준화 잔차 (오른쪽 정렬)
    hk = {m: np.zeros(n, int) for m in HA}
    for i in range(n):
        t = start + i
        past = r.iloc[t - window:t]
        pv = past.to_numpy()
        gpast = r.iloc[t - gw:t]            # GARCH 계열의 (더 긴) 추정 창
        if g is None or i % refit == 0:     # 모수만 20일마다 — σ 는 매일 갱신
            gr = _grid(gpast)
            g, gt = garch_fit(gpast, gr), garch_t_fit(gpast, gr)
            qt, et = t_std_var_es(alpha, gt.nu)
        gv = gpast.to_numpy()
        xp = gv - g.mu
        s2, nxt, _ = _filter(xp, g.omega, g.alpha, g.beta, float(xp.var()))
        z, sn = xp / np.sqrt(s2), math.sqrt(nxt)
        qz = np.quantile(z, 1 - alpha)
        V["FHS-GARCH"][i], E["FHS-GARCH"][i], S["FHS-GARCH"][i] = \
            -(g.mu + qz * sn), -(g.mu + z[z <= qz].mean() * sn), sn
        fz[i], mu_g[i] = z, g.mu
        xt = gv - gt.mu
        _, nxt, _ = _filter(xt, gt.omega, gt.alpha, gt.beta, float(xt.var()))
        st_ = math.sqrt(nxt)
        V["GARCH-t"][i], E["GARCH-t"][i], S["GARCH-t"][i] = -gt.mu + qt * st_, -gt.mu + et * st_, st_
        mu_t[i], nu_t[i] = gt.mu, gt.nu
        rows = _har_rows(t, gw)
        for m, A in HA.items():
            # 계수는 refit 일마다(첫날 포함) 다시 적합, 그 사이엔 고정
            hb[m], _, sn, z = _har_step(A, hy, hf2, hfs, rows, t - 1, 1, None if i % refit == 0 else hb[m])
            q = np.quantile(z, 1 - alpha)
            V[m][i], E[m][i], S[m][i] = -q * sn, -z[z <= q].mean() * sn, sn
            hz[m][i, gw - len(z):], hk[m][i] = z, len(z)
        m, sd, sk, ku = pv.mean(), pv.std(ddof=1), float(past.skew()), float(past.kurt())
        mu_w[i], sd_w[i], sk_w[i], ku_w[i] = m, sd, sk, ku
        V["역사적"][i], E["역사적"][i], S["역사적"][i] = var_historical(past, alpha), es_historical(past, alpha), sd
        V["정규(분산-공분산)"][i], E["정규(분산-공분산)"][i], S["정규(분산-공분산)"][i] = -(m - zq * sd), -(m - zes * sd), sd
        V["코니시-피셔"][i], E["코니시-피셔"][i], S["코니시-피셔"][i] = \
            -(m + _cf(-zq, sk, ku) * sd), -(m + _cf(tz, sk, ku).mean() * sd), sd
        V["EWMA λ=0.94"][i], E["EWMA λ=0.94"][i], S["EWMA λ=0.94"][i] = zq * ew[t], zes * ew[t], ew[t]

    def null_draws(name: str, rng: np.random.Generator, k: int) -> np.ndarray:
        """그 방법이 맞다고 할 때의 실현 수익률 k 세트 (날짜 × k)."""
        if name == "역사적":
            return x[(start - window + np.arange(n))[:, None] + rng.integers(0, window, (n, k))]
        if name == "FHS-GARCH":
            return mu_g[:, None] + S[name][:, None] * fz[np.arange(n)[:, None], rng.integers(0, gw, (n, k))]
        if name in HA:
            k_ = hk[name][:, None]
            j_ = gw - k_ + (rng.random((n, k)) * k_).astype(int)
            return S[name][:, None] * hz[name][np.arange(n)[:, None], j_]
        if name == "GARCH-t":
            nu = nu_t[:, None]
            return mu_t[:, None] + S[name][:, None] * np.sqrt((nu - 2) / nu) * rng.standard_t(nu, (n, k))
        zn = rng.standard_normal((n, k))
        if name == "코니시-피셔":
            return mu_w[:, None] + sd_w[:, None] * _cf(zn, sk_w[:, None], ku_w[:, None])
        if name == "EWMA λ=0.94":
            return S[name][:, None] * zn
        return mu_w[:, None] + sd_w[:, None] * zn

    real = pd.Series(x[start:], index=idx)
    rv = real.to_numpy()
    rows = []
    scale = 250 / n if n else 0
    for j, m in enumerate(names):
        hits = rv < -V[m]
        k = int(hits.sum())
        lr_pof, p_pof = kupiec_pof(k, n, alpha)
        lr_ind, p_ind = christoffersen_ind(hits)
        p_cc = _chi2_sf(lr_pof + lr_ind, 2)
        k250 = k * scale
        # 신호등 경계(0–4/5–9/10+)는 99% 전용으로 보정된 값이다. 95% 에선 정상 모델도 적색이 된다.
        if abs(alpha - 0.99) > 1e-9:
            light = "— (99% 전용)"
        else:
            light = "🟢 녹색" if k250 < 5 else ("🟡 황색" if k250 < 10 else "🔴 적색")
        z2 = es_z2(rv, V[m], E[m], alpha)
        p_z2 = float("nan")
        if n_sim:
            rng = np.random.default_rng(seed + j)
            sims = np.concatenate([_z2_many(null_draws(m, rng, b), V[m], E[m], alpha)
                                   for b in [250] * (n_sim // 250) + ([n_sim % 250] if n_sim % 250 else [])])
            p_z2 = float((1 + (sims <= z2).sum()) / (len(sims) + 1))
        mf, p_mf, _ = mcneil_frey(rv, V[m], E[m], S[m], seed=seed + j)
        ratio = float(-rv[hits].mean() / E[m][hits].mean()) if k else float("nan")
        rows.append({"방법": m, "위반": k, "기대": round(n * (1 - alpha), 1), "표본일": n,
                     "Kupiec p": p_pof, "독립성 p": p_ind, "조건부커버리지 p": p_cc,
                     "바젤(250일 환산)": light,
                     "초과손실/ES": ratio, "Z2": z2, "Z2 p": p_z2, "MF 잔차": mf, "MF p": p_mf,
                     "ES 산식": ES_BASIS[m]})
    fc = pd.DataFrame(V, index=idx)
    return Backtest(fc, real, pd.DataFrame(rows).set_index("방법"),
                    pd.DataFrame(E, index=idx), pd.DataFrame(S, index=idx))


# ── 스트레스 테스트 ─────────────────────────────────────────────────────

# 과거 사건 재현 — 기간 [start, end] 의 누적 수익률을 지금 비중에 그대로 적용한다.
# 날짜는 대표 지수(S&P500/ACWI)의 고점→저점 기준으로 널리 인용되는 구간이다.
HISTORICAL_SCENARIOS = {
    "리먼 파산 (2008-09-12→11-20)": ("2008-09-12", "2008-11-20"),
    "미국 신용등급 강등 (2011-07-22→08-10)": ("2011-07-22", "2011-08-10"),
    "위안화 절하 (2015-08-17→08-25)": ("2015-08-17", "2015-08-25"),
    "볼마겟돈 (2018-01-26→02-08)": ("2018-01-26", "2018-02-08"),
    "2018 Q4 긴축 공포 (2018-10-03→12-24)": ("2018-10-03", "2018-12-24"),
    "코로나 폭락 (2020-02-19→03-23)": ("2020-02-19", "2020-03-23"),
    "2022 금리 쇼크 (2022-01-03→06-16)": ("2022-01-03", "2022-06-16"),
    "SVB 사태 (2023-03-08→03-13)": ("2023-03-08", "2023-03-13"),
    "엔 캐리 청산 (2024-07-16→08-05)": ("2024-07-16", "2024-08-05"),
    "관세 쇼크 (2025-04-02→04-08)": ("2025-04-02", "2025-04-08"),
}

# CPI 발표일 하루 — 단기 보유 북의 최대 변수인 CPI 발표를 과거 같은 날로 본다.
CPI_DAYS = {
    "2022-06-10 (5월 CPI 8.6% · 예상 상회)": "2022-06-10",
    "2022-09-13 (8월 CPI · 예상 상회, S&P −4.3%)": "2022-09-13",
    "2022-11-10 (10월 CPI · 예상 하회, S&P +5.5%)": "2022-11-10",
    "2024-04-10 (3월 CPI · 예상 상회)": "2024-04-10",
    "2024-07-11 (6월 CPI · 예상 하회, 소형주 로테이션)": "2024-07-11",
}


def _window_return(px: pd.Series, start: str, end: str) -> float:
    """start 직전 종가 → end 종가 누적수익률. 데이터가 그 구간을 못 덮으면 NaN."""
    s = px.dropna()
    after = s.loc[:end]
    # start 당일부터의 움직임을 보려면 start 전날 종가가 기준이다. 그날 이전 데이터가
    # 없으면(상장 전) 구간 중간부터 잰 숫자는 거짓이라 NaN 으로 돌려 대리값을 쓰게 한다.
    base = s.loc[:pd.Timestamp(start) - pd.Timedelta(days=1)]
    if after.empty or after.index[-1] < pd.Timestamp(start) or base.empty:
        return float("nan")
    # 구간 양끝 근처에 가격이 없으면(상장폐지·데이터 공백) 낡은 가격으로 잰 셈이라 버린다
    if (pd.Timestamp(end) - after.index[-1]).days > 5 or \
            (pd.Timestamp(start) - base.index[-1]).days > 5:
        return float("nan")
    return float(after.iloc[-1] / base.iloc[-1] - 1)


def stress_replay(px: pd.DataFrame, weights: dict[str, float], bench: str,
                  scenarios: dict[str, tuple[str, str]], betas: pd.Series) -> pd.DataFrame:
    """과거 구간을 지금 북에 재생. 그때 상장 전이던 종목은 '베타 × 벤치 움직임'으로
    대리한다 — 대리 비중이 크면 그 시나리오 숫자는 믿을 게 못 된다(열로 같이 낸다)."""
    rows = []
    for name, (a, b) in scenarios.items():
        rb = _window_return(px[bench], a, b)
        if math.isnan(rb):
            continue
        pnl, proxied = 0.0, 0.0
        worst = (None, 0.0)
        for t, w in weights.items():
            ri = _window_return(px[t], a, b) if t in px else float("nan")
            if math.isnan(ri):
                ri = float(betas.get(t, 1.0)) * rb
                proxied += w
            pnl += w * ri
            if w * ri < worst[1]:
                worst = (t, w * ri)
        rows.append({"시나리오": name, "북": pnl, "벤치": rb, "상대(북−벤치)": pnl - rb,
                     "최악 기여": f"{worst[0]} {worst[1]:+.1%}" if worst[0] else "—",
                     "대리 비중": proxied})
    return pd.DataFrame(rows).set_index("시나리오") if rows else pd.DataFrame()


def stress_days(px: pd.DataFrame, weights: dict[str, float], bench: str,
                days: dict[str, str], betas: pd.Series) -> pd.DataFrame:
    return stress_replay(px, weights, bench, {k: (v, v) for k, v in days.items()}, betas)


def factor_betas(rets: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """종목별 다중회귀 베타 (OLS, 절편 포함). 행=종목, 열=팩터, +R² ."""
    F = factors.dropna()
    out = {}
    for c in rets.columns:
        y = rets[c].reindex(F.index).dropna()
        X = F.loc[y.index]
        if len(y) < 60:
            continue
        A = np.column_stack([np.ones(len(X)), X.to_numpy()])
        coef, *_ = np.linalg.lstsq(A, y.to_numpy(), rcond=None)
        resid = y.to_numpy() - A @ coef
        r2 = 1 - resid.var() / y.to_numpy().var() if y.var() > 0 else float("nan")
        out[c] = dict(zip(F.columns, coef[1:]), R2=r2)
    return pd.DataFrame(out).T


def factor_shock(betas: pd.DataFrame, weights: dict[str, float], bench: str,
                 shocks: dict[str, float]) -> pd.DataFrame:
    """가상 시나리오: 팩터 충격(수익률) → 선형 근사 손익. 잔차(고유위험)는 0 으로 둔다.
    R² 가 낮은 종목일수록 이 근사는 실제와 멀다."""
    s = pd.Series(shocks, dtype=float)
    cols = [c for c in s.index if c in betas.columns]
    move = betas[cols] @ s[cols]
    w = pd.Series(weights, dtype=float)
    w = w[w.index.isin(move.index)]
    book = float((w * move[w.index]).sum())
    b = float(move.get(bench, float("nan")))
    per = (w * move[w.index]).sort_values()
    return pd.DataFrame({"종목 영향": per}), book, b


# ── 유동성 ──────────────────────────────────────────────────────────────

KRW_SUFFIX = (".KS", ".KQ")


def liquidity_inputs(close: pd.DataFrame, volume: pd.DataFrame, tickers: list[str],
                     n: int = 20, krw_per_usd: float | None = None) -> tuple[pd.Series, pd.Series]:
    """종목마다 **자기 거래일**로 마지막 종가와 최근 n거래일 평균 거래량을 낸다.

    표의 마지막 줄을 그대로 쓰면 안 된다. 환율(KRW=X)·한국 주식처럼 미국 휴장일에도 거래하는
    열이 있으면 그날 줄이 생기고 미국 종목은 NaN 이다 — 추수감사절(2026-11-26)에 종가가 전부
    NaN 이 되어 청산일이 모두 NaN, 유동성 탭이 KeyError 로 죽었다. 거래량 창도 그 빈 줄을 하루로
    세어 19거래일 평균이 됐다.

    종가는 **달러**로 돌려준다. 한국 종목(.KS/.KQ)은 원화 종가라 그대로 두면 거래대금이 원화로
    나와 달러 보유 금액과 비교된다 — 거래대금이 ~1,400배 커 보여 청산일이 그만큼 짧게 나왔다
    (KAN-31). krw_per_usd 가 없으면 한국 종목 종가는 NaN — 틀린 숫자보다 빈칸이 낫다."""
    px, adv = {}, {}
    for t in tickers:
        c = close[t] if t in close else pd.Series(dtype=float)
        v = volume[t] if t in volume else pd.Series(dtype=float)
        traded = c.notna() & v.reindex(c.index).notna()
        px[t] = float(c[c.notna()].iloc[-1]) if c.notna().any() else float("nan")
        if t.upper().endswith(KRW_SUFFIX):
            px[t] = px[t] / krw_per_usd if krw_per_usd and krw_per_usd > 0 else float("nan")
        vv = v.reindex(c.index)[traded].iloc[-n:]
        adv[t] = float(vv.mean()) if len(vv) else float("nan")
    return pd.Series(px, dtype=float), pd.Series(adv, dtype=float)


def liquidity(weights: dict[str, float], last_px: pd.Series, adv_shares: pd.Series,
              nav: float = 1_000_000, participation: float = 0.20) -> pd.DataFrame:
    """청산 소요일 = 보유 금액 / (20일 평균 거래대금 × 참여율). 실무는 10~25% 를 쓴다.
    거래대금을 모르거나 0 이면 청산일은 NaN — 호출하는 쪽이 건너뛴다."""
    rows = []
    for t, w in weights.items():
        adv_usd = float(adv_shares.get(t, float("nan")) * last_px.get(t, float("nan")))
        ok = adv_usd > 0  # NaN 은 참으로 읽히므로 `if adv_usd` 로는 못 거른다
        pos = w * nav
        rows.append({"종목": t, "보유($)": pos, "ADV($)": adv_usd,
                     "ADV 대비": pos / adv_usd if ok else float("nan"),
                     "청산일": pos / (adv_usd * participation) if ok else float("nan")})
    return pd.DataFrame(rows).set_index("종목")


# ── 성과·리스크 요약 ────────────────────────────────────────────────────

def summary_stats(port: pd.Series, bench: pd.Series) -> dict[str, float]:
    i = port.index.intersection(bench.index)
    p, b = port.loc[i], bench.loc[i]
    ex = p - b
    ann = 252
    vol = float(p.std() * math.sqrt(ann))
    down = p[p < 0]
    dd_curve = (1 + p).cumprod()
    mdd = float((dd_curve / dd_curve.cummax() - 1).min())
    cagr = float(dd_curve.iloc[-1] ** (ann / len(p)) - 1) if len(p) else float("nan")
    beta = float(p.cov(b) / b.var()) if b.var() > 0 else float("nan")
    te = float(ex.std() * math.sqrt(ann))
    return {
        "연 변동성": vol,
        "베타(벤치)": beta,
        "추적오차(연)": te,
        "정보비율": float(ex.mean() * ann / te) if te else float("nan"),
        "소르티노(rf=0)": float(p.mean() * ann / (down.std() * math.sqrt(ann))) if len(down) > 1 else float("nan"),
        "최대낙폭": mdd,
        "칼마": cagr / abs(mdd) if mdd < 0 else float("nan"),
        "왜도": float(p.skew()),
        "초과첨도": float(p.kurt()),
    }


# ── GARCH(1,1) · 필터드 역사적 시뮬레이션(FHS) ─────────────────────────
#
# 백테스트에서 위반이 뭉쳐 왔다(독립성 기각) — 변동성이 군집한다는 뜻이다.
# 표준 처방: GARCH 로 오늘의 조건부 변동성 σ_t 를 추정하고, 과거 수익률을 그날의
# σ 로 나눈 표준화 잔차 z 를 '분포 모양'으로 쓴다(Barone-Adesi 1999). 모양은 역사적
# (팻테일 그대로), 크기는 지금 변동성. h일은 √h 대신 GARCH 를 따라 경로를 굴린다.
#
# scipy 없이 MLE 를 푼다: 분산 타기팅(ω = σ̄²(1−α−β))으로 모수를 둘로 줄이고
# (α, β) 격자를 벡터로 한 번에 평가한다. 격자 간격(α 0.01, β 0.01)이 추정오차보다 작다.

@dataclass
class Garch:
    omega: float
    alpha: float
    beta: float
    mu: float
    sigma: pd.Series     # σ_t — t 일 시작 시점 예측 (t−1 까지만 사용)
    z: pd.Series         # 표준화 잔차 (r_t − μ)/σ_t
    sigma_next: float    # 마지막 날 다음 거래일의 σ
    nu: float | None = None  # GARCH-t 의 자유도 (정규 준우도 적합이면 None)

    @property
    def persistence(self) -> float:
        return self.alpha + self.beta

    @property
    def long_run_vol(self) -> float:
        return math.sqrt(self.omega / max(1e-12, 1 - self.persistence))


_A = np.arange(0.01, 0.31, 0.01)
_B = np.arange(0.50, 0.995, 0.01)
_AA, _BB = np.meshgrid(_A, _B)
_OK = (_AA + _BB) < 0.995
_GA, _GB = _AA[_OK], _BB[_OK]


def _filter(x: np.ndarray, omega, a, b, s0):
    """σ² 재귀. omega·a·b 가 배열이면 격자 전체를 한 번에 돈다."""
    if np.ndim(omega) == 0:  # 모수 한 벌 — 백테스트가 매일 부르므로 0차원 배열 대신 float 로 돈다
        s2, om, a, b, ll, out = float(s0), float(omega), float(a), float(b), 0.0, []
        for xt in np.asarray(x, float).tolist():
            out.append(s2)
            ll += -0.5 * (math.log(s2) + xt * xt / s2)
            s2 = om + a * xt * xt + b * s2
        return np.array(out), s2, ll
    s2 = np.broadcast_to(np.asarray(s0, float), np.shape(omega)).copy()
    out = np.empty((len(x),) + np.shape(omega))
    ll = np.zeros(np.shape(omega))
    for t, xt in enumerate(x):
        out[t] = s2
        ll += -0.5 * (np.log(s2) + xt * xt / s2)
        s2 = omega + a * xt * xt + b * s2
    return out, s2, ll


def _grid(r: pd.Series):
    """(α, β) 격자 전체의 σ² 경로와 정규 로그우도. garch_fit·garch_t_fit 이 함께 쓴다."""
    r = r.dropna()
    mu = float(r.mean())
    x = r.to_numpy() - mu
    v = float(x.var())
    s2g, _, ll = _filter(x, v * (1 - _GA - _GB), _GA, _GB, v)
    return r, mu, x, v, s2g, ll


def garch_fit(r: pd.Series, grid=None) -> Garch:
    r, mu, x, v, _, ll = grid or _grid(r)
    i = int(np.argmax(ll))
    a, b = float(_GA[i]), float(_GB[i])
    s2, nxt, _ = _filter(x, v * (1 - a - b), a, b, v)
    sig = np.sqrt(s2)
    return Garch(v * (1 - a - b), a, b, mu, pd.Series(sig, r.index),
                 pd.Series(x / sig, r.index), float(math.sqrt(nxt)))


def var_fhs(r: pd.Series, alpha: float = 0.99, g: Garch | None = None) -> float:
    g = g or garch_fit(r)
    return float(-(g.mu + np.quantile(g.z, 1 - alpha) * g.sigma_next))


def es_fhs(r: pd.Series, alpha: float = 0.99, g: Garch | None = None) -> float:
    g = g or garch_fit(r)
    q = np.quantile(g.z, 1 - alpha)
    return float(-(g.mu + g.z[g.z <= q].mean() * g.sigma_next))


def simulate_fhs(g: Garch, h: int, n: int = 20_000, seed: int = 11) -> np.ndarray:
    """h 거래일 누적수익(합) n 개. 매일 z 를 복원추출하고 σ² 를 GARCH 로 갱신한다 —
    큰 손실 다음 날엔 변동성이 커지는 군집이 경로 안에 들어간다."""
    rng = np.random.default_rng(seed)
    z = g.z.to_numpy()
    s2 = np.full(n, g.sigma_next ** 2)
    tot = np.zeros(n)
    for _ in range(h):
        x = np.sqrt(s2) * rng.choice(z, n)
        tot += g.mu + x
        s2 = g.omega + g.alpha * x * x + g.beta * s2
    return tot


# ── GARCH-t: 스튜던트 t 혁신 ────────────────────────────────────────────
#
# FHS 는 꼬리 모양을 창 안의 잔차 250개에서 빌린다 — 99% 꼬리엔 점이 두세 개뿐이다.
# GARCH-t 는 꼬리 모양을 모수 하나(자유도 ν)로 요약한다: z_t ~ 분산 1 로 맞춘 t_ν.
# ν → ∞ 면 정규, ν = 4~6 이 주식 일간 수익률의 전형적 값이다.
#
# 추정은 같은 격자 방식을 넓힌다: 분산 타기팅 ω = σ̄²(1−α−β) 는 그대로 두고, (α, β) 격자 ×
# ν 격자에서 t 로그우도를 한 번에 평가한 뒤, 최적 (α, β) 에서 ν 만 황금분할로 연속 정밀화한다.
# σ² 경로는 ν 와 무관하므로 격자 전체에 대해 한 번만 돌린다 — 롤링 재추정에도 충분히 빠르다.
# scipy 없이 t 분포 CDF 는 정규화 불완전 베타 함수(연분수, Numerical Recipes 6.4)로 푼다.

_NU_GRID = np.array([2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0, 15.0, 20.0, 30.0, 50.0, 100.0])
NU_MAX = 200.0


def _betacf(a: float, b: float, x: float) -> float:
    """불완전 베타의 연분수 (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1)
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((a + m2 - 1) * (a + m2))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (a + b + m) * x / ((a + m2) * (a + m2 + 1))
        d = 1 + aa * d; d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / c; c = c if abs(c) > tiny else tiny
        dl = d * c
        h *= dl
        if abs(dl - 1) < 1e-14:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """정규화 불완전 베타 I_x(a, b)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def t_cdf(x: float, nu: float) -> float:
    """표준(분산 ν/(ν−2)) 스튜던트 t 의 CDF."""
    tail = 0.5 * _betainc(nu / 2, 0.5, nu / (nu + x * x))
    return tail if x < 0 else 1 - tail


def t_pdf(x: float, nu: float) -> float:
    return math.exp(math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(nu * math.pi)
                    - (nu + 1) / 2 * math.log1p(x * x / nu))


def t_ppf(p: float, nu: float) -> float:
    """t_ν 분위수. 이분법 — 단조라 안전하고, ν 는 재추정 때만 바뀌어 호출이 드물다."""
    if p == 0.5:
        return 0.0
    if p > 0.5:
        return -t_ppf(1 - p, nu)
    lo, hi = -1.0, 0.0
    while t_cdf(lo, nu) > p:   # 왼쪽 끝을 넓혀 가며 감싼다
        lo *= 2
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_cdf(mid, nu) > p:
            hi = mid
        else:
            lo = mid
        if hi - lo < 1e-12:
            break
    return 0.5 * (lo + hi)


def t_std_var_es(alpha: float, nu: float) -> tuple[float, float]:
    """분산 1 로 맞춘 t_ν 의 왼쪽 꼬리 (VaR, ES) — 손실 양수. 닫힌 꼴:
    q = |t_ν⁻¹(1−α)|,  ES_t = g_ν(q)/(1−α)·(ν + q²)/(ν − 1),  둘 다 √((ν−2)/ν) 를 곱한다."""
    p = 1 - alpha
    q = -t_ppf(p, nu)
    es = t_pdf(q, nu) / p * (nu + q * q) / (nu - 1)
    k = math.sqrt((nu - 2) / nu)
    return q * k, es * k


def _t_ll(u: np.ndarray, logs2_sum: np.ndarray, n: int, nu: float) -> np.ndarray:
    """표준화 t 로그우도 (상수 포함). u = x²/σ² (시간 × 격자), logs2_sum = Σ log σ²."""
    c = math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2) - 0.5 * math.log(math.pi * (nu - 2))
    return n * c - 0.5 * logs2_sum - (nu + 1) / 2 * np.log1p(u / (nu - 2)).sum(axis=0)


def garch_t_fit(r: pd.Series, grid=None) -> Garch:
    """GARCH(1,1)-t. 반환 Garch 의 nu 가 채워진다. z 는 분산 1 기준 표준화 잔차.
    grid: _grid(r) 결과를 넘기면 σ² 격자 경로(ν 와 무관)를 다시 돌리지 않는다."""
    r, mu, x, v, s2g, _ = grid or _grid(r)
    n = len(x)
    u = (x * x)[:, None] / s2g
    ls = np.log(s2g).sum(axis=0)
    # (α, β) 우도면은 ν 에 따라 거의 움직이지 않는다: ν 3개로 전체 격자를 훑어 후보 (α, β) 를
    # 추린 뒤, 후보에서만 ν 격자 전체를 본다 (전체 격자 × ν 16개의 약 1/5 비용).
    cand = np.unique(np.concatenate([np.argsort(_t_ll(u, ls, n, nu))[-40:] for nu in (4.0, 8.0, 30.0)]))
    uc, lc = u[:, cand], ls[cand]
    best = (-np.inf, 0, 0.0)
    for nu in _NU_GRID:
        ll = _t_ll(uc, lc, n, nu)
        j = int(np.argmax(ll))
        if ll[j] > best[0]:
            best = (float(ll[j]), int(cand[j]), float(nu))
    i = best[1]
    ui, li = u[:, i:i + 1], ls[i:i + 1]
    # ν 정밀화: log(ν−2) 위에서 황금분할 (격자 이웃 사이)
    f = lambda y: -float(_t_ll(ui, li, n, 2 + math.exp(y))[0])
    k = int(np.searchsorted(_NU_GRID, best[2]))
    lo = math.log((_NU_GRID[k - 1] if k > 0 else 2.05) - 2)
    hi = math.log((_NU_GRID[k + 1] if k + 1 < len(_NU_GRID) else NU_MAX) - 2)
    gr = (math.sqrt(5) - 1) / 2
    c1, c2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = f(c1), f(c2)
    for _ in range(40):
        if f1 < f2:
            hi, c2, f2 = c2, c1, f1
            c1 = hi - gr * (hi - lo); f1 = f(c1)
        else:
            lo, c1, f1 = c1, c2, f2
            c2 = lo + gr * (hi - lo); f2 = f(c2)
    nu = 2 + math.exp(0.5 * (lo + hi))
    a, b = float(_GA[i]), float(_GB[i])
    s2, nxt, _ = _filter(x, v * (1 - a - b), a, b, v)
    sig = np.sqrt(s2)
    return Garch(v * (1 - a - b), a, b, mu, pd.Series(sig, r.index),
                 pd.Series(x / sig, r.index), float(math.sqrt(nxt)), nu)


def var_garch_t(r: pd.Series, alpha: float = 0.99, g: Garch | None = None) -> float:
    g = g or garch_t_fit(r)
    return float(-g.mu + t_std_var_es(alpha, g.nu)[0] * g.sigma_next)


def es_garch_t(r: pd.Series, alpha: float = 0.99, g: Garch | None = None) -> float:
    g = g or garch_t_fit(r)
    return float(-g.mu + t_std_var_es(alpha, g.nu)[1] * g.sigma_next)


def simulate_garch_t(g: Garch, h: int, n: int = 20_000, seed: int = 11) -> np.ndarray:
    """simulate_fhs 와 같지만 z 를 잔차 복원추출 대신 분산 1 t_ν 에서 뽑는다 (h일 누적수익 n 개)."""
    rng = np.random.default_rng(seed)
    k = math.sqrt((g.nu - 2) / g.nu)
    s2 = np.full(n, g.sigma_next ** 2)
    tot = np.zeros(n)
    for _ in range(h):
        x = np.sqrt(s2) * rng.standard_t(g.nu, n) * k
        tot += g.mu + x
        s2 = g.omega + g.alpha * x * x + g.beta * s2
    return tot


# ── HAR / HAR-X: 실현분산 회귀로 내일 σ 를 예측한다 ─────────────────────
#
# HAR(Corsi 2009): 내일 분산의 로그를 어제·지난 5일·지난 22일 평균 분산의 로그에 OLS 로 회귀한다.
#   장중 자료가 없어 그날의 '실현분산'은 r²(일간 수익률 제곱) 하나다 — 아주 시끄러운 대용치(log χ²₁)라
#   로그에 바닥이 필요하다: floor_s = HAR_FLOOR × (s 까지 66일 평균 r²). |r| 이 최근 σ 의 10% 보다 작은
#   날은 그 값으로 올린다. 바닥도 s 까지의 자료로만 정해진다.
#   예측 σ̂² = c·exp(ŷ). 로그 회귀의 예측은 기하평균이라 분산을 낮게 잡는다(젠센) — c = 적합 창의
#   mean(r²/exp ŷ) 로 맞춘다(quant_auto KAN-24 vol.py 와 같은 배율, QLIKE 를 최소로 하는 값).
#   VaR·ES = σ̂ × (적합 창 표준화 잔차 z = r/σ̂ 의 분위수·꼬리평균) — FHS 와 같은 방식, 평균 0.
#   (FHS 에서는 c 가 σ̂ 와 z 양쪽에서 상쇄된다. 정규·t 변형에서만 c 가 값을 바꾼다.)
#
# HAR-X = HAR + lv66 + 레버리지(어제 하락폭 neg = min(r,0)/σ₂₂, 5일 하방 준분산 lsd5) + 실적 노출 earn.
#   KAN-24(S&P 500 585종목 일간 패널, 2017–2026)에서 선형 HAR-X 는 HAR 보다 QLIKE 가 5.7%(h=1)·18.2%(h=5)
#   낮았고, 이득의 61–72% 가 실적 발표가 낀 창에서 나왔다(LightGBM 은 선형 위에 더한 것이 없다).
#   그러나 1일 99% VaR(필터드 HS) 위반율은 EWMA 1.09%, HAR-X 1.10%, HAR 1.31% — 예측 손실이 낫다고
#   VaR 커버리지가 그만큼 나아지지는 않았다. 여기서는 종목이 아니라 북(또는 벤치 대비) 수익률 하나에 적합한다.
#
# 🔴 실적 노출 earn_s = Σ w_i · 1{종목 i 의 '투영' 다음 반응일이 s 기준 [−3, h+2] 세션 안}.
#   투영은 s 까지 **이미 반응한** 발표만 쓴다 (vol.py 와 같은 규칙):
#     다음 = (4분기 전 반응일) + 364일. 기록이 4건 미만이거나 그 값이 마지막 반응일 + 30일 이하면 마지막 + 91일.
#   발표 시각 → 반응일은 reaction_candidates 로 정한다(개장 전이면 그날, 장중·마감 후면 다음 거래일,
#   00:00 이면 늦은 후보). 가격은 보지 않는다. 회사가 공지한 예정일은 쓰지 않는다 — 공지 시점 기록이 없어
#   과거 어느 날에 그 예정일을 알았는지 재현할 수 없다. 창 앞쪽 −3 은 투영일이 지났는데 아직 발표가 없는
#   경우(임박)를 잡는다. KAN-24 에서 투영 오차는 중앙값 1일, ±7일 안 94%.
#   한계: 종목별 첫 발표 기록 이전 구간은 노출 0 으로 들어간다(과소). Yahoo 기록은 6년 남짓이다.

HAR_FLOOR = 0.01
HAR_MIN_ROW = 65                      # lv66·바닥이 처음 정의되는 행 (66일째)
HAR_COLS = ["lv1", "lv5", "lv22"]
HARX_COLS = HAR_COLS + ["lv66", "neg", "lsd5", "earn"]
EARN_WINDOW = (-3, 2)                 # 노출 창: 투영 반응일 − s ∈ [−3, h + 2] 세션


def reaction_dates(stamps, trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """발표 시각들 → 반응 거래일 (정렬, 중복 제거). 시각이 00:00 이면 늦은 후보 — 가격을 보지 않는다."""
    out = []
    for ts in stamps:
        c = reaction_candidates(pd.Timestamp(ts), trading_days)
        if c:
            out.append(max(c))
    return pd.DatetimeIndex(sorted(set(out)))


def project_next(rx: pd.DatetimeIndex) -> pd.Timestamp | None:
    """이미 반응한 발표들(오름차순) → 다음 반응일 투영. 기록이 없으면 None."""
    if not len(rx):
        return None
    last = rx[-1]
    if len(rx) >= 4:
        ya = rx[-4] + pd.Timedelta(days=364)
        if ya > last + pd.Timedelta(days=30):
            return ya
    return last + pd.Timedelta(days=91)


def earnings_exposure(stamps: dict[str, list], weights: dict[str, float], sessions: pd.DatetimeIndex,
                      h: int = 1, calendar: pd.DatetimeIndex | None = None) -> pd.Series:
    """날짜 s 마다 '다음 h 거래일 앞뒤로 실적이 투영된 보유 종목의 비중 합' (위 🔴 규칙).
    stamps: 티커 → 발표 시각 목록 (data.earnings_dates_one). sessions: 값을 낼 거래일.
    calendar: 발표 → 반응일 변환과 세션 차에 쓸 거래일 달력(sessions 보다 길게, 예: 가격 표 전체 날짜).
      달력 시작 전의 발표는 버린다 — 첫 거래일로 잘못 붙지 않게.
    s 의 값은 s 종가까지 반응한 발표만으로 정해진다 — s 뒤의 발표 기록을 바꿔도 변하지 않는다."""
    sessions = pd.DatetimeIndex(sessions)
    out = np.zeros(len(sessions))
    if not len(sessions):
        return pd.Series(out, index=sessions)
    cal = sessions if calendar is None else sessions.union(pd.DatetimeIndex(calendar))
    # 투영일이 자료 끝을 넘어도 세션 차를 셀 수 있게 평일 달력을 덧붙인다 (미래 휴장일은 무시)
    cal = cal.union(pd.bdate_range(cal[-1] + pd.Timedelta(days=1), periods=400))
    pos_s = cal.get_indexer(sessions)
    lo, hi = EARN_WINDOW[0], h + EARN_WINDOW[1]
    for t, w in weights.items():
        if not w or not stamps.get(t):
            continue
        rx = reaction_dates([x for x in stamps[t] if pd.Timestamp(pd.Timestamp(x).date()) >= cal[0]], cal)
        if not len(rx):
            continue
        k = rx.searchsorted(sessions, side="right")          # s 까지 반응한 발표 수
        ppos = np.full(len(rx) + 1, -10**9)
        for kk in range(1, len(rx) + 1):
            ppos[kk] = cal.searchsorted(project_next(rx[:kk]))
        off = ppos[k] - pos_s
        out += w * ((k > 0) & (off >= lo) & (off <= hi))
    return pd.Series(out, index=sessions)


def _har_design(r: pd.Series, earn: pd.Series | None = None, h: int = 1):
    """행 s = s 종가까지 아는 피처, 목표 = s+1…s+h. (피처 표, 목표 log 분산, 앞 h일 평균 r², 앞 h일 수익 합).
    목표 열은 앞날의 값이다 — 쓰는 쪽이 s+h 가 이미 지난 행만 적합에 넣는다."""
    x = r.to_numpy(float)
    n = len(x)
    r2 = x * x
    s2 = pd.Series(r2)
    m5, m22, m66 = (s2.rolling(k).mean().to_numpy() for k in (5, 22, 66))
    sd5 = pd.Series(r2 * (x < 0)).rolling(5).mean().to_numpy()
    floor = HAR_FLOOR * m66
    with np.errstate(divide="ignore", invalid="ignore"):
        lg = lambda v: np.log(np.maximum(v, floor))
        F = pd.DataFrame({"lv1": lg(r2), "lv5": lg(m5), "lv22": lg(m22), "lv66": lg(m66),
                          "neg": np.minimum(x, 0) / np.sqrt(m22), "lsd5": lg(sd5),
                          "earn": (earn.reindex(r.index).fillna(0.0).to_numpy(float)
                                   if earn is not None else np.zeros(n))}, index=r.index)
        fwd2, fwds = np.full(n, np.nan), np.full(n, np.nan)
        if n > h:
            i = np.arange(n - h)
            c2, c1 = np.concatenate([[0.0], np.cumsum(r2)]), np.concatenate([[0.0], np.cumsum(x)])
            fwd2[i] = (c2[i + 1 + h] - c2[i + 1]) / h
            fwds[i] = c1[i + 1 + h] - c1[i + 1]
        y = np.log(np.maximum(fwd2, floor))
    return F, y, fwd2, fwds


def _har_rows(t: int, window: int, h: int = 1) -> np.ndarray:
    """t 일 아침(자료는 t−1 까지)에 적합할 수 있는 행: 목표 끝 s+h ≤ t−1, 최근 window 일 안."""
    return np.arange(max(HAR_MIN_ROW, t - window), t - h)


@dataclass
class Har:
    cols: list[str]
    b: np.ndarray           # 절편 + cols 계수 (log 분산 회귀)
    c: float                # 분산 배율 (젠센 보정)
    sigma_next: float       # 자료 다음 h 거래일 누적수익의 σ̂
    z: np.ndarray           # 적합 창 표준화 잔차 (h일 누적수익 / σ̂, 평균 0 가정)
    h: int = 1


def _har_step(A: np.ndarray, y: np.ndarray, fwd2: np.ndarray, fwds: np.ndarray, rows: np.ndarray,
              pred_row: int, h: int, b: np.ndarray | None = None):
    """한 번의 적합(b 가 없을 때)과 그날 σ̂·z. (b, c, σ̂, z)."""
    if b is None:
        b = np.linalg.lstsq(A[rows], y[rows], rcond=None)[0]
    e = np.exp(A[rows] @ b)
    c = float(np.mean(fwd2[rows] / e))
    z = fwds[rows] / np.sqrt(h * c * e)
    return b, c, math.sqrt(h * c * math.exp(float(A[pred_row] @ b))), z


def har_fit(r: pd.Series, earn: pd.Series | None = None, h: int = 1, x: bool = True,
            window: int = GARCH_WINDOW) -> Har:
    """최근 window 일로 HAR(x=False) 또는 HAR-X(x=True) 를 적합하고 다음 h 거래일 σ̂ 를 낸다.
    earn: earnings_exposure(…, h=h) — 같은 h 로 만든 것. 없으면 HAR-X 의 실적 항은 0 이다.
    h > 1 이면 목표가 앞 h일 평균 r²(직접 예측)이고 z 는 겹치는 h일 누적수익이다."""
    r = r.dropna()
    F, y, fwd2, fwds = _har_design(r, earn, h)
    cols = HARX_COLS if x else HAR_COLS
    A = np.column_stack([np.ones(len(r)), F[cols].to_numpy()])
    n = len(r)
    rows = _har_rows(n, window, h)
    if len(rows) < len(cols) + 10:
        raise ValueError(f"HAR 적합에 행이 부족하다 ({len(rows)})")
    b, c, sn, z = _har_step(A, y, fwd2, fwds, rows, n - 1, h)
    return Har(cols, b, c, sn, z, h)


def _t_nu(z: np.ndarray) -> float:
    """분산 1 t 의 ν 를 격자 최우추정 (σ 는 이미 나눠져 있다)."""
    u = (z * z)[:, None]
    ll = [float(_t_ll(u, np.zeros(1), len(z), nu)[0]) for nu in _NU_GRID]
    return float(_NU_GRID[int(np.argmax(ll))])


def har_var_es(g: Har, alpha: float = 0.99, dist: str = "fhs") -> tuple[float, float]:
    """(VaR, ES) 손실 양수, 평균 0. dist: 'fhs'(표준화 잔차 분위수) · 'normal' · 't'(잔차로 ν 추정)."""
    if dist == "fhs":
        q = np.quantile(g.z, 1 - alpha)
        return float(-q * g.sigma_next), float(-g.z[g.z <= q].mean() * g.sigma_next)
    if dist == "normal":
        zq = -_N.inv_cdf(1 - alpha)
        return zq * g.sigma_next, _N.pdf(zq) / (1 - alpha) * g.sigma_next
    if dist == "t":
        qv, qe = t_std_var_es(alpha, _t_nu(g.z))
        return qv * g.sigma_next, qe * g.sigma_next
    raise ValueError(dist)


# ── 실적 점프 ───────────────────────────────────────────────────────────
#
# 일간 수익률 분포로 만든 VaR 는 실적일을 '평범한 날 250개 중 4개'로 희석한다.
# 그런데 남은 며칠 안에 실적이 낀 종목은 그 하루가 리스크의 대부분이다.
# 그래서 ① 과거 실적 반응일을 평소 분포에서 빼고(이중계산 방지) ② 앞으로 실적이
# 낀 종목만 그 종목의 과거 실적일 '초과 반응'(종목 − β·벤치)을 따로 뽑아 더한다.

def _next_td(d: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp | None:
    pos = trading_days.searchsorted(d)
    return trading_days[pos] if pos < len(trading_days) else None


def reaction_candidates(ts: pd.Timestamp, trading_days: pd.DatetimeIndex) -> list[pd.Timestamp]:
    """발표 시각 → 가격이 반응하는 거래일 후보. 개장(09:30 ET) 전이면 그날, 장중·마감 후면
    다음 거래일. 시각이 00:00 이면 진짜 시각이 아니라 자리표시일 수 있어 두 날을 다 낸다."""
    t = ts.tz_convert("America/New_York") if ts.tzinfo else ts
    d = pd.Timestamp(t.date())
    same, nxt = _next_td(d, trading_days), _next_td(d + pd.Timedelta(days=1), trading_days)
    if t.hour == 0 and t.minute == 0:
        return [x for x in dict.fromkeys([same, nxt]) if x is not None]
    pick = nxt if t.hour * 60 + t.minute >= 9 * 60 + 30 else same
    return [pick] if pick is not None else []


def reaction_day(ts: pd.Timestamp, trading_days: pd.DatetimeIndex,
                 abnormal: pd.Series | None = None) -> pd.Timestamp | None:
    """후보가 둘이면(시각 불명) 초과반응 |종목 − β·벤치| 이 큰 날을 고른다. 미래라 가격이
    없으면 앞 후보를 쓴다."""
    c = reaction_candidates(ts, trading_days)
    if len(c) > 1 and abnormal is not None:
        known = [x for x in c if x in abnormal.index and pd.notna(abnormal[x])]
        if known:
            return max(known, key=lambda x: abs(abnormal[x]))
    return c[0] if c else None


def earnings_jumps(stock: pd.Series, bench: pd.Series, days: list[pd.Timestamp], beta: float) -> pd.Series:
    """과거 실적 반응일의 초과수익 (종목 − β·벤치)."""
    out = {}
    for d in days:
        if d in stock.index and d in bench.index and pd.notna(stock[d]) and pd.notna(bench[d]):
            out[d] = float(stock[d] - beta * bench[d])
    return pd.Series(out, dtype=float).sort_index()


def strip_jumps(rets: pd.DataFrame, bench: pd.Series, days: dict[str, list], betas: dict[str, float]) -> pd.DataFrame:
    """실적 반응일의 종목 수익률을 β·벤치로 바꾼다 — '실적 없는 평소' 분포."""
    out = rets.copy()
    for t, ds in days.items():
        if t not in out:
            continue
        ix = [d for d in ds if d in out.index]
        out.loc[ix, t] = betas.get(t, 1.0) * bench.reindex(ix)
    return out


@dataclass
class Outcome:
    base: np.ndarray        # 실적 점프 없이 h일 상대수익 시뮬레이션
    total: np.ndarray       # + 실적 점프
    jumps: dict             # 티커 → (비중, 점프 표본)

    def stats(self, which: str = "total") -> dict:
        return horizon_stats(getattr(self, which))


def horizon_stats(s: np.ndarray, hurdles=(0.0, 0.01, 0.02, 0.05)) -> dict:
    """h일 상대수익 표본 → 순위 게임용 요약 (이길 확률·꼬리)."""
    s = np.asarray(s, float)
    q = np.quantile(s, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    v99 = -q[0]
    return {"평균": float(s.mean()), "중앙": float(q[3]),
            **{f"P(상대 > {x:+.0%})": float((s > x).mean()) for x in hurdles},
            "P(상대 < −2%)": float((s < -0.02).mean()),
            "VaR 95%": float(-q[1]), "VaR 99%": float(v99),
            "ES 97.5%": float(-s[s <= np.quantile(s, 0.025)].mean()),
            "5%": float(q[1]), "25%": float(q[2]), "75%": float(q[4]), "95%": float(q[5])}


def realized_horizon(r: pd.Series, h: int) -> np.ndarray:
    """과거 실현 h일 누적(합) — 겹치는 창. 모형 없이 '이 북을 들고 있었다면'의 분포.
    겹치는 창이라 표본이 서로 독립이 아니다(유효 표본 ≈ 길이/h)."""
    return r.dropna().rolling(h).sum().dropna().to_numpy()


def variance_ratio(r: pd.Series, h: int) -> float:
    """VR(h) = Var(h일 합) / (h·Var(1일)). 1 이면 √h 가 맞다. <1 이면 되돌림(평균회귀) —
    √h 와 GARCH 경로가 h일 리스크를 과대평가한다. >1 이면 추세 — 과소평가한다.
    겹치는 창은 표본이 작을수록 아래로 치우쳐서 Lo–MacKinlay(1988) 보정 분모를 쓴다."""
    x = r.dropna().to_numpy()
    n = len(x)
    if n < 2 * h + 2:
        return float("nan")
    mu = x.mean()
    var1 = ((x - mu) ** 2).sum() / (n - 1)
    s = np.convolve(x, np.ones(h), "valid")      # 겹치는 h일 합, n−h+1 개
    m = h * (n - h + 1) * (1 - h / n)
    return float(((s - h * mu) ** 2).sum() / m / var1)  # m 에 h 가 이미 들어 있다


MIN_JUMP_SAMPLES = 4


def deflate_jumps(js: np.ndarray, idio_sd: float) -> np.ndarray:
    """점프 표본에서 '그날의 평범한 고유 변동' 몫을 뺀다. 평소 분포(GARCH)가 그날도
    고유 변동을 이미 한 번 넣으므로, 안 빼면 그 하루치가 두 번 들어간다.
    평균은 두고 편차만 √((σ²_J − σ²_idio)/σ²_J) 로 줄인다."""
    js = np.asarray(js, float)
    v = js.var()
    if v <= 0:
        return js
    k = math.sqrt(max(0.0, v - idio_sd ** 2) / v)
    return js.mean() + (js - js.mean()) * k


def simulate_outcome(excess_ex: pd.Series, h: int, jumps: dict[str, tuple[float, np.ndarray]],
                     n: int = 20_000, seed: int = 11) -> Outcome:
    """남은 h 거래일 상대수익 분포 = FHS(실적 뺀 평소) + Σ w_i·(실적 초과반응 복원추출).
    점프는 같은 경로 안에서 서로 독립으로 뽑는다(종목 간 실적 반응 상관은 무시)."""
    g = garch_fit(excess_ex)
    base = simulate_fhs(g, h, n, seed)
    rng = np.random.default_rng(seed + 1)
    add = np.zeros(n)
    for t, (w, sample) in jumps.items():
        if len(sample) >= MIN_JUMP_SAMPLES:  # 표본 1~3개면 분포가 아니라 점 몇 개다
            add += w * rng.choice(np.asarray(sample), n)
    return Outcome(base, base + add, jumps)


# ── 성과 기여도 (attribution) ───────────────────────────────────────────
# 질문: "이 비중으로 기간 초에 사서 들고 있었다면, 벤치보다 왜 이만큼 앞섰나/뒤졌나."
# 규약
# - 매매 없음(buy-and-hold). 비중은 가격을 따라 흘러간다 — drift_weights.
# - 하루 단위로 나눈 뒤 **Carino 로그 연결**로 이어 붙인다. 일간 기여를 그냥 더하면
#   합이 기간 초과수익(Π(1+r_p) − Π(1+r_b))과 맞지 않는다. 연결 후에는 정확히 맞는다.
# - 모든 분해는 매일 Σ(항목) = r_p − r_b 가 되도록 짠다(잔차 항목을 명시해서).

def drift_weights(rets: pd.DataFrame, w0: dict[str, float]) -> pd.DataFrame:
    """각 날 **시작 시점**의 비중. 현금(1 − Σw)은 수익 0 으로 '현금' 열에 둔다."""
    w = pd.Series(w0, dtype=float)
    w = w[w.index.isin(rets.columns)]
    R = rets[w.index].fillna(0.0)
    val = (1 + R).cumprod().shift(1).fillna(1.0) * w          # 그날 시작 시점 평가액
    cash = 1.0 - float(w.sum())
    tot = val.sum(axis=1) + cash
    out = val.div(tot, axis=0)
    out["현금"] = cash / tot
    return out


def carino_scale(rp: pd.Series, rb: pd.Series) -> pd.Series:
    """일간 기여에 곱할 연결 계수 k_t / K (Carino 1999). Σ_t (k_t/K)·(r_p,t − r_b,t) = R_p − R_b."""
    def k(a, b):
        a, b = np.asarray(a, float), np.asarray(b, float)
        d = a - b
        with np.errstate(divide="ignore", invalid="ignore"):
            v = (np.log1p(a) - np.log1p(b)) / d
        return np.where(np.abs(d) < 1e-12, 1.0 / (1.0 + a), v)
    Rp, Rb = float((1 + rp).prod() - 1), float((1 + rb).prod() - 1)
    K = float(k(Rp, Rb))
    return pd.Series(k(rp.to_numpy(), rb.to_numpy()) / K, index=rp.index)


def link(daily: pd.DataFrame, rp: pd.Series, rb: pd.Series) -> pd.Series:
    """일간 기여 표(행=날, 열=항목)를 기간 기여로. 합 = 기간 초과수익."""
    return daily.mul(carino_scale(rp, rb), axis=0).sum()


def holding_attribution(rets: pd.DataFrame, w0: dict[str, float], bench_ret: pd.Series):
    """종목별 기여 (고정 비중으로 사서 보유). 반환 (기간 기여 Series, r_p, r_b)."""
    W = drift_weights(rets, w0)
    return holding_attribution_w(W, rets, bench_ret)


def holding_attribution_w(W: pd.DataFrame, rets: pd.DataFrame, bench_ret: pd.Series):
    """날짜별 비중 W(행=날, 열=종목·매매 단위, '현금' 열은 선택)로 종목별 기여를 낸다.
    c_i,t = w_i,t·(r_i,t − r_b,t), 현금은 w_cash·(0 − r_b). 매매가 있는 실제 포트폴리오용."""
    idx = W.index
    names = [c for c in W.columns if c != "현금"]
    cash = W["현금"] if "현금" in W.columns else 1.0 - W[names].sum(axis=1)
    rb = bench_ret.reindex(idx).fillna(0.0)
    R = rets[names].reindex(idx).fillna(0.0)
    rp = (W[names] * R).sum(axis=1)
    daily = W[names] * R.sub(rb, axis=0)
    daily["현금"] = cash * (0.0 - rb)
    return link(daily, rp, rb), rp, rb


def brinson_attribution(rets: pd.DataFrame, w0: dict[str, float], sectors: dict[str, str],
                        bench_ret: pd.Series, bench_w: dict[str, float], sector_ret: pd.DataFrame):
    """고정 비중 버전 — drift 후 brinson_attribution_w."""
    return brinson_attribution_w(drift_weights(rets, w0), rets, sectors, bench_ret, bench_w, sector_ret)


def brinson_attribution_w(W: pd.DataFrame, rets: pd.DataFrame, sectors: dict[str, str],
                          bench_ret: pd.Series, bench_w: dict[str, float], sector_ret: pd.DataFrame):
    """Brinson-Fachler 섹터 분해 (매일 계산 → Carino 연결).

    배분_s = (w_p,s − w_b,s)·(r_b,s − r_b*)   벤치보다 많이 담은 섹터가 벤치보다 잘 갔나
    선택_s =  w_p,s·(r_p,s − r_b,s)          그 섹터 안에서 고른 종목이 섹터보다 잘 갔나
                                               (상호작용 항 포함 — 실무에서 흔히 합친다)
    r_b* = Σ w_b,s,t·r_b,s 는 **섹터 ETF 로 다시 만든 벤치**다(w_b,s,t 는 끝 비중에서 되짚은 매일 비중). 실제 벤치와의 차이
    (r_b* − r_b)는 '벤치 재구성 오차'로 따로 둔다 — 숨기면 합이 안 맞는다.
    섹터 ETF 가 없는 섹터(분류 실패 등)의 종목은 r_b,s = r_b* 로 두어 배분 0, 선택만 잡힌다.
    현금은 w_b = 0, 수익 0 인 섹터다.

    반환 (섹터별 표, 기간 합계 dict). W 는 날짜별 비중('현금' 열 선택)."""
    idx = W.index
    if "현금" not in W.columns:
        W = W.assign(현금=1.0 - W.sum(axis=1))
    rb = bench_ret.reindex(idx).fillna(0.0)
    bw = pd.Series(bench_w, dtype=float)
    bw = bw[bw.index.isin(sector_ret.columns) & (bw > 0)]
    bw = bw / bw.sum()
    SR = sector_ret[bw.index].reindex(idx).fillna(0.0)
    # 벤치 섹터 비중은 **기간 끝의 현재값**만 안다. 시가총액 가중 벤치는 섹터 비중이 섹터 수익을
    # 따라 흘러가므로, 끝 비중에서 거꾸로 되짚어 매일 시작 비중을 만든다: w_s,t ∝ w_s,끝 / Π_{u≥t}(1+r_s,u).
    # 고정 비중을 1년 내내 쓰면 재구성 오차가 SPY 3.6%p·ACWI 4.6%p 였고, 되짚으면 1.4·2.5%p 로 준다(2026-11-09).
    grow_after = (1 + SR)[::-1].cumprod()[::-1]
    BW = (bw / grow_after)
    BW = BW.div(BW.sum(axis=1), axis=0)
    rb_star = (SR * BW).sum(axis=1)

    names = [c for c in W.columns if c != "현금"]
    R = rets[names].reindex(idx).fillna(0.0)
    sec_of = {t: sectors.get(t) or "미분류" for t in names}
    secs = sorted(set(sec_of.values()) | set(bw.index)) + ["현금"]
    rp = (W[names] * R).sum(axis=1)

    alloc, select, wp_s, rp_s = {}, {}, {}, {}
    for s in secs:
        if s == "현금":
            w_s, r_s = W["현금"], pd.Series(0.0, index=idx)
        else:
            mem = [t for t in names if sec_of[t] == s]
            w_s = W[mem].sum(axis=1) if mem else pd.Series(0.0, index=idx)
            ret_contrib = (W[mem] * R[mem]).sum(axis=1) if mem else pd.Series(0.0, index=idx)
            r_s = (ret_contrib / w_s).where(w_s > 0, 0.0)
        # 현금의 '벤치 섹터 수익'은 0 이다 → 배분 = w_cash·(0 − r_b*). 벤치가 오르면 현금은 깎는다.
        # (처음엔 SR 에 없는 섹터처럼 r_b* 를 넣어 현금 몫이 0 이 됐고, 합이 초과수익과 안 맞았다.)
        rbs = pd.Series(0.0, index=idx) if s == "현금" else (SR[s] if s in SR.columns else rb_star)
        w_b = BW[s] if s in BW.columns else 0.0
        alloc[s] = (w_s - w_b) * (rbs - rb_star)
        select[s] = w_s * (r_s - rbs) if s != "현금" else pd.Series(0.0, index=idx)
        wp_s[s], rp_s[s] = w_s, r_s

    A, S = pd.DataFrame(alloc), pd.DataFrame(select)
    daily = pd.concat({"배분": A, "선택": S}, axis=1)
    resid = rb_star - rb                                   # 매일: Σ배분+Σ선택 + 오차 = r_p − r_b
    scale = carino_scale(rp, rb)
    La = A.mul(scale, axis=0).sum()
    Ls = S.mul(scale, axis=0).sum()
    Lr = float((resid * scale).sum())

    period = lambda x: float((1 + x).prod() - 1)
    table_ = pd.DataFrame({
        "북 비중": {s: float(wp_s[s].mean()) for s in secs},          # 기간 평균
        "벤치 비중": {s: float(BW[s].mean()) if s in BW.columns else 0.0 for s in secs},     # 기간 평균
        "북 섹터 수익": {s: period(rp_s[s]) if float(wp_s[s].abs().sum()) > 0 else float("nan") for s in secs},
        "벤치 섹터 수익": {s: period(SR[s]) if s in SR.columns else float("nan") for s in secs},
        "배분": La, "선택": Ls,
    })
    table_["합계"] = table_["배분"] + table_["선택"]
    table_ = table_[(table_["북 비중"] > 0) | (table_["벤치 비중"] > 0)]
    totals = dict(북=period(rp), 벤치=period(rb), 재구성벤치=period(rb_star), 초과=period(rp) - period(rb),
                  배분=float(La.sum()), 선택=float(Ls.sum()), 재구성오차=Lr)
    return table_.sort_values("합계"), totals


def factor_attribution(rets: pd.DataFrame, w0: dict[str, float], bench_ret: pd.Series,
                       betas: pd.DataFrame, bench_betas: pd.Series, factors: pd.DataFrame):
    """고정 비중 버전 — drift 후 factor_attribution_w."""
    return factor_attribution_w(drift_weights(rets, w0), rets, bench_ret, betas, bench_betas, factors)


def factor_attribution_w(W: pd.DataFrame, rets: pd.DataFrame, bench_ret: pd.Series,
                         betas: pd.DataFrame, bench_betas: pd.Series, factors: pd.DataFrame):
    """요인별 기여: 매일 (북 베타 − 벤치 베타)·요인수익, 나머지는 '종목 고유'.

    🔴 베타는 **기간 시작 전** 자료로 추정해서 넘겨야 한다. 같은 기간으로 회귀하면 잔차가
    정의상 평균 0 에 가깝게 맞춰져 '종목 고유'가 작아 보인다(표본내 과적합).
    북 베타는 흘러가는 비중으로 매일 다시 가중한다. 현금 베타는 0.
    반환 (기간 기여 Series: 요인들 + '종목 고유', 일간 활성 베타 평균 Series).
    betas 의 행은 W 의 열 이름이다(매매 단위별로 넘기면 된다). 베타가 없는 열은 요인 노출 0 으로
    보고 그 수익은 전부 '종목 고유'로 간다."""
    idx = W.index
    names = [c for c in W.columns if c != "현금" and c in betas.index]
    fcols = [c for c in factors.columns if c in betas.columns and c in bench_betas.index]
    F = factors[fcols].reindex(idx).fillna(0.0)
    B = betas.loc[names, fcols]
    beta_p = W[names].to_numpy() @ B.to_numpy()               # (일, 요인)
    act = pd.DataFrame(beta_p - bench_betas[fcols].to_numpy(), index=idx, columns=fcols)
    allnames = [c for c in W.columns if c != "현금"]
    rp = (W[allnames] * rets[allnames].reindex(idx).fillna(0.0)).sum(axis=1)
    rb = bench_ret.reindex(idx).fillna(0.0)
    daily = act * F
    daily["종목 고유"] = (rp - rb) - daily.sum(axis=1)
    return link(daily, rp, rb), act.mean()


# ── 무작위 포트폴리오 검정 ────────────────────────────────────────────────
# 기여도 분해는 '어디서' 초과수익이 났는지는 말해 주지만, 그게 **실력인지 운인지**는 말하지 않는다.
# 같은 비중·같은 기간에 종목만 무작위로(같은 섹터 안에서) 바꾼 포트폴리오를 수만 개 만들어,
# 실제 포트폴리오가 그 분포의 어디에 서는지 본다(Burns 2004, 'Random portfolios for evaluating
# trading strategies'; Dawson & Young 2003). 분포의 위쪽 꼬리에 있어야 종목 고르기에 실력이 있다고
# 말할 수 있다. 섹터를 묶으면 섹터 배분 효과는 빼고 '섹터 안 종목 선택'만 검정한다.

@dataclass
class RandomTest:
    actual: float          # 실제 포트폴리오 기간 수익
    sims: np.ndarray       # 무작위 포트폴리오 기간 수익들
    pct: float             # 실제보다 낮은 무작위 비율 (= 백분위)
    p_value: float         # 실제 이상인 무작위 비율 (한쪽 검정)
    replaced: list         # 무작위로 바꾼 종목
    kept: list             # 그대로 둔 종목(ETF·해외·분류 없음)


def random_portfolio_test(weights: dict[str, float], own: pd.Series, pools: dict[str, np.ndarray | None],
                          cand: np.ndarray, n: int = 20_000, seed: int = 0) -> RandomTest:
    """weights: 종목 → 시작 비중(합 ≤ 1, 나머지는 현금 수익 0).
    own: 종목 → 실제 기간 수익(사서 그대로 보유).  cand: 후보 종목들의 같은 기간 수익.
    pools: 종목 → 바꿔 넣을 후보의 위치(cand 인덱스). None 이면 그 종목은 그대로 둔다.
    사서 보유하면 기간 수익 = Σ w·R 이 정확하므로(비중 표류 포함) 날마다 다시 계산할 필요가 없다."""
    rng = np.random.default_rng(seed)
    tick = list(weights)
    w = np.array([weights[t] for t in tick])
    R = np.array([float(own[t]) for t in tick])
    actual = float(w @ R)
    sims = np.zeros(n)
    replaced, kept = [], []
    for wi, Ri, t in zip(w, R, tick):
        pool = pools.get(t)
        if pool is None or len(pool) == 0:
            sims += wi * Ri
            kept.append(t)
        else:
            sims += wi * cand[rng.choice(pool, n)]
            replaced.append(t)
    return RandomTest(actual, sims, float((sims < actual).mean()), float((sims >= actual).mean()), replaced, kept)
