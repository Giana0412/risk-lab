"""Risk Lab — what could a stock portfolio be worth in a few weeks, and why.

Run:  streamlit run app.py          (?lang=en for English, ?lang=ko for Korean)

The front page is a calculator: holdings in, a few plain numbers out. The detailed
risk-desk analysis (VaR, backtests, stress tests, intraday, liquidity) sits behind
a toggle. Calculations live in risk.py, data in data.py, page components in ui.py.
Every visible string goes through T(korean, english).
"""
from __future__ import annotations

import math
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
import risk
import ui
from ui import pct, pp, money, num, p, h2, keys, figure, table_title, note, table, html_, esc

T = ui.t

# ── 언어: ?lang=en 이면 영어로 연다 ─────────────────────────────────────
_q = st.query_params.get("lang", "ko")
if "lang" not in st.session_state:
    st.session_state.lang = "en" if _q == "en" else "ko"
ui.set_lang(st.session_state.lang)

st.set_page_config(page_title=T("리스크 랩", "Risk Lab"), page_icon="▤", layout="centered",
                   initial_sidebar_state="collapsed")
ui.inject_css()
ui.chapter(0)


# ── 화면 표기용 이름 (계산 모듈의 내부 키는 한국어 그대로 둔다) ─────────────

METHOD_NAMES = {
    "역사적": T("역사적 시뮬레이션", "Historical simulation"),
    "정규(분산-공분산)": T("분산-공분산 (정규)", "Variance-covariance (normal)"),
    "코니시-피셔": T("코니시-피셔", "Cornish-Fisher"),
    "EWMA λ=0.94": "EWMA (λ=0.94)",
    "FHS-GARCH": "FHS-GARCH",
    "몬테카를로 정규": T("몬테카를로 (정규)", "Monte Carlo (normal)"),
    "몬테카를로 t": T("몬테카를로 (t, ν=5)", "Monte Carlo (t, ν=5)"),
    "ES 역사적": T("ES · 역사적", "ES · historical"),
    "ES 정규": T("ES · 정규", "ES · normal"),
}
SCEN_EN = {
    "리먼 파산 (2008-09-12→11-20)": "Lehman collapse (2008-09-12→11-20)",
    "미국 신용등급 강등 (2011-07-22→08-10)": "US downgrade (2011-07-22→08-10)",
    "위안화 절하 (2015-08-17→08-25)": "Yuan devaluation (2015-08-17→08-25)",
    "볼마겟돈 (2018-01-26→02-08)": "Volmageddon (2018-01-26→02-08)",
    "2018 Q4 긴축 공포 (2018-10-03→12-24)": "Q4 2018 tightening scare (2018-10-03→12-24)",
    "코로나 폭락 (2020-02-19→03-23)": "COVID crash (2020-02-19→03-23)",
    "2022 금리 쇼크 (2022-01-03→06-16)": "2022 rate shock (2022-01-03→06-16)",
    "SVB 사태 (2023-03-08→03-13)": "SVB collapse (2023-03-08→03-13)",
    "엔 캐리 청산 (2024-07-16→08-05)": "Yen carry unwind (2024-07-16→08-05)",
    "관세 쇼크 (2025-04-02→04-08)": "Tariff shock (2025-04-02→04-08)",
}
CPI_EN = {
    "2022-06-10 (5월 CPI 8.6% · 예상 상회)": "2022-06-10 (May CPI 8.6% · above consensus)",
    "2022-09-13 (8월 CPI · 예상 상회, S&P −4.3%)": "2022-09-13 (Aug CPI · above consensus, S&P −4.3%)",
    "2022-11-10 (10월 CPI · 예상 하회, S&P +5.5%)": "2022-11-10 (Oct CPI · below consensus, S&P +5.5%)",
    "2024-04-10 (3월 CPI · 예상 상회)": "2024-04-10 (Mar CPI · above consensus)",
    "2024-07-11 (6월 CPI · 예상 하회, 소형주 로테이션)": "2024-07-11 (Jun CPI · below consensus, small-cap rotation)",
}
FACTOR_EN = {
    "시장(ACWI)": "Market (ACWI)", "소형주 스프레드(IWM−SPY)": "Small-cap spread (IWM−SPY)",
    "장기채(TLT)": "Long bonds (TLT)", "유가(USO)": "Oil (USO)", "달러(UUP)": "Dollar (UUP)",
}
BASEL_EN = {"녹색": "Green", "황색": "Yellow", "적색": "Red"}


def scen(k: str) -> str:
    return (SCEN_EN.get(k) or CPI_EN.get(k) or k) if ui.LANG == "en" else k


def factor(k: str) -> str:
    return FACTOR_EN.get(k, k) if ui.LANG == "en" else k


def short(k: str) -> str:
    return scen(k).split(" (")[0]


def ro(word: str) -> str:
    """'로/으로' — 받침 있으면(ㄹ 제외) 으로."""
    c = word.strip()[-1:]
    if "가" <= c <= "힣":
        jong = (ord(c) - 0xAC00) % 28
        return word + ("으로" if jong not in (0, 8) else "로")
    return word + "로"


def mday(d) -> str:
    return f"{d:%b %-d}" if ui.LANG == "en" else f"{d:%-m월 %-d일}"


# ── 캐시된 데이터 ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=T("주가 기록을 받는 중이에요. 처음 한 번은 20초쯤 걸려요.",
                                         "Downloading price history. The first load takes about 20 seconds."))
def get_daily(tickers: tuple[str, ...]):
    return data.daily_prices(list(tickers))


@st.cache_data(ttl=6 * 3600, show_spinner=T("실적 발표 일정을 받는 중…", "Downloading earnings dates…"))
def _earnings_one(t: str):
    return data.earnings_dates_one(t)  # 예외는 캐시되지 않는다 — 실패는 다음 실행에 다시 시도


def get_earnings(tickers):
    out, failed = {}, {}
    for t in tickers:
        try:
            out[t] = _earnings_one(t)
        except Exception as e:
            failed[t] = f"{type(e).__name__}: {str(e)[:80]}"
    return out, failed


@st.cache_data(ttl=3600, show_spinner=T("VaR 백테스트 중…", "Backtesting VaR…"))
def cached_backtest(ser: pd.Series, alpha: float, window: int):
    return risk.backtest_var(ser, alpha, window)


@st.cache_data(ttl=24 * 3600, show_spinner=T("섹터 정보를 받는 중…", "Downloading sector data…"))
def get_sectors(tickers: tuple[str, ...], bench: str):
    return {t: data.stock_sector(t) for t in tickers}, data.bench_sector_weights(bench)


@st.cache_data(ttl=24 * 3600, show_spinner=T("후보 종목 가격을 받는 중… 1분 반쯤 걸린다.",
                                              "Downloading candidate prices… about 90 seconds."))
def get_universe(start: str, asof: str):
    """asof 는 캐시 키로만 쓴다 — 주가 자료에 새 날이 붙으면 후보 가격도 다시 받는다.
    S&P 1500 을 섹터 비율대로 1/3 만 뽑는다(전부 받으면 4분 넘게 걸린다)."""
    u = data.load_universe(frac=1 / 3)
    return u, data.universe_closes(u["ticker"].tolist(), start)


@st.cache_data(ttl=60, show_spinner=False)
def get_intraday(tickers: tuple[str, ...]):
    return data.intraday_prices(list(tickers))


# ════════════════════════════════════════════════════════════════════════
# 첫 화면 — 입력
# ════════════════════════════════════════════════════════════════════════

def _switch_lang():
    st.session_state.lang = "en" if st.session_state._lang_pick == "English" else "ko"
    st.query_params["lang"] = st.session_state.lang


c_brand, c_lang = st.columns([3, 1.3], vertical_alignment="center")
c_brand.markdown('<div class="q-top">Risk Lab</div>', unsafe_allow_html=True)
c_lang.radio("Language", ["한국어", "English"], index=1 if ui.LANG == "en" else 0, horizontal=True,
             key="_lang_pick", on_change=_switch_lang, label_visibility="collapsed")

html_(f'<div class="q-title">{T("내 주식, 몇 주 뒤엔 얼마가 될까?", "What could my stocks be worth in a few weeks?")}</div>'
      f'<div class="q-sub">{T("종목과 비중을 넣으면, 과거 주가가 움직인 방식대로 앞날을 2만 번 돌려 보고 "
                             "나올 만한 결과를 보여 줘요.",
                             "Enter your stocks and how much of each you hold. The app replays the way these "
                             "prices have moved 20,000 times and shows the range of likely outcomes.")}</div>')

# ① 종목
html_(f'<div class="q-step"><span>1</span>{T("무엇을 갖고 있나요?", "What do you hold?")}</div>')
books = data.load_books()
if books:
    def _book_label(b):
        kind = b.get("kind_en", b.get("kind", "")) if ui.LANG == "en" else b.get("kind", "")
        return kind if data.using_sample() else f"{b['date']} · {kind}"
    labels = [_book_label(b) for b in books]
    pick = st.selectbox(T("예시로 채우기", "Start from an example"), range(len(books)),
                        format_func=lambda i: labels[i])
    base_book = books[pick]["book"]
else:
    pick, base_book = None, {"SPY": 0.5, "QQQ": 0.5}

edited = st.data_editor(
    pd.DataFrame({"티커": list(base_book), "비중": [float(v) * 100 for v in base_book.values()]}),
    num_rows="dynamic", hide_index=True, width="stretch",
    key=f"book-{pick}",
    column_config={
        "티커": st.column_config.TextColumn(T("종목 코드", "Ticker"), width="medium",
                                           help=T("야후 파이낸스 코드. 미국 주식은 AAPL, 한국 주식은 005930.KS처럼",
                                                  "Yahoo Finance symbol, e.g. AAPL")),
        "비중": st.column_config.NumberColumn(T("비중 (%)", "Share (%)"), format="%.1f",
                                            min_value=0.0, max_value=100.0, step=0.1),
    },
)
weights = {str(t).strip().upper(): float(w) / 100 for t, w in zip(edited["티커"], edited["비중"])
           if isinstance(t, str) and t.strip() and pd.notna(w) and w > 0}
wsum = sum(weights.values())
if wsum > 1 + 1e-9:
    html_(f'<div class="q-hint bad">{T(f"비중을 합하면 {wsum:.0%}예요. 100% 이하로 맞춰 주세요.",
                                       f"The shares add up to {wsum:.0%}. Keep the total at 100% or less.")}</div>')
    st.stop()
html_(f'<div class="q-hint">{T("표를 눌러 고치거나 맨 아래 줄에 종목을 더할 수 있어요. ", "Click a cell to edit; add a stock in the bottom row. ")}'
      + (T(f"합계 {wsum:.0%}, 나머지 {1 - wsum:.0%}는 현금으로 계산해요.",
           f"Total {wsum:.0%}; the other {1 - wsum:.0%} is treated as cash.") if wsum < 1 - 1e-9 else
         T("합계 100%.", "Total 100%."))
      + '</div>')

# ② 조건
html_(f'<div class="q-step"><span>2</span>{T("조건", "Settings")}</div>')
PERIODS = {5: T("1주", "1 week"), 10: T("2주", "2 weeks"), 21: T("1개월", "1 month"), 63: T("3개월", "3 months")}
BENCHES = {"SPY": T("미국 S&P 500", "US S&P 500"), "ACWI": T("전 세계 주식", "Global stocks (ACWI)"),
           "QQQ": T("나스닥 100", "Nasdaq 100"), "_": T("직접 입력", "Other…")}
c1, c2, c3 = st.columns(3)
if ui.LANG == "ko":
    nav = c1.number_input("투자 금액 (원)", min_value=100_000, value=10_000_000, step=1_000_000, format="%d",
                          key="amt_ko")
else:
    nav = c1.number_input("Amount invested ($)", min_value=100, value=10_000, step=1_000, format="%d", key="amt_en")
c1.caption(money(nav))
H = c2.selectbox(T("얼마 동안", "For how long"), list(PERIODS), index=2, format_func=lambda k: PERIODS[k])
bpick = c3.selectbox(T("비교할 시장", "Compare with"), list(BENCHES), format_func=lambda k: BENCHES[k])
if bpick == "_":
    bench = st.text_input(T("비교할 지수나 ETF 코드", "Index or ETF ticker to compare with"), "ACWI").strip().upper()
    bench_name = bench
else:
    bench, bench_name = bpick, BENCHES[bpick]

if not weights:
    html_(f'<div class="q-hint">{T("종목을 하나 이상 넣어 주세요.", "Enter at least one stock.")}</div>')
    st.stop()

# 전문가용 설정 — 위젯은 아래 전문가 영역에 있지만 값은 여기서 먼저 쓴다
lookback = st.session_state.get("lookback", 250)
horizon = st.session_state.get("horizon", 1)

# ── 데이터 준비 ─────────────────────────────────────────────────────────

FX = "KRW=X"
all_tk = tuple(sorted(set(weights) | {bench} | set(data.FACTOR_TICKERS) | ({FX} if ui.LANG == "ko" else set())))
close, volume = get_daily(all_tk)
missing = [t for t in weights if t not in close or close[t].dropna().empty]
if missing:
    miss_s = ", ".join(missing)
    html_('<div class="q-hint bad">' + T(f"찾을 수 없는 종목 코드라 빼고 계산했어요: {miss_s}. "
                                         "미국 주식은 AAPL, 한국 주식은 005930.KS처럼 넣어 주세요.",
                                         f"Couldn’t find these tickers, so they were left out: {miss_s}.") + '</div>')
    weights = {t: w for t, w in weights.items() if t not in missing}
if bench not in close or close[bench].dropna().empty or not weights:
    st.error(T("주가를 받지 못했어요. 잠시 뒤 새로 고쳐 주세요.", "Couldn’t load prices. Please reload in a moment."))
    st.stop()
fx = float(close[FX].dropna().iloc[-1]) if FX in close and close[FX].notna().any() else 1400.0
nav_usd = nav / fx if ui.LANG == "ko" else nav  # 유동성 계산은 달러 거래대금과 비교한다

# 🔴 종목마다 **자기 거래일**로 수익률을 낸다. 표 전체로 pct_change 하면, 다른 열(환율 KRW=X,
# 한국 주식)이 거래한 미국 휴장일에 빈칸이 생기고 그 **다음 날 수익률까지 NaN** 이 되어
# dropna 로 사라진다 — 노동절 다음 날(2026-09-08)이 VaR 표본·기여도 기간에서 통째로 빠졌다.
rets_all = close.apply(lambda s: s.dropna().pct_change(fill_method=None))
hold = list(weights)
cols = list(dict.fromkeys(hold + [bench]))  # 비교 지수를 종목으로도 담으면 열이 중복된다
win = rets_all[cols].dropna().iloc[-lookback:]
if len(win) < 60:
    st.error(T(f"모든 종목이 함께 거래된 날이 {len(win)}일뿐이라 계산할 수 없어요. 최근 상장한 종목을 빼 보세요.",
               f"Only {len(win)} days when all stocks traded together, which is too few. Try removing a recent listing."))
    st.stop()

R = win[hold]
port = risk.portfolio_returns(R, weights)
bret = win[bench]
excess = port - bret
asof = win.index[-1].date()
h = math.sqrt(horizon)

betas_bench = {}
for t in hold:
    if t == bench:
        betas_bench[t] = 1.0
        continue
    x = rets_all[[t, bench]].dropna().iloc[-500:]
    betas_bench[t] = float(x[t].cov(x[bench]) / x[bench].var()) if len(x) > 60 else 1.0
betas_bench = pd.Series(betas_bench)

remaining = data.trading_days(pd.Timestamp(asof) + pd.Timedelta(days=1),
                              pd.Timestamp(asof) + pd.Timedelta(days=H * 2 + 20))[:H]

# ── 앞날 시뮬레이션 (첫 화면과 전문가 1장이 같이 쓴다) ──────────────────

earn, earn_failed = get_earnings(tuple(t for t in hold if t != bench))
future_days = close.index.append(remaining).unique().sort_values()
past_rx, next_rx, next_ann = {}, {}, {}
for t, ts_list in earn.items():
    abn = rets_all[t] - betas_bench[t] * rets_all[bench]
    rx = [risk.reaction_day(ts, future_days, abn) for ts in ts_list]
    past_rx[t] = [d for d in rx if d is not None and d <= pd.Timestamp(asof)]
    fut = [d for d in rx if d is not None and d > pd.Timestamp(asof)]
    next_rx[t] = min(fut) if fut else None
    ann = [ts for ts in ts_list if pd.Timestamp(ts.date()) > pd.Timestamp(asof)]
    next_ann[t] = min(ann) if ann else None

R_ex = risk.strip_jumps(R, bret, past_rx, betas_bench.to_dict())
port_ex = risk.portfolio_returns(R_ex, weights)
excess_ex = port_ex - bret
c_w, c_next, c_in, c_avg, c_lo, c_hi, c_n = (
    T("비중", "Weight"), T("다음 발표", "Next report"), T("평가 기간", "In window"),
    T("평균 |반응|", "Mean |reaction|"), T("최저", "Worst"), T("최고", "Best"), T("표본", "Obs."))
jump_rows, jumps_h, jumps_1, thin = [], {}, {}, []
for t in hold:
    if t == bench:
        continue
    js = risk.earnings_jumps(rets_all[t], rets_all[bench], past_rx.get(t, []), betas_bench[t])
    nd = next_rx.get(t)
    in_h = nd is not None and nd in remaining
    if in_h and len(js) >= risk.MIN_JUMP_SAMPLES:
        idio = (R_ex[t] - betas_bench[t] * bret).std()
        jd = risk.deflate_jumps(js.to_numpy(), float(idio))
        jumps_h[t] = (weights[t], jd)
        if nd == remaining[0]:
            jumps_1[t] = (weights[t], jd)
    elif in_h:
        thin.append((t, len(js)))
    jump_rows.append({
        "_t": t, c_w: weights[t],
        c_next: f"{next_ann[t]:%Y-%m-%d %H:%M}" if next_ann.get(t) is not None else T("미정", "TBA"),
        c_in: T("포함", "Yes") if in_h else "",
        c_avg: js.abs().mean() if len(js) else np.nan,
        c_lo: js.min() if len(js) else np.nan,
        c_hi: js.max() if len(js) else np.nan,
        c_n: len(js),
    })

out = risk.simulate_outcome(excess_ex, H, jumps_h)
s_tot, s_base = out.stats("total"), out.stats("base")
g = risk.garch_fit(excess_ex)
long_cols = rets_all[cols].dropna().iloc[-1000:]
ex_long = risk.portfolio_returns(long_cols[hold], weights) - long_cols[bench]
s_real = risk.horizon_stats(risk.realized_horizon(ex_long, H)) if len(ex_long) > H + 60 else None
vr = risk.variance_ratio(ex_long, H)
p_win, q05, q01 = s_tot["P(상대 > +0%)"], s_tot["5%"], -s_tot["VaR 99%"]

# 금액 결과는 평균 수익을 0으로 둔다 — 지난 1년 상승이 그대로 이어진다고 가정하지 않는다
out_abs = risk.simulate_outcome(port_ex - port_ex.mean(), H, jumps_h)
a_tot = out_abs.total  # 기간 누적수익 (일간 수익의 합)
a05, a01, a50 = (float(np.quantile(out_abs.total, q)) for q in (0.05, 0.01, 0.5))

# ════════════════════════════════════════════════════════════════════════
# 첫 화면 — 결과
# ════════════════════════════════════════════════════════════════════════

per = PERIODS[H]
html_(f'<div class="q-result"><div class="when">'
      + T(f"{asof:%Y년 %-m월 %-d일} 종가까지의 주가로 계산 · 기간 {per} ({H}거래일)",
          f"Based on prices up to the close on {asof:%B %-d, %Y} · {per} ({H} trading days)")
      + '</div></div>')


def after(r: float) -> str:
    return money(nav * (1 + r))


def delta(r: float) -> str:
    v = nav * r
    return ("+" if v >= 0 else "") + money(v)


ui.big([
    (T("보통이면", "Typical outcome"), T("가운데 결과", "the middle result"), after(a50), pct(a50, True, 1), False),
    (T("운이 나쁘면", "A bad stretch"), T("20번에 1번 꼴", "about 1 in 20"), after(a05),
     f"{delta(a05)} · {pct(a05, True, 1)}", True),
    (T("아주 나쁘면", "A very bad stretch"), T("100번에 1번 꼴", "about 1 in 100"), after(a01),
     f"{delta(a01)} · {pct(a01, True, 1)}", True),
])

lean = (T("시장보다 뒤처질 가능성이 더 커요.", "more likely to trail the market than beat it.") if p_win < 0.45 else
        T("동전 던지기와 비슷해요.", "about a coin flip.") if p_win < 0.55 else
        T("시장보다 앞설 가능성이 조금 더 커요.", "slightly more likely to beat the market."))
html_(f'<div class="q-line">' + T(
    f"{bench_name}보다 수익이 높을 확률은 <b>{p_win:.0%}</b>예요. {lean}",
    f"Chance of doing better than the {bench_name}: <b>{p_win:.0%}</b>, {lean}") + '</div>')

# 결과 분포 — 기간 끝 금액
cnt, edges = np.histogram(nav * (1 + a_tot), bins=60)
mids = (edges[:-1] + edges[1:]) / 2
lo, hi = np.quantile(nav * (1 + a_tot), [0.003, 0.997])
fig = go.Figure(go.Bar(
    x=mids, y=cnt, marker_color=[ui.BAD if m < nav else ui.BOOK for m in mids], marker_line_width=0,
    width=(edges[1] - edges[0]) * 0.85,
    customdata=[f"{money(a)} ~ {money(b)}" for a, b in zip(edges[:-1], edges[1:])],
    hovertemplate="%{customdata}<br>" + T("%{y:,}번", "%{y:,} runs") + "<extra></extra>"))
fig.add_vline(x=nav, line_color=ui.INK, line_width=1)
fig.add_annotation(x=nav, y=1, yref="paper", text=T(f"지금 {money(nav)}", f"Now {money(nav)}"),
                   showarrow=False, xanchor="left", yanchor="top", xshift=8, bgcolor="#fbfaf7", font=dict(size=12, color=ui.INK))
# 눈금은 원금 기준 깔끔한 % 간격에 — 946만 원 같은 어중간한 값을 피한다
span_r = (hi - lo) / nav
step = next(x for x in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5) if span_r / x <= 6)
ticks = [nav * (1 + k * step) for k in range(-50, 51) if lo <= nav * (1 + k * step) <= hi]
fig.update_xaxes(range=[lo, hi], tickvals=ticks, ticktext=[money(v).replace("&#36;", "$") for v in ticks])
fig.update_yaxes(showticklabels=False, showgrid=False)
html_(f'<div class="q-h" style="margin-top:22px">{T(f"{per} 뒤 금액, 2만 번 돌려 본 결과", f"Value after {per}: 20,000 simulated runs")}</div>')
html_(f'<div class="q-cap">{T("막대가 높을수록 자주 나온 금액이에요. 빨간 막대는 원금보다 줄어든 경우예요.", "Taller bars are more common outcomes. Red bars are runs that ended below what you put in.")}</div>')
st.plotly_chart(ui.layout(fig, 240, legend=False), width="stretch", config={"displayModeBar": False})

# ── 왜 그런가 ────────────────────────────────────────────────────────────

# 종목별 흔들림
if len(hold) > 1:
    cv = risk.component_var(R, weights, 0.99)
    top = cv.index[0]
    html_(f'<div class="q-h">{T("어느 종목이 가장 많이 흔들까요?", "Which stocks move the total most?")}</div>')
    html_('<div class="q-line">' + T(
        f"<b>{top}</b>의 비중은 {cv.loc[top, '비중']:.0%}인데, 전체 흔들림의 <b>{cv.loc[top, '기여 비율']:.0%}</b>를 만들어요."
        if cv.loc[top, "기여 비율"] > cv.loc[top, "비중"] * 1.2 else
        f"흔들림이 비중과 비슷하게 나뉘어 있어요. 가장 큰 몫은 <b>{top}</b>({cv.loc[top, '기여 비율']:.0%})예요.",
        f"<b>{top}</b> is {cv.loc[top, '비중']:.0%} of the money but drives <b>{cv.loc[top, '기여 비율']:.0%}</b> of the ups and downs."
        if cv.loc[top, "기여 비율"] > cv.loc[top, "비중"] * 1.2 else
        f"The ups and downs are spread roughly in line with the amounts held. The largest share is <b>{top}</b> "
        f"({cv.loc[top, '기여 비율']:.0%}).") + '</div>')
    order = list(cv.index)[::-1]
    fig = go.Figure()
    fig.add_bar(y=order, x=cv.loc[order, "비중"], orientation="h", name=T("넣은 돈의 비중", "Share of money"),
                marker_color="#c9c7c0", text=[f"{v:.0%}" for v in cv.loc[order, "비중"]], textposition="outside",
                cliponaxis=False, hovertemplate="%{y} %{x:.0%}<extra></extra>")
    fig.add_bar(y=order, x=cv.loc[order, "기여 비율"], orientation="h", name=T("흔들림의 몫", "Share of the swings"),
                marker_color=ui.BOOK, text=[f"{v:.0%}" for v in cv.loc[order, "기여 비율"]], textposition="outside",
                cliponaxis=False, hovertemplate="%{y} %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08, legend_traceorder="reversed")
    fig.update_xaxes(visible=False, range=[0, float(max(cv["기여 비율"].max(), cv["비중"].max())) * 1.25])
    fig.update_yaxes(showgrid=False, type="category")
    st.plotly_chart(ui.layout(fig, 70 + 46 * len(order)), width="stretch", config={"displayModeBar": False})
    html_(f'<div class="q-cap">{T("파란 막대가 회색보다 길면, 넣은 돈에 비해 값이 크게 출렁이거나 다른 종목과 같이 움직이는 종목이에요.", "A blue bar longer than the grey one means the stock swings more than its size suggests, or moves together with the others.")}</div>')

# 실적 발표
if jumps_h or thin:
    names = ", ".join(list(jumps_h) + [t for t, _ in thin])
    html_(f'<div class="q-h">{T("이 기간에 실적 발표가 있어요", "Earnings reports fall in this period")}</div>')
    html_('<div class="q-line">' + T(
        f"발표 예정: <b>{names}</b>. 실적 발표 날에는 주가가 평소보다 훨씬 크게 움직이곤 해서, 그 종목들의 과거 발표 날 움직임을 결과에 넣었어요.",
        f"Reporting: <b>{names}</b>. Prices often jump on earnings day, so each one's past earnings-day moves are built into the results.")
        + (T(" 다만 과거 기록이 적어 넣지 못한 종목이 있어요(" + ", ".join(t for t, _ in thin) + "). 실제 위험은 위보다 클 수 있어요.",
             " " + ", ".join(t for t, _ in thin) + " has too little history to include, so real risk may be larger.")
           if thin else "") + '</div>')
if earn_failed:
    html_(f'<div class="q-hint">{T("실적 발표 일정을 받지 못한 종목이 있어 그 영향은 빠졌어요: ", "Earnings dates unavailable, so their effect is left out: ")}'
          f'{", ".join(earn_failed)}</div>')

# 과거 위기
hist = risk.stress_replay(close, weights, bench, risk.HISTORICAL_SCENARIOS, betas_bench)
if not hist.empty:
    worst = hist["북"].idxmin()
    beat = int((hist["북"] > hist["벤치"]).sum())
    html_(f'<div class="q-h">{T("과거 위기 때 갖고 있었다면", "If you had held it through past crashes")}</div>')
    html_('<div class="q-line">' + T(
        f"가장 크게 잃은 때는 <b>{short(worst)}</b>{ro(short(worst))[len(short(worst)):]} {pct(hist.loc[worst, '북'], True, 0)}, "
        f"{money(nav)}이 {money(nav * (1 + hist.loc[worst, '북']))}이 됐을 거예요. "
        f"위기 {len(hist)}번 중 {beat}번은 {bench_name}보다 덜 잃었어요.",
        f"The worst was the <b>{short(worst)}</b>: {pct(hist.loc[worst, '북'], True, 0)}, which would have turned {money(nav)} into "
        f"{money(nav * (1 + hist.loc[worst, '북']))}. It lost less than the {bench_name} in {beat} of {len(hist)} crises.")
        + '</div>')
    hh = hist.sort_values("북")
    lab = [short(i) for i in hh.index]
    fig = go.Figure()
    fig.add_bar(y=lab, x=hh["북"], orientation="h", name=T("내 주식", "Your stocks"), marker_color=ui.BAD,
                hovertemplate="%{y}<br>%{x:+.1%}<extra></extra>")
    fig.add_scatter(y=lab, x=hh["벤치"], mode="markers", name=bench_name,
                    marker=dict(symbol="line-ns", size=16, line=dict(width=2.5, color=ui.INK)),
                    hovertemplate="%{y}<br>" + bench_name + " %{x:+.1%}<extra></extra>")
    span = float(np.abs(hh[["북", "벤치"]].to_numpy()).max()) * 1.08
    # 숫자는 막대 끝이 아니라 오른쪽 한 열에 — 시장 눈금과 겹치지 않게
    fig.add_scatter(y=lab, x=[span * 0.06] * len(lab), mode="text", showlegend=False, hoverinfo="skip",
                    text=[f"<b>{pct(b, True, 0)}</b>  <span style='color:{ui.INK2}'>{T('시장', 'market')} {pct(m, True, 0)}</span>"
                          for b, m in zip(hh["북"], hh["벤치"])],
                    textposition="middle right", textfont=dict(size=12, color=ui.INK))
    fig.update_xaxes(range=[-span, span * 0.75], tickvals=[v for v in (-0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0) if v > -span],
                     tickformat=".0%", zeroline=True, zerolinecolor=ui.INK, zerolinewidth=1)
    fig.update_yaxes(autorange="reversed", showgrid=False, type="category")
    st.plotly_chart(ui.layout(fig, 70 + 30 * len(lab)), width="stretch", config={"displayModeBar": False})
    proxied = float(hist["대리 비중"].max())
    html_(f'<div class="q-cap">{T("지금의 비중을 그 기간에 그대로 적용한 결과예요. 세로 선은 비교 시장.", "Today’s mix applied to each period. The vertical tick is the market you compare with.")}'
          + (T(" 당시 상장 전이던 종목은 시장 움직임으로 대신 계산했어요.", " Stocks not yet listed at the time are stood in for by the market.") if proxied > 0 else "")
          + '</div>')

html_(f'<div class="q-cap" style="margin-top:28px">{T(
    "주의: 과거 주가의 움직임으로 만든 추정이에요. 앞으로도 그렇게 움직인다는 보장은 없고, 투자 조언이 아니에요. "
    "금액 결과는 평균 수익을 0으로 두고(지난 상승이 이어진다고 가정하지 않음) 기간 동안 사고팔지 않는다고 가정했어요.",
    "These are estimates built from how prices moved in the past. There is no guarantee they will move the same way, "
    "and this is not investment advice. The value results assume zero average return (no assumption that recent gains "
    "continue) and no buying or selling during the period.")}</div>')

# ════════════════════════════════════════════════════════════════════════
# 전문가용
# ════════════════════════════════════════════════════════════════════════

html_('<div class="q-expert"></div>')
expert = st.toggle(T("자세히 보기 (전문가용)", "Show the detailed analysis (for practitioners)"), key="expert")
html_(f'<div class="q-cap" style="margin-top:2px">{T(
    "리스크 관리 실무 지표: VaR·ES 방법 비교, 모형 사후검증, 스트레스 테스트, 장중 모니터링, 유동성.",
    "Risk-desk measures: VaR and ES across methods, backtests, stress tests, intraday monitoring, liquidity.")}</div>')
if not expert:
    html_('<div class="q-foot">Risk Lab · ' + T("가격 자료: Yahoo Finance", "Prices: Yahoo Finance") +
          ' · <a href="https://github.com/Giana0412/risk-lab" style="color:inherit">GitHub</a></div>')
    st.stop()

with st.expander(T("계산 설정", "Calculation settings")):
    e1, e2 = st.columns(2)
    e1.select_slider(T("추정 창 (거래일)", "Estimation window (trading days)"), [125, 250, 500, 750],
                     value=lookback, key="lookback", help=T("리스크·VaR 탭 계산에 쓰는 과거 구간 길이", "History used in the Risk and VaR tabs"))
    e2.radio(T("리스크·VaR 탭 보유기간", "Horizon for the Risk and VaR tabs"), [1, 5, 10], index=[1, 5, 10].index(horizon),
             horizontal=True, key="horizon", format_func=lambda x: T(f"{x}일", f"{x}d"),
             help=T("h일 VaR = 1일 × √h (i.i.d. 가정)", "h-day VaR = 1-day × √h (i.i.d. assumption)"))
    st.caption(T(f"비교 지수 {bench} · {len(hold)}종목 · 현금 {1 - sum(weights.values()):.1%} · 추정 구간 {len(win)}거래일",
                 f"Benchmark {bench} · {len(hold)} holdings · cash {1 - sum(weights.values()):.1%} · {len(win)}-day window"))

tab_out, tab_sum, tab_var, tab_bt, tab_st, tab_attr, tab_live, tab_liq, tab_help = st.tabs([
    T("전망", "Outlook"), T("리스크", "Risk"), T("VaR", "VaR"), T("검증", "Backtest"),
    T("스트레스", "Stress"), T("기여도", "Attribution"), T("장중", "Intraday"), T("유동성", "Liquidity"),
    T("읽는 법", "Guide")])

# ════════════════════════════════════════════════════════════════════════
# 1. 전망 / Outlook
# ════════════════════════════════════════════════════════════════════════

with tab_out:
    ui.chapter(1)
    h2(T(f"앞으로 {H}거래일, 벤치를 이길 확률 {p_win:.0%}",
         f"{H} trading days to go: {p_win:.0%} probability of beating the benchmark"))
    lean_ko = ("벤치에 뒤처지는 쪽으로 기운다" if p_win < 0.45 else
               "어느 쪽으로도 기울지 않는다" if p_win < 0.55 else "벤치를 앞서는 쪽으로 기운다")
    lean_en = ("leans toward underperformance" if p_win < 0.45 else
               "is roughly balanced" if p_win < 0.55 else "leans toward outperformance")
    p(T(f"현재 북을 {mday(remaining[0])}부터 {mday(remaining[-1])}까지 그대로 보유한다고 가정하면, "
        f"{bench} 대비 상대수익이 플러스일 확률은 <b>{p_win:.0%}</b>로 추정되며 분포는 {lean_ko}. "
        f"하위 5% 시나리오는 <b>{pp(q05)}</b>, 하위 1% 시나리오는 <b>{pp(q01)}</b>이다. "
        f"투자 금액 기준으로 하위 5%는 약 {money(q05 * nav)}에 해당한다.",
        f"If the current book is held unchanged from {mday(remaining[0])} to {mday(remaining[-1])}, the "
        f"probability of a positive return relative to {bench} is estimated at <b>{p_win:.0%}</b>, and the "
        f"distribution {lean_en}. The 5th-percentile scenario is <b>{pp(q05)}</b> and the 1st-percentile "
        f"scenario is <b>{pp(q01)}</b>. On the amount invested, the 5th percentile is about "
        f"{money(q05 * nav)}."))
    if jumps_h:
        p(T(f"평가 기간 중 실적 발표가 예정된 종목은 {', '.join(jumps_h)}이며, 해당 종목의 과거 실적일 반응을 "
            "분포에 반영했다.",
            f"{', '.join(jumps_h)} report earnings within the window; their historical earnings-day reactions "
            "are included in the distribution."))
    else:
        p(T("평가 기간 중 실적 발표가 예정된 보유 종목은 없다.",
            "No holding is scheduled to report earnings within the window."))
    if jumps_1:
        o1 = risk.simulate_outcome(excess_ex, 1, jumps_1)
        p(T(f"<b>다음 거래일({mday(remaining[0])})에 {', '.join(jumps_1)}의 실적 반응이 예정되어 있다.</b> "
            f"이 경우 하루 하위 1% 상대손실은 {pp(-risk.var_fhs(excess_ex, 0.99))}에서 "
            f"<b>{pp(-o1.stats()['VaR 99%'])}</b>로 커진다.",
            f"<b>{', '.join(jumps_1)} react to earnings on the next trading day ({mday(remaining[0])}).</b> "
            f"That widens the one-day 1st-percentile relative loss from {pp(-risk.var_fhs(excess_ex, 0.99))} "
            f"to <b>{pp(-o1.stats()['VaR 99%'])}</b>."))
    for t, n in thin:
        p(T(f"{t}는 평가 기간 중 실적 발표가 있으나 과거 표본이 {n}건에 그쳐 분포에 반영하지 못했다. "
            "실제 위험은 위 수치보다 크다.",
            f"{t} reports within the window, but with only {n} past observations its earnings risk is not "
            "included. True risk is larger than shown."))
    if earn_failed:
        p(T("다음 종목은 실적 일정을 받지 못해 실적 위험이 빠져 있다: ",
            "Earnings calendar unavailable, so earnings risk is excluded for: ") +
          ", ".join(f"{t} ({ui.esc(e)})" for t, e in earn_failed.items()) + ".")

    keys([(T("벤치를 이길 확률", "P(beat benchmark)"), f"{p_win:.0%}", False),
          (T("중앙값", "Median"), pp(s_tot["중앙"]), s_tot["중앙"] < 0),
          (T("하위 5%", "5th percentile"), pp(q05), True),
          (T("하위 1%", "1st percentile"), pp(q01), True)])

    figure(T(f"남은 {H}거래일 상대수익 분포", f"Distribution of relative return over the next {H} trading days"),
           T(f"북 − {bench}, 시뮬레이션 경로 수", f"Book − {bench}, number of simulated paths"))
    cnt, edges = np.histogram(out.total, bins=70)
    mids = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(
        x=mids, y=cnt, marker_color=[ui.BAD if m < q05 else ui.BOOK for m in mids], marker_line_width=0,
        width=(edges[1] - edges[0]) * 0.9, customdata=np.stack([edges[:-1], edges[1:]], axis=1),
        hovertemplate="%{customdata[0]:+.1%} ~ %{customdata[1]:+.1%}: %{y:,}<extra></extra>"))
    fig.add_vline(x=0, line_color=ui.INK, line_width=1)
    fig.add_annotation(x=q05, y=1, yref="paper", text=f"{T('하위 5%', '5th pct')} {pp(q05)}", showarrow=False,
                       xanchor="right", yanchor="top", font=dict(size=11, color=ui.BAD))
    lo, hi = np.quantile(out.total, [0.002, 0.998])
    fig.update_xaxes(range=[lo, hi])
    fig.update_yaxes(showticklabels=False, showgrid=False)
    st.plotly_chart(ui.layout(fig, 260, "x", legend=False), width="stretch")
    note(T("평소 거래일은 실적일을 제외한 초과수익에 GARCH(1,1)를 적합하고 표준화 잔차를 복원추출(FHS)해 "
           "2만 개 경로를 생성했다. 실적 반응은 종목별 과거 실적일 초과수익에서 추출했다. "
           "손절과 리밸런싱은 반영하지 않았다. 붉은 막대는 하위 5% 구간.",
           "Ordinary days: GARCH(1,1) fitted to excess returns with earnings days removed, then 20,000 paths "
           "generated by resampling standardized residuals (filtered historical simulation). Earnings "
           "reactions are drawn from each holding's past earnings-day excess returns. Stop-losses and "
           "rebalancing are not modelled. Red bars mark the bottom 5%."))

    h2(T("모형 추정치와 과거 실현치의 차이", "Model estimate versus realized history"))
    vols = {T("4년", "4y"): float(ex_long.std()), T("1년", "1y"): float(ex_long.iloc[-250:].std()),
            T("3개월", "3m"): float(ex_long.iloc[-60:].std()), T("1개월", "1m"): float(ex_long.iloc[-20:].std())}
    v4, v1m = list(vols.values())[0], list(vols.values())[3]
    hot = g.sigma_next > 1.2 * v4
    if s_real:
        wider = s_real["5%"] > q05
        p(T(f"같은 북을 지난 {len(ex_long)}거래일 동안 보유했을 때 실제 {H}일 상대수익의 하위 5%는 "
            f"<b>{pp(s_real['5%'])}</b>로, 모형 추정치({pp(q05)})보다 {'좁다' if wider else '넓다'}. "
            + (f"일간 상대 변동성이 4년 평균 {v4:.2%}에서 최근 1개월 {v1m:.2%}로 높아진 상태이며, 모형은 이 "
               f"수준이 점차 낮아진다고 가정한다(다음 거래일 추정 {g.sigma_next:.2%}). 변동성이 유지되면 모형 "
               "쪽이, 평균으로 돌아가면 과거 실현치 쪽이 현실에 가깝다."
               if hot else "현재 변동성은 평소 수준이어서 두 추정치의 차이는 크지 않다."),
            f"Had the same book been held over the past {len(ex_long)} trading days, the realized 5th percentile "
            f"of {H}-day relative return would have been <b>{pp(s_real['5%'])}</b>, "
            f"{'narrower' if wider else 'wider'} than the model's {pp(q05)}. "
            + (f"Daily relative volatility has risen from a four-year average of {v4:.2%} to {v1m:.2%} over the "
               f"past month; the model assumes it decays gradually (next-day estimate {g.sigma_next:.2%}). If "
               "volatility persists the model is closer to reality; if it mean-reverts, history is."
               if hot else "Volatility is near its usual level, so the two estimates are close.")))
    rows = [(T("벤치를 이길 확률", "P(beat benchmark)"), "P(상대 > +0%)", "p"),
            (T("+2%p 이상 앞설 확률", "P(lead by more than 2pp)"), "P(상대 > +2%)", "p"),
            (T("2%p 넘게 뒤처질 확률", "P(trail by more than 2pp)"), "P(상대 < −2%)", "p"),
            (T("중앙값", "Median"), "중앙", "v"), (T("하위 5%", "5th percentile"), "5%", "v"),
            (T("하위 1%", "1st percentile"), "VaR 99%", "neg")]

    def cell(sd, k, kind):
        return f"{sd[k]:.0%}" if kind == "p" else pp(-sd[k] if kind == "neg" else sd[k])
    comp = {T("모형", "Model"): [cell(s_tot, k, kd) for _, k, kd in rows]}
    if s_real:
        comp[T(f"과거 실현 ({len(ex_long)}일)", f"Realized ({len(ex_long)}d)")] = [cell(s_real, k, kd) for _, k, kd in rows]
    if jumps_h:
        comp[T("모형 (실적 제외)", "Model (ex-earnings)")] = [cell(s_base, k, kd) for _, k, kd in rows]
    table_title(T(f"{H}거래일 상대수익 분포 비교", f"{H}-day relative return: model versus history"))
    table(pd.DataFrame(comp, index=[r[0] for r in rows]))
    extra = ""
    if not math.isnan(vr) and (vr < 0.85 or vr > 1.15):
        extra = T(f" 분산비 {vr:.2f}: 초과수익이 " + ("되돌림을 보여 모형이 꼬리를 과대평가할 수 있다." if vr < 0.85
                                                else "추세를 보여 모형이 꼬리를 과소평가할 수 있다."),
                  f" Variance ratio {vr:.2f}: excess returns " +
                  ("mean-revert, so the model may overstate the tails." if vr < 0.85
                   else "trend, so the model may understate the tails."))
    note(T("과거 실현은 겹치는 구간으로 계산했으며 실적일을 포함한다.",
           "Realized figures use overlapping windows and include earnings days.") + extra)

    h2(T("보유 종목 실적 일정", "Earnings calendar for holdings"))
    jr = pd.DataFrame(jump_rows).set_index("_t").sort_values(c_in, ascending=False)
    table_title(T("실적 발표 일정과 과거 실적일 반응", "Earnings dates and historical earnings-day reactions"),
                T("반응 = 반응일 종목 수익률 − β × 벤치 수익률", "Reaction = stock return − β × benchmark return"))
    table(jr, fmt={c_w: lambda v: f"{v:.1%}", c_avg: lambda v: pct(v, digits=1),
                   c_lo: lambda v: pct(v, True, 1), c_hi: lambda v: pct(v, True, 1), c_n: lambda v: f"{int(v)}"},
          neg_cols=(c_lo,), index_name=T("종목", "Ticker"))
    note(T("발표 시각이 개장(09:30 ET) 전이면 당일, 이후면 다음 거래일을 반응일로 본다. "
           "시각이 00:00으로 기록된 경우 두 날 중 반응이 큰 날을 택했다. 다음 발표 시각은 미 동부 기준.",
           "Reports before the 09:30 ET open react the same day; later reports react the next trading day. "
           "When the time is recorded as 00:00, the day with the larger reaction is used. Times are US Eastern."))

# ════════════════════════════════════════════════════════════════════════
# 2. 리스크 개요 / Risk summary
# ════════════════════════════════════════════════════════════════════════

with tab_sum:
    ui.chapter(2)
    a = 0.99
    s = risk.summary_stats(port, bret)
    conc = risk.concentration(weights)
    v_abs, v_rel = risk.var_historical(port, a) * h, risk.var_historical(excess, a) * h
    es_rel = risk.es_historical(excess, 0.975) * h
    cv = risk.component_var(R, weights, a)
    top = cv.index[0]

    h2(T("손실 규모와 위험 집중", "Loss size and risk concentration"))
    p(T(f"과거 {len(win)}거래일의 분포를 그대로 적용하면, {horizon}일 기준 99% 신뢰수준에서 북의 손실은 "
        f"<b>{pct(v_abs, digits=1)}</b>({money(v_abs * nav)}), 벤치 대비 손실은 <b>{pp(-v_rel)}</b>를 넘지 않는다. "
        f"그 너머 하위 2.5% 구간의 평균 상대손실(ES)은 {pp(-es_rel)}이다. "
        f"위험 기여가 가장 큰 종목은 {top}로, 비중 {cv.loc[top, '비중']:.0%}에 비해 "
        f"전체 위험의 {cv.loc[top, '기여 비율']:.0%}를 차지한다.",
        f"Applying the distribution of the past {len(win)} trading days, the {horizon}-day 99% loss on the book "
        f"does not exceed <b>{pct(v_abs, digits=1)}</b> ({money(v_abs * nav)}), and the loss relative to the "
        f"benchmark does not exceed <b>{pp(-v_rel)}</b>. Beyond that, the average relative loss in the worst 2.5% "
        f"(expected shortfall) is {pp(-es_rel)}. The largest risk contributor is {top}, which accounts for "
        f"{cv.loc[top, '기여 비율']:.0%} of total risk on a {cv.loc[top, '비중']:.0%} weight."))
    keys([(T("VaR 99% · 북", "VaR 99% · book"), pct(v_abs, digits=1), False),
          (T("VaR 99% · 벤치 대비", "VaR 99% · relative"), pp(-v_rel), True),
          (T("ES 97.5% · 벤치 대비", "ES 97.5% · relative"), pp(-es_rel), True),
          (T("최대낙폭", "Max drawdown"), pct(s["최대낙폭"], digits=1), True)])
    note(T(f"역사적 시뮬레이션, {horizon}일. 최대낙폭은 북 자체 기준.",
           f"Historical simulation, {horizon}-day. Max drawdown is for the book itself."))

    table_title(T("벤치 대비 특성", "Characteristics versus the benchmark"))
    te = s["추적오차(연)"]
    c_val, c_mean = T("값", "Value"), T("해석", "Interpretation")
    feat = pd.DataFrame({
        c_val: [num(s["베타(벤치)"]), pct(te, digits=1), num(s["정보비율"]), num(conc["유효 종목 수"], 1),
                pct(s["연 변동성"], digits=1), num(s["소르티노(rf=0)"]), num(s["칼마"]), num(s["왜도"]),
                num(s["초과첨도"])],
        c_mean: [T("벤치가 1% 움직일 때 북의 평균 반응", "Average book move for a 1% benchmark move"),
                 T(f"일간 약 ±{te / math.sqrt(252):.2%} 수준의 이탈",
                   f"Roughly ±{te / math.sqrt(252):.2%} daily deviation"),
                 T("초과수익 ÷ 추적오차 (표본내, 상향 편향 가능)",
                   "Excess return ÷ tracking error (in-sample, may be biased upward)"),
                 T(f"{len(hold)}종목을 비중으로 환산한 분산 정도 (1/HHI)",
                   f"Diversification of {len(hold)} holdings after weighting (1/HHI)"),
                 T("북 자체", "Book itself"), T("하방 변동성 대비 수익", "Return per unit of downside volatility"),
                 T("연수익 ÷ 최대낙폭", "Annual return ÷ max drawdown"),
                 T("음수면 큰 손실이 큰 이익보다 잦다", "Negative: large losses more frequent than large gains"),
                 T("0이면 정규분포, 클수록 두꺼운 꼬리", "0 for a normal distribution; higher means fatter tails")]},
        index=[T("베타", "Beta"), T("추적오차 (연)", "Tracking error (ann.)"), T("정보비율", "Information ratio"),
               T("유효 종목 수", "Effective N"), T("연 변동성", "Volatility (ann.)"), T("소르티노", "Sortino"),
               T("칼마", "Calmar"), T("왜도", "Skewness"), T("초과첨도", "Excess kurtosis")])
    table(feat, index_name=T("지표", "Metric"))
    note(T("무위험금리는 0으로 두었다.", "Risk-free rate set to zero."))

    table_title(T("종목별 위험 기여", "Risk contribution by holding"),
                T("정규 VaR 99%의 오일러 분해. 배수 = 위험 기여 ÷ 비중",
                  "Euler decomposition of normal VaR 99%. Multiple = risk share ÷ weight"))
    c_w, c_rc, c_mul, c_vol = T("비중", "Weight"), T("위험 기여", "Risk share"), T("배수", "Multiple"), T("연 변동성", "Vol (ann.)")
    cv_show = pd.DataFrame({c_w: cv["비중"], c_rc: cv["기여 비율"], c_mul: cv["기여 비율"] / cv["비중"],
                            c_vol: cv["개별 변동성(연)"]})
    table(cv_show, fmt={c_w: lambda v: f"{v:.1%}", c_rc: lambda v: f"{v:.1%}", c_mul: lambda v: f"{v:.2f}",
                        c_vol: lambda v: f"{v:.0%}"}, index_name=T("종목", "Ticker"), em_rows=(top,))
    note(T("배수가 1보다 크면 비중에 비해 위험을 많이 낸다. 변동성이 높거나 다른 종목과 함께 움직이는 경우다.",
           "A multiple above 1 means the holding contributes more risk than its weight, because it is more "
           "volatile or moves with the rest of the book."))

# ════════════════════════════════════════════════════════════════════════
# 3. VaR 방법론 / VaR methods
# ════════════════════════════════════════════════════════════════════════

with tab_var:
    ui.chapter(3)
    L_BOOK, L_REL = T("북", "Book"), T("벤치 대비", "Relative")
    rows_v = {}
    for label, series in [(L_BOOK, port), (L_REL, excess)]:
        t = risk.var_table(series, horizon=horizon)
        gg = risk.garch_fit(series)
        if horizon == 1:
            t.loc["FHS-GARCH"] = [risk.var_fhs(series, 0.95, gg), risk.var_fhs(series, 0.99, gg)]
        else:
            sim = risk.simulate_fhs(gg, horizon)
            t.loc["FHS-GARCH"] = [-np.quantile(sim, 0.05), -np.quantile(sim, 0.01)]
        if series is port:
            for dof, nm in [(None, "몬테카를로 정규"), (5.0, "몬테카를로 t")]:
                t.loc[nm] = [risk.var_monte_carlo(R, weights, al, dof=dof)[0] * h for al in (0.95, 0.99)]
        rows_v[label] = t
    var_methods = [m for m in rows_v[L_BOOK].index if not m.startswith("ES")]
    rel99 = rows_v[L_REL]["VaR 99%"].reindex(var_methods).dropna()
    lo_m, hi_m = rel99.idxmin(), rel99.idxmax()
    ratio = rel99.max() / rel99.min()

    h2(T(f"방법에 따라 VaR는 최대 {ratio:.1f}배 달라진다", f"VaR varies by up to {ratio:.1f}x across methods"))
    p(T(f"동일한 북과 데이터에 대해 벤치 대비 {horizon}일 99% VaR는 {METHOD_NAMES[lo_m]}에서 "
        f"<b>{pp(-rel99.min())}</b>, {METHOD_NAMES[hi_m]}에서 <b>{pp(-rel99.max())}</b>로 추정된다. 최근 데이터에 "
        "가중을 두는 방법(EWMA, FHS-GARCH)일수록 값이 크다면 최근 변동성이 표본 평균보다 높다는 뜻이다. "
        "방법 간 적합성은 검증 탭에서 확인한다.",
        f"For the same book and data, the {horizon}-day 99% relative VaR ranges from <b>{pp(-rel99.min())}</b> "
        f"({METHOD_NAMES[lo_m]}) to <b>{pp(-rel99.max())}</b> ({METHOD_NAMES[hi_m]}). When methods that weight "
        "recent data (EWMA, FHS-GARCH) give larger values, recent volatility is above the sample average. "
        "The Backtest tab tests which method fits."))

    figure(T(f"방법별 {horizon}일 99% VaR", f"{horizon}-day 99% VaR by method"),
           T("오른쪽일수록 손실을 크게 추정", "Further right = larger estimated loss"))
    fig = go.Figure()
    for label, color, sym in [(L_BOOK, ui.MUTED, "circle-open"), (L_REL, ui.BOOK, "circle")]:
        v = rows_v[label]["VaR 99%"].reindex(var_methods)
        fig.add_scatter(x=v.values, y=[METHOD_NAMES[m] for m in v.index], mode="markers", name=label,
                        marker=dict(size=10, color=color, symbol=sym, line=dict(color=color, width=1.5)),
                        hovertemplate="%{y}<br>" + label + " %{x:.2%}<extra></extra>")
    fig.update_yaxes(autorange="reversed", showgrid=True)
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(ui.layout(fig, 300), width="stretch")
    note(T("몬테카를로는 개별 종목 공분산으로 계산하므로 북에만 해당한다.",
           "Monte Carlo uses the holdings' covariance matrix, so it applies to the book only."))

    table_title(T(f"방법별 VaR와 ES ({horizon}일)", f"VaR and ES by method ({horizon}-day)"))
    both = pd.concat({k: v.rename(index=METHOD_NAMES) for k, v in rows_v.items()}, axis=1)
    both.columns = [f"{a_} {b_.replace('VaR ', '')}" for a_, b_ in both.columns]
    table(both, fmt={c: (lambda v: pct(v)) for c in both.columns}, index_name=T("방법", "Method"))
    note(T("손실을 양수로 표기했다.", "Losses are shown as positive numbers."))

    figure(T("북 변동성: EWMA와 60일 이동평균", "Book volatility: EWMA versus 60-day rolling"),
           T("연율화", "Annualized"))
    long = risk.portfolio_returns(rets_all[hold].dropna().iloc[-max(lookback, 500):], weights)
    ew = risk.ewma_vol(long) * math.sqrt(252)
    roll = long.rolling(60).std() * math.sqrt(252)
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name=T("60일", "60-day"), line=dict(color=ui.MUTED, width=1.5))
    fig.add_scatter(x=ew.index, y=ew, name="EWMA", line=dict(color=ui.BOOK, width=1.8))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(ui.layout(fig, 260), width="stretch")
    note(T("EWMA가 60일선 위에 있으면 최근 변동성이 구간 평균보다 높다.",
           "EWMA above the 60-day line means recent volatility is above the window average."))

# ════════════════════════════════════════════════════════════════════════
# 4. 모형 검증 / Backtesting
# ════════════════════════════════════════════════════════════════════════

with tab_bt:
    ui.chapter(4)
    L_REL, L_BOOK = T("벤치 대비", "Relative"), T("북", "Book")
    c1, c2, c3 = st.columns(3)
    bt_alpha = c1.radio(T("신뢰수준", "Confidence"), [0.95, 0.99], index=1, horizontal=True,
                        format_func=lambda x: f"{x:.0%}")
    bt_target = c2.radio(T("대상", "Series"), [L_REL, L_BOOK], horizontal=True)
    bt_window = c3.select_slider(T("추정 창 (일)", "Window (days)"), [125, 250, 500], value=250)
    long_R = rets_all[cols].dropna()
    if len(long_R) < bt_window + 60:
        p(T(f"공통 표본이 {len(long_R)}일로 검증에 필요한 길이(창 + 60일)에 못 미친다.",
            f"Only {len(long_R)} common days; backtesting needs the window plus 60 days."))
    else:
        lp = risk.portfolio_returns(long_R[hold], weights)
        ser = (lp - long_R[bench]) if bt_target == L_REL else lp
        ser = ser.iloc[-(bt_window + 750):]
        bt = cached_backtest(ser, bt_alpha, bt_window)
        sm = bt.summary.copy()
        ok = (sm["Kupiec p"] >= 0.05) & (sm["독립성 p"] >= 0.05)
        passed = [METHOD_NAMES.get(m, m) for m in sm.index[ok]]
        best = (sm["위반"] - sm["기대"]).abs().idxmin()
        n_days = int(sm["표본일"].iloc[0])

        h2(T("사후검증: " + (f"{len(passed)}개 방법이 두 검정을 통과" if passed else "두 검정을 모두 통과한 방법 없음"),
             "Backtest: " + (f"{len(passed)} methods pass both tests" if passed else "no method passes both tests")))
        p(T(f"매 거래일 직전 {bt_window}일만으로 다음 날 {bt_alpha:.0%} VaR를 추정하고 실제 손익과 비교했다"
            f"(검증 기간 {n_days}일). 초과 횟수의 적정성(Kupiec)과 초과의 독립성(Christoffersen)을 5% 유의수준에서 "
            "검정했다. "
            + (f"두 검정을 모두 통과한 방법은 <b>{', '.join(passed)}</b>이다. " if passed else
               "<b>두 검정을 모두 통과한 방법은 없다.</b> ")
            + f"초과 횟수가 기대치에 가장 가까운 방법은 {METHOD_NAMES.get(best, best)}이다.",
            f"Each day, the next day's {bt_alpha:.0%} VaR is estimated from the preceding {bt_window} days only "
            f"and compared with the realized P&L over {n_days} days. The exception rate (Kupiec) and the "
            "independence of exceptions (Christoffersen) are tested at the 5% level. "
            + (f"Methods passing both: <b>{', '.join(passed)}</b>. " if passed else
               "<b>No method passes both tests.</b> ")
            + f"The method closest to the expected exception count is {METHOD_NAMES.get(best, best)}."))

        def basel(x: str) -> str:
            if "99%" in x:
                return T("해당 없음", "n/a")
            z = x.split(" ")[-1]
            return BASEL_EN.get(z, z) if ui.LANG == "en" else z
        c_ex, c_exp, c_ind, c_dec, c_bz = (T("초과", "Exceptions"), T("기대", "Expected"), T("독립성 p", "Indep. p"),
                                           T("판정", "Verdict"), T("바젤 구간", "Basel zone"))
        view = pd.DataFrame({  # .values — 행 이름을 바꾸므로 원래 인덱스로 정렬되면 안 된다
            c_ex: sm["위반"].astype(int).values, c_exp: sm["기대"].values, "Kupiec p": sm["Kupiec p"].values,
            c_ind: sm["독립성 p"].values,
            c_dec: [T("적합", "Pass") if v else T("부적합", "Fail") for v in ok],
            c_bz: [basel(x) for x in sm["바젤(250일 환산)"]],
        }, index=[METHOD_NAMES.get(m, m) for m in sm.index])
        table_title(T(f"VaR 사후검증 ({bt_target}, {bt_alpha:.0%})", f"VaR backtest ({bt_target}, {bt_alpha:.0%})"))
        table(view, fmt={c_ex: lambda v: f"{int(v)}", c_exp: lambda v: f"{v:.1f}",
                         "Kupiec p": lambda v: f"{v:.3f}", c_ind: lambda v: f"{v:.3f}"},
              index_name=T("방법", "Method"), em_rows=tuple(passed))
        note(T("판정은 두 p값이 모두 0.05 이상일 때 적합. 바젤 구간은 250일 환산 초과 횟수 기준(99%에만 적용). "
               "현재 북을 과거 전 기간 보유했다고 가정한 재현이다.",
               "Pass when both p-values are at least 0.05. Basel zones use exceptions per 250 days and apply to "
               "99% only. This replays the current book as if it had been held throughout."))

        m = st.selectbox(T("그림에 표시할 방법", "Method to plot"), list(bt.forecasts.columns),
                         index=list(bt.forecasts.columns).index(best), format_func=lambda k: METHOD_NAMES.get(k, k))
        hits = bt.realized < -bt.forecasts[m]
        figure(T(f"일간 수익률과 VaR 한도: {METHOD_NAMES.get(m, m)}", f"Daily returns and VaR limit: {METHOD_NAMES.get(m, m)}"),
               T(f"초과 {int(hits.sum())}회", f"{int(hits.sum())} exceptions"))
        fig = go.Figure()
        fig.add_scatter(x=bt.realized.index, y=bt.realized, mode="lines", name=T("일간 수익률", "Daily return"),
                        line=dict(color="#b9b7b0", width=0.8))
        fig.add_scatter(x=bt.forecasts.index, y=-bt.forecasts[m], mode="lines", name=T("VaR 한도", "VaR limit"),
                        line=dict(color=ui.BOOK, width=1.6))
        fig.add_scatter(x=bt.realized.index[hits], y=bt.realized[hits], mode="markers", name=T("초과", "Exception"),
                        marker=dict(color=ui.BAD, size=6))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(ui.layout(fig, 300, "y"), width="stretch")
        note(T("VaR 한도는 손실 방향(음수)으로 표시했다.", "The VaR limit is plotted on the loss side (negative)."))

# ════════════════════════════════════════════════════════════════════════
# 5. 스트레스 테스트 / Stress tests
# ════════════════════════════════════════════════════════════════════════

with tab_st:
    ui.chapter(5)
    cpi = risk.stress_days(close, weights, bench, risk.CPI_DAYS, betas_bench)
    hist = risk.stress_replay(close, weights, bench, risk.HISTORICAL_SCENARIOS, betas_bench)

    def rel_bars(df: pd.DataFrame, labels: list[str], height: int):
        d = df["상대(북−벤치)"]
        fig = go.Figure(go.Bar(
            y=labels, x=d.values, orientation="h",
            marker_color=[ui.NEG_BAR if v < 0 else ui.POS_BAR for v in d.values],
            text=[pp(v) for v in d.values], textposition="outside", cliponaxis=False, textfont=dict(size=11),
            customdata=np.stack([df["북"], df["벤치"]], axis=1),
            hovertemplate="%{y}<br>" + T("북", "Book") + " %{customdata[0]:+.1%} · " + T("벤치", "Bench") +
                          " %{customdata[1]:+.1%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
        span = float(np.abs(d).max()) * 1.4
        fig.update_xaxes(range=[-span, span], tickformat=".0%", zeroline=True, zerolinecolor=ui.INK, zerolinewidth=1)
        return ui.layout(fig, height, legend=False)

    def stress_tbl(df: pd.DataFrame, labels: list[str]):
        cb, cbe, crel, cw, cpx = (T("북", "Book"), T("벤치", "Bench"), T("벤치 대비", "Relative"),
                                  T("최대 손실 종목", "Largest detractor"), T("대리 비중", "Proxied"))
        t = pd.DataFrame({cb: df["북"].values, cbe: df["벤치"].values, crel: df["상대(북−벤치)"].values,
                          cw: df["최악 기여"].values, cpx: df["대리 비중"].values}, index=labels)
        table(t, fmt={cb: lambda v: pct(v, True, 1), cbe: lambda v: pct(v, True, 1), crel: pp,
                      cpx: lambda v: f"{v:.0%}" if v else "–"},
              neg_cols=(crel,), index_name=T("시나리오", "Scenario"))

    if not cpi.empty and not hist.empty:
        wc, wh = cpi["상대(북−벤치)"].idxmin(), hist["상대(북−벤치)"].idxmin()
        beat = int((hist["상대(북−벤치)"] > 0).sum())
        h2(T(f"과거 위기 {len(hist)}건 중 {beat}건에서 벤치보다 선방",
             f"Outperformed the benchmark in {beat} of {len(hist)} historical crises"))
        p(T(f"현재 비중을 과거 위기 구간에 적용하면 {len(hist)}건 중 {beat}건에서 북의 하락폭이 {bench}보다 작았다. "
            f"가장 불리한 구간은 {short(wh)}로 벤치 대비 <b>{pp(hist.loc[wh, '상대(북−벤치)'])}</b>이다. "
            f"과거 CPI 발표일 하루 기준으로는 {wc.split(' ')[0]}가 가장 불리했다({pp(cpi.loc[wc, '상대(북−벤치)'])}).",
            f"Applying current weights to past crisis windows, the book fell less than {bench} in {beat} of "
            f"{len(hist)} cases. The worst relative outcome was the {short(wh)} at "
            f"<b>{pp(hist.loc[wh, '상대(북−벤치)'])}</b>. Among single CPI release days, {wc.split(' ')[0]} was "
            f"the worst ({pp(cpi.loc[wc, '상대(북−벤치)'])})."))

    if not cpi.empty:
        lab = [f"{scen(k).split(' ')[0]}  {scen(k).split('(')[1].split('·')[-1].strip(' )')}" for k in cpi.index]
        figure(T("CPI 발표일의 벤치 대비 수익", "Relative return on CPI release days"),
               T("현재 비중을 과거 발표일 하루에 적용", "Current weights applied to each past release day"))
        st.plotly_chart(rel_bars(cpi, lab, 230), width="stretch")
        table_title(T("CPI 발표일 재현", "CPI release-day replay"))
        stress_tbl(cpi, [k.split(" ")[0] for k in cpi.index])
        note(T("CPI 발표를 앞두고 시나리오를 점검하는 용도다.", "For checking scenarios ahead of a CPI release."))

    if not hist.empty:
        figure(T("과거 위기 구간의 벤치 대비 수익", "Relative return over historical crisis windows"),
               T("구간 누적", "Cumulative over each window"))
        st.plotly_chart(rel_bars(hist, [short(i) for i in hist.index], 380), width="stretch")
        table_title(T("과거 위기 재현", "Historical crisis replay"))
        stress_tbl(hist, [short(i) for i in hist.index])
        proxied = [short(i) for i in hist.index[hist["대리 비중"] > 0.2]]
        note(T("대리 비중은 당시 상장 전이어서 벤치 베타로 대신한 비중이다."
               + (f" 대리 비중이 20%를 넘는 구간({', '.join(proxied)})은 해석에 유의한다." if proxied else ""),
               "Proxied weight belongs to holdings not yet listed at the time, replaced by beta × benchmark."
               + (f" Interpret windows with more than 20% proxied ({', '.join(proxied)}) with care." if proxied else "")))

    h2(T("가상 충격", "Hypothetical shocks"))
    p(T("최근 2년 일간 수익률을 시장, 소형주 스프레드, 장기채, 유가, 달러의 다섯 요인에 회귀한 베타로 "
        "충격의 영향을 선형 근사한다. 고유위험은 0으로 둔다.",
        "Each holding's daily returns over the past two years are regressed on five factors (market, small-cap "
        "spread, long bonds, oil, dollar); shocks are applied linearly through those betas. Idiosyncratic risk "
        "is set to zero."))
    F = data.factor_returns(close).iloc[-500:]
    fb = risk.factor_betas(rets_all[cols].iloc[-500:], F)
    presets = {
        T("CPI 상회 (금리 상승, 성장주 약세)", "Hot CPI (rates up, growth down)"):
            {"시장(ACWI)": -0.025, "소형주 스프레드(IWM−SPY)": -0.01, "장기채(TLT)": -0.025, "유가(USO)": 0.0,
             "달러(UUP)": 0.008},
        T("CPI 하회 (소형주 강세)", "Cool CPI (small caps rally)"):
            {"시장(ACWI)": 0.015, "소형주 스프레드(IWM−SPY)": 0.025, "장기채(TLT)": 0.02, "유가(USO)": 0.0,
             "달러(UUP)": -0.006},
        T("경기침체 우려", "Recession scare"):
            {"시장(ACWI)": -0.08, "소형주 스프레드(IWM−SPY)": -0.03, "장기채(TLT)": 0.05, "유가(USO)": -0.10,
             "달러(UUP)": 0.02},
        T("유가 급등", "Oil spike"):
            {"시장(ACWI)": -0.02, "소형주 스프레드(IWM−SPY)": 0.0, "장기채(TLT)": -0.01, "유가(USO)": 0.15,
             "달러(UUP)": 0.005},
        T("직접 입력", "Custom"): None,
    }
    pre = st.selectbox(T("시나리오", "Scenario"), list(presets))
    shocks = {}
    shock_cols = st.columns(len(data.FACTOR_LABELS))
    for col, lab_ in zip(shock_cols, data.FACTOR_LABELS):
        default = presets[pre][lab_] if presets[pre] else 0.0
        shocks[lab_] = col.number_input(f"{factor(lab_)} (%)", value=float(default) * 100, step=0.5, format="%.1f",
                                        key=f"shock-{pre}-{lab_}") / 100
    per, book_move, bench_move = risk.factor_shock(fb, weights, bench, shocks)
    keys([(T("북", "Book"), pct(book_move, True, 1), book_move < 0),
          (bench, pct(bench_move, True, 1), bench_move < 0),
          (T("벤치 대비", "Relative"), pp(book_move - bench_move), book_move - bench_move < 0)])
    r2 = fb.drop(index=bench, errors="ignore")["R2"]
    note(T("프리셋 값은 예시 가정이다. 종목별 설명력(R²) 중앙값 ", "Preset values are illustrative assumptions. Median R² ")
         + f"{r2.median():.0%}"
         + (T(" — 설명력이 낮아 방향 참고용으로만 본다.", " — low explanatory power; read for direction only.")
            if r2.median() < 0.3 else "."))
    table_title(T("요인 베타", "Factor betas"))
    fb_show = fb.rename(columns={k: factor(k) for k in fb.columns})
    table(fb_show, fmt={c: (lambda v: num(v)) for c in fb_show.columns}, index_name=T("종목", "Ticker"))
    note(T("최근 500거래일 일간 수익률 OLS.", "OLS on daily returns over the last 500 trading days."))

# ════════════════════════════════════════════════════════════════════════
# 6. 장중 / Intraday
# ════════════════════════════════════════════════════════════════════════

with tab_live:
    ui.chapter(6)
    refresh = st.toggle(T("60초마다 갱신", "Refresh every 60 seconds"), value=False,
                        help=T("yfinance 분봉은 약 15분 지연될 수 있다. 미 동부 09:30–16:00에만 의미가 있다.",
                               "yfinance bars may be delayed about 15 minutes. Meaningful only 09:30–16:00 ET."))

    @st.fragment(run_every=timedelta(seconds=60) if refresh else None)
    def live():
        ui.chapter(6)  # 조각만 다시 돌 때도 6장 번호로
        tk = tuple(sorted(set(hold) | {bench}))
        ip = get_intraday(tk)
        if ip.empty:
            p(T("분봉 데이터가 없다. 장 마감 직후나 휴장일에는 정상이다.",
                "No intraday bars. This is normal right after the close or on holidays."))
            return
        session = ip.index[-1].date()
        today = ip[ip.index.date == session].ffill()
        prev = close[list(tk)].loc[:pd.Timestamp(session) - pd.Timedelta(days=1)].ffill().iloc[-1]
        bad = [t for t in tk if pd.isna(prev.get(t)) or t not in today or today[t].dropna().empty]
        if bad:
            p(T(f"전일 종가나 당일 분봉이 없는 종목이 있어 계산하지 않는다: {', '.join(bad)}.",
                f"Skipped: missing previous close or intraday bars for {', '.join(bad)}."))
            return
        today = today.bfill()
        rel = today / prev - 1
        w = pd.Series(weights)
        book_path = (rel[hold] * w).sum(axis=1, min_count=len(hold))
        rel_path = book_path - rel[bench]
        now_book, now_bench = float(book_path.iloc[-1]), float(rel[bench].iloc[-1])
        now_rel = now_book - now_bench
        v95, v99 = risk.var_historical(excess, 0.95), risk.var_historical(excess, 0.99)
        level = (T("99% VaR를 넘었다", "beyond the 99% VaR") if now_rel < -v99 else
                 T("95% VaR를 넘었다", "beyond the 95% VaR") if now_rel < -v95 else
                 T("95% VaR 이내다", "within the 95% VaR"))
        h2(T(f"{mday(session)} {ip.index[-1]:%H:%M} ET 기준 벤치 대비 {pp(now_rel)}",
             f"{pp(now_rel)} versus the benchmark as of {mday(session)} {ip.index[-1]:%H:%M} ET"))
        p(T(f"북은 전일 종가 대비 {pct(now_book, True)}, {bench}는 {pct(now_bench, True)}로 "
            f"벤치 대비 {pp(now_rel)}({money(now_rel * nav)})이다. 일간 상대 손실 기준으로 {level}"
            f"(95% {pp(-v95)}, 99% {pp(-v99)}).",
            f"The book is {pct(now_book, True)} from the previous close and {bench} is {pct(now_bench, True)}, "
            f"so the book is {pp(now_rel)} ({money(now_rel * nav)}) relative, which is {level} for daily "
            f"relative loss (95% {pp(-v95)}, 99% {pp(-v99)})."))
        figure(T("장중 벤치 대비 수익", "Intraday relative return"),
               T("전일 종가 대비, 5분봉", "From previous close, 5-minute bars"))
        fig = go.Figure()
        fig.add_hline(y=0, line_color=ui.INK, line_width=1)
        for v, lab_, colr in [(v95, "VaR 95%", ui.MUTED), (v99, "VaR 99%", ui.BAD)]:
            fig.add_hline(y=-v, line_dash="dot", line_color=colr, line_width=1, annotation_text=lab_,
                          annotation_position="bottom right", annotation_font=dict(color=colr, size=11))
        fig.add_scatter(x=rel_path.index, y=rel_path, line=dict(color=ui.BOOK, width=1.8),
                        hovertemplate="%{x|%H:%M} %{y:+.2%}<extra></extra>")
        fig.update_yaxes(range=[min(float(rel_path.min()), -v99) * 1.2, max(float(rel_path.max()), v95) * 1.2])
        st.plotly_chart(ui.layout(fig, 280, "y", legend=False), width="stretch")
        note(T("yfinance 분봉 기준이며 지연될 수 있다.", "Based on yfinance bars, which may be delayed."))
        c_d, c_w2, c_c = T("당일", "Today"), T("비중", "Weight"), T("기여", "Contribution")
        tbl = pd.DataFrame({c_d: rel[hold].iloc[-1], c_w2: w, c_c: rel[hold].iloc[-1] * w}).sort_values(c_c)
        table_title(T("종목별 당일 기여", "Contribution by holding today"))
        table(tbl, fmt={c_d: lambda v: pct(v, True), c_w2: lambda v: f"{v:.1%}", c_c: pp},
              neg_cols=(c_d, c_c), index_name=T("종목", "Ticker"))

    live()

# ════════════════════════════════════════════════════════════════════════
# 7. 유동성 / Liquidity
# ════════════════════════════════════════════════════════════════════════

with tab_liq:
    ui.chapter(7)
    part = st.slider(T("체결 가능 비율 (일평균 거래대금 대비)", "Participation rate (share of average daily value)"),
                     5, 30, 20, 5, format="%d%%") / 100
    adv = volume[hold].iloc[-20:].mean()
    lq = risk.liquidity(weights, close[hold].iloc[-1], adv, nav_usd, part)
    worst = lq["청산일"].idxmax()
    dmax = lq.loc[worst, "청산일"]
    h2(T(f"전 종목 청산에 최대 {dmax:.3f}일", f"Full liquidation takes at most {dmax:.3f} days"))
    p(T(f"일평균 거래대금의 {part:.0%}까지 체결할 수 있다고 가정하면, 청산에 가장 오래 걸리는 {worst}도 "
        f"{dmax:.3f}일이면 정리된다. "
        + ("현재 운용 규모에서 유동성 위험은 무시할 수준이다." if lq["청산일"].max() < 0.5
           else "하루 이상 걸리는 종목이 있어 청산 계획이 필요하다."),
        f"Assuming trades of up to {part:.0%} of average daily traded value, even the slowest holding, {worst}, "
        f"is exited in {dmax:.3f} days. "
        + ("Liquidity risk is negligible at this size." if lq["청산일"].max() < 0.5
           else "Some holdings take more than a day; plan the exit.")))
    c_pos, c_adv, c_sh, c_days = (T("보유 금액", "Position"), T("일평균 거래대금", "Avg daily value"),
                                  T("거래대금 대비", "Share of ADV"), T("청산 소요 (일)", "Days to exit"))
    table_title(T("종목별 청산 소요", "Time to liquidate by holding"))
    table(pd.DataFrame({c_pos: lq["보유($)"], c_adv: lq["ADV($)"], c_sh: lq["ADV 대비"], c_days: lq["청산일"]}),
          fmt={c_pos: ui.usd, c_adv: lambda v: f"{ui.usd(v / 1e6)}M" if not math.isnan(v) else "–",
               c_sh: lambda v: f"{v:.3%}", c_days: lambda v: f"{v:.3f}"},
          index_name=T("종목", "Ticker"))
    note(T("일평균 거래대금은 최근 20거래일 평균. 체결 비용과 슬리피지는 반영하지 않았다.",
           "Average daily value over the last 20 trading days. Transaction costs and slippage are not modelled."))

# ════════════════════════════════════════════════════════════════════════
# 성과 기여도 / Attribution
# ════════════════════════════════════════════════════════════════════════

with tab_attr:
    ui.chapter(8)
    A_PER = {21: T("1개월", "1 month"), 63: T("3개월", "3 months"), 126: T("6개월", "6 months"), 252: T("1년", "1 year")}
    n_back = st.radio(T("돌아볼 기간", "Look-back period"), list(A_PER), index=1, horizontal=True,
                      format_func=lambda k: A_PER[k], key="attr_period")
    rr = rets_all[cols].dropna()
    if len(rr) < n_back + 120:
        p(T("이 기간을 나누기에는 함께 거래된 날이 모자란다. 기간을 줄이거나 최근 상장 종목을 빼 보라.",
            "Not enough common trading history for this period. Shorten it or remove a recent listing."))
    else:
        per_idx = rr.index[-n_back:]
        start, end = per_idx[0], per_idx[-1]
        Rp_ = rets_all.loc[per_idx, hold]
        rb_ = rets_all.loc[per_idx, bench]
        hc, rp_, rbx = risk.holding_attribution(Rp_, weights, rb_)
        P_, B_ = float((1 + rp_).prod() - 1), float((1 + rbx).prod() - 1)
        ex_ = P_ - B_

        # 섹터 분해 — 벤치 섹터 비중과 섹터 ETF 가 있어야 한다
        etf_map = data.sector_etfs(bench)
        secs_, bw_ = get_sectors(tuple(t for t in hold if t != bench), bench)
        brin = None
        if bw_:
            etf_close, _ = get_daily(tuple(sorted(set(etf_map.values()))))
            etf_r = etf_close.pct_change(fill_method=None)
            SR_ = pd.DataFrame({k: etf_r[v] for k, v in etf_map.items() if v in etf_r}).reindex(per_idx)
            if SR_.notna().all().any():
                brin = risk.brinson_attribution(Rp_, weights, secs_, rb_, bw_, SR_.dropna(axis=1, how="all"))

        # 요인 분해 — 베타는 기간 시작 **전** 250거래일로 추정한다
        fac = data.factor_returns(close)
        pre = rets_all.loc[:start].iloc[:-1]
        est = pre[cols].dropna().iloc[-250:]
        fb = risk.factor_betas(est, fac.reindex(est.index))
        fct = None
        if bench in fb.index and all(t in fb.index for t in hold if t != bench):
            fct, act_beta = risk.factor_attribution(Rp_, weights, rb_, fb.drop(columns="R2"),
                                                    fb.loc[bench].drop("R2"), fac.reindex(per_idx))

        sec_name = lambda k: data.SECTOR_NAME.get(k, (k, k))[0 if ui.LANG == "ko" else 1]
        lead = T("앞섰다", "beat") if ex_ >= 0 else T("뒤졌다", "trailed")
        if brin is not None:
            tab_b, tot_b = brin
            big = T("섹터 배분", "sector allocation") if abs(tot_b["배분"]) >= abs(tot_b["선택"]) else T("종목 선택", "stock selection")
            h2(T(f"지난 {A_PER[n_back]} {bench_name}보다 {pp(ex_)} {lead} — 더 큰 몫은 {big}",
                 f"Over the past {A_PER[n_back]} the mix {lead} the {bench_name} by {pp(ex_)}, mostly through {big}"))
        else:
            h2(T(f"지난 {A_PER[n_back]} {bench_name}보다 {pp(ex_)} {lead}",
                 f"Over the past {A_PER[n_back]} the mix {lead} the {bench_name} by {pp(ex_)}"))
        top_c, bot_c = hc.drop("현금", errors="ignore").idxmax(), hc.drop("현금", errors="ignore").idxmin()
        p(T(f"{start:%Y-%m-%d} 종가에 지금 비중으로 사서 {end:%Y-%m-%d}까지 그대로 들고 있었다고 보고 나눴다. "
            f"북 {pct(P_, True, 1)}, {bench_name} {pct(B_, True, 1)}. 가장 많이 보탠 종목은 {top_c}({pp(hc[top_c])}), "
            f"가장 많이 깎은 종목은 {bot_c}({pp(hc[bot_c])})다.",
            f"Assumes today's weights were bought at the {start:%Y-%m-%d} close and held unchanged to {end:%Y-%m-%d}. "
            f"Book {pct(P_, True, 1)}, {bench_name} {pct(B_, True, 1)}. The largest contributor was {top_c} "
            f"({pp(hc[top_c])}); the largest detractor was {bot_c} ({pp(hc[bot_c])})."))
        k_items = [(T("북 수익", "Book return"), pct(P_, True, 1), P_ < 0),
                   (T("벤치 수익", "Benchmark return"), pct(B_, True, 1), B_ < 0),
                   (T("초과수익", "Active return"), pp(ex_), ex_ < 0)]
        if brin is not None:
            k_items += [(T("섹터 배분", "Allocation"), pp(tot_b["배분"]), tot_b["배분"] < 0),
                        (T("종목 선택", "Selection"), pp(tot_b["선택"]), tot_b["선택"] < 0)]
        keys(k_items)

        # 종목별 기여
        figure(T("종목별 기여", "Contribution by holding"),
               T("w·(종목 수익 − 벤치 수익)을 매일 계산해 이어 붙인 값. 합 = 초과수익",
                 "Daily w·(holding return − benchmark return), linked over the period. Sum = active return"))
        hh_ = hc[hc.abs() > 1e-6].sort_values()
        lab_ = [T("현금", "Cash") if i == "현금" else i for i in hh_.index]
        fig = go.Figure(go.Bar(y=lab_, x=hh_.values, orientation="h",
                               marker_color=[ui.NEG_BAR if v < 0 else ui.POS_BAR for v in hh_.values],
                               text=[pp(v) for v in hh_.values], textposition="outside", cliponaxis=False,
                               hovertemplate="%{y} %{x:+.2%}<extra></extra>"))
        span = float(hh_.abs().max()) * 1.35 or 0.01
        fig.update_xaxes(range=[-span, span], tickformat=".1%", zeroline=True, zerolinecolor=ui.INK, zerolinewidth=1)
        fig.update_yaxes(type="category", showgrid=False)
        st.plotly_chart(ui.layout(fig, 60 + 32 * len(lab_), legend=False), width="stretch",
                        config={"displayModeBar": False})
        note(T("현금은 수익 0으로 본다. 벤치가 오른 기간에는 현금이 초과수익을 깎는다.",
               "Cash earns zero, so it detracts when the benchmark rises."))

        # 섹터 분해
        if brin is not None:
            c_wp, c_wb, c_rp, c_rb, c_al, c_se, c_tt = (
                T("북 비중", "Book wt"), T("벤치 비중", "Bench wt"), T("북 섹터 수익", "Book sector ret"),
                T("벤치 섹터 수익", "Bench sector ret"), T("배분", "Allocation"), T("선택", "Selection"),
                T("합계", "Total"))
            show = tab_b.rename(columns={"북 비중": c_wp, "벤치 비중": c_wb, "북 섹터 수익": c_rp,
                                         "벤치 섹터 수익": c_rb, "배분": c_al, "선택": c_se, "합계": c_tt})
            show.index = [sec_name(i) for i in show.index]
            show = show.iloc[::-1]
            table_title(T("섹터별 분해 (Brinson-Fachler)", "Sector attribution (Brinson-Fachler)"),
                        T("배분 = 벤치보다 많이/적게 담은 섹터가 벤치 평균보다 잘/못 간 몫 · 선택 = 같은 섹터 안에서 고른 종목이 섹터보다 잘/못 간 몫",
                          "Allocation: over/underweighting sectors that beat/lagged the benchmark · Selection: holdings beating/lagging their own sector"))
            f_w = lambda v: "–" if pd.isna(v) or v == 0 else f"{v:.1%}"
            f_r = lambda v: "–" if pd.isna(v) else pct(v, True, 1)
            table(show, fmt={c_wp: f_w, c_wb: f_w, c_rp: f_r, c_rb: f_r, c_al: pp, c_se: pp, c_tt: pp},
                  neg_cols=(c_al, c_se, c_tt), index_name=T("섹터", "Sector"))
            note(T(f"섹터 수익은 {', '.join(sorted(set(etf_map.values())))} ETF로 대신했다. 벤치 섹터 비중은 Yahoo 펀드 자료의 현재 값에서 "
                   f"섹터 수익으로 거꾸로 되짚어 매일 비중을 만들었다(표의 비중은 기간 평균). 섹터 ETF로 다시 만든 벤치({pct(tot_b['재구성벤치'], True, 1)})와 실제 {bench}의 차이 "
                   f"{pp(tot_b['재구성오차'])}는 '벤치 재구성 오차'로 남겼다 — 배분 {pp(tot_b['배분'])} + 선택 {pp(tot_b['선택'])} "
                   f"+ 오차 {pp(tot_b['재구성오차'])} = 초과수익 {pp(ex_)}. 선택에는 상호작용 효과가 들어 있다.",
                   f"Sector returns are proxied by the {', '.join(sorted(set(etf_map.values())))} ETFs; benchmark sector weights "
                   f"start from Yahoo's current fund data and are rolled back through sector returns to get each day's weights "
                   f"(the table shows period averages). The ETF-rebuilt benchmark returned "
                   f"{pct(tot_b['재구성벤치'], True, 1)}; its gap to the actual {bench}, {pp(tot_b['재구성오차'])}, is shown as "
                   f"reconstruction error: allocation {pp(tot_b['배분'])} + selection {pp(tot_b['선택'])} + error "
                   f"{pp(tot_b['재구성오차'])} = active {pp(ex_)}. Selection includes the interaction term."))
            if abs(tot_b["재구성오차"]) > max(0.01, 0.5 * abs(ex_)):
                p(T(f"주의: 섹터 ETF로 다시 만든 벤치가 실제 {bench}와 {pp(tot_b['재구성오차'])} 어긋난다. 초과수익에 비해 큰 차이라 "
                    "배분·선택의 크기는 방향 정도로만 읽어야 한다. 섹터 ETF의 종목 비중 상한과 지수 구성 차이 때문이며, "
                    "기간이 길수록 커진다.",
                    f"Caution: the ETF-rebuilt benchmark is off from the actual {bench} by {pp(tot_b['재구성오차'])}, large "
                    "relative to the active return, so read the allocation and selection figures for direction only. The gap "
                    "comes from sector-ETF weight caps and index construction differences, and grows with the period."))
        else:
            p(T(f"{bench}의 섹터 비중을 받지 못해 섹터별 분해는 건너뛰었다(ETF가 아닌 지수이거나 자료 없음).",
                f"Sector weights for {bench} were unavailable (not a fund, or no data), so sector attribution is skipped."))

        # 요인 분해
        if fct is not None:
            c_ab, c_fr, c_ct = T("활성 베타", "Active beta"), T("요인 수익", "Factor return"), T("기여", "Contribution")
            fr_ = (1 + fac.reindex(per_idx)[act_beta.index].fillna(0.0)).prod() - 1
            ft = pd.DataFrame({c_ab: act_beta, c_fr: fr_, c_ct: fct[act_beta.index]})
            ft.loc[T("종목 고유", "Stock-specific")] = [np.nan, np.nan, fct["종목 고유"]]
            if ui.LANG == "en":
                ft.index = [{"시장(ACWI)": "Market (ACWI)", "소형주 스프레드(IWM−SPY)": "Small-cap spread (IWM−SPY)",
                             "장기채(TLT)": "Long bonds (TLT)", "유가(USO)": "Oil (USO)", "달러(UUP)": "Dollar (UUP)"}.get(i, i)
                            for i in ft.index]
            table_title(T("요인별 분해", "Factor attribution"),
                        T("기여 = (북 베타 − 벤치 베타) × 요인 수익을 매일 계산해 이은 값. 나머지가 종목 고유",
                          "Contribution = (book beta − benchmark beta) × factor return, daily and linked. The rest is stock-specific"))
            table(ft, fmt={c_ab: lambda v: "–" if pd.isna(v) else f"{v:+.2f}".replace("-", "−"), c_fr: lambda v: "–" if pd.isna(v) else pct(v, True, 1), c_ct: pp},
                  neg_cols=(c_ct,), index_name=T("요인", "Factor"), em_rows=(T("종목 고유", "Stock-specific"),))
            note(T(f"베타는 기간 시작 전 {len(est)}거래일로 추정했다. 같은 기간으로 추정하면 종목 고유 몫이 작게 나온다(표본내 과적합). "
                   "종목 고유가 크면 초과수익은 시장·금리·유가 같은 공통 요인보다 개별 종목에서 나왔다는 뜻이다.",
                   f"Betas are estimated on the {len(est)} trading days before the period. Estimating on the same period "
                   "would shrink the stock-specific part (in-sample fit). A large stock-specific share means the active "
                   "return came from the individual names rather than common factors."))

        # 실력인가 운인가 — 무작위 포트폴리오
        table_title(T("실력인가 운인가", "Skill or luck?"),
                    T("같은 비중·같은 기간에 종목만 무작위로 바꾼 포트폴리오 2만 개와 비교한다",
                      "Compares the mix with 20,000 portfolios that keep the weights and period but swap the stocks at random"))
        if st.toggle(T("무작위 포트폴리오와 비교하기 (후보 종목을 받느라 처음 한 번은 1분 반쯤 걸린다)",
                       "Compare with random portfolios (downloads candidate stocks; the first run takes about 90 seconds)"),
                     key="attr_rand"):
            try:
                uni_, ucl_ = get_universe(f"{pd.Timestamp.today() - pd.Timedelta(days=400):%Y-%m}-01", f"{end:%Y-%m-%d}")
            except Exception as e:
                uni_ = None
                p(T(f"후보 종목 가격을 받지 못했다({type(e).__name__}). 잠시 뒤 다시 켜 보라.",
                    f"Could not download the candidate stocks ({type(e).__name__}). Try again shortly."))
            if uni_ is not None:
                prev_ = ucl_.index[ucl_.index < start]
                ok_ = ucl_.loc[per_idx[0]:end] if len(prev_) else ucl_.iloc[0:0]
                if len(prev_) == 0 or ok_.empty or end not in ucl_.index:
                    p(T("후보 종목 가격이 이 기간을 덮지 못한다.", "Candidate prices do not cover this period."))
                else:
                    base_ = ucl_.loc[prev_[-1]]
                    full_ = ucl_.reindex([prev_[-1]] + list(per_idx)).notna().all()
                    cr_ = (ucl_.loc[end] / base_ - 1)[full_]
                    cr_ = cr_[cr_.index.isin(uni_.ticker)]
                    usec_ = uni_.set_index("ticker").loc[cr_.index, "sector"].to_numpy()
                    cand_ = cr_.to_numpy()
                    own_ = (1 + Rp_.fillna(0.0)).prod() - 1
                    us_ = lambda t: t != bench and "." not in t and t.isascii()
                    all_idx = np.arange(len(cand_))
                    rows_, sims_sec = {}, None
                    for lab_k, pick in [(T("S&P 1500 아무 종목", "Any S&P 1500 stock"),
                                         lambda t: all_idx if us_(t) and secs_.get(t) else None),
                                        (T("같은 섹터 종목", "Same-sector stocks"),
                                         lambda t: np.where(usec_ == secs_.get(t))[0] if us_(t) and secs_.get(t) else None)]:
                        rt_ = risk.random_portfolio_test(weights, own_, {t: pick(t) for t in weights}, cand_)
                        rows_[lab_k] = [1 - rt_.pct, rt_.p_value, float(np.median(rt_.sims)),
                                        float(np.percentile(rt_.sims, 5)), float(np.percentile(rt_.sims, 95))]
                        sims_sec = rt_
                    rt_ = sims_sec
                    if not rt_.replaced:
                        p(T("바꿔 넣을 수 있는 미국 개별 종목이 없어 검정을 건너뛰었다(ETF·해외 종목은 그대로 둔다).",
                            "No US single stocks to swap, so the test is skipped (ETFs and non-US listings stay fixed)."))
                    else:
                        top_ = 1 - rt_.pct
                        verdict = (T("운만으로는 잘 나오지 않는 성과다", "hard to get by luck alone") if rt_.p_value < 0.05
                                   else T("운과 구별되지 않는다", "indistinguishable from luck"))
                        p(T(f"같은 섹터에서 아무 종목이나 골랐어도 {top_:.0%}는 실제보다 잘 했다(p = {rt_.p_value:.2f}) — {verdict}. "
                            f"실제 {pct(rt_.actual, True, 1)}, 무작위 중앙값 {pct(float(np.median(rt_.sims)), True, 1)}.",
                            f"Picking any stocks from the same sectors would have beaten the mix {top_:.0%} of the time "
                            f"(p = {rt_.p_value:.2f}) — {verdict}. Actual {pct(rt_.actual, True, 1)}, random median "
                            f"{pct(float(np.median(rt_.sims)), True, 1)}."))
                        fig = go.Figure(go.Histogram(x=rt_.sims, nbinsx=60, marker_color=ui.MUTED,
                                                     hovertemplate="%{x:.1%}: %{y}<extra></extra>"))
                        fig.add_vline(x=rt_.actual, line_color=ui.POS_BAR if rt_.actual >= 0 else ui.NEG_BAR, line_width=2,
                                      annotation_text=T("실제", "Actual"), annotation_position="top")
                        fig.update_xaxes(tickformat=".0%")
                        fig.update_yaxes(showticklabels=False, showgrid=False)
                        st.plotly_chart(ui.layout(fig, 220, legend=False), width="stretch", config={"displayModeBar": False})
                        c_top, c_pv, c_med, c_lo, c_hi = (T("실제보다 잘 한 비율", "Share beating the mix"), T("p값", "p-value"),
                                                          T("무작위 중앙값", "Random median"), T("하위 5%", "5th pct"),
                                                          T("상위 5%", "95th pct"))
                        rt_tab = pd.DataFrame.from_dict(rows_, orient="index", columns=[c_top, c_pv, c_med, c_lo, c_hi])
                        f_r1 = lambda v: pct(v, True, 1)
                        table(rt_tab, fmt={c_top: lambda v: f"{v:.0%}", c_pv: lambda v: f"{v:.2f}", c_med: f_r1, c_lo: f_r1, c_hi: f_r1},
                              index_name=T("바꿔 넣은 후보", "Swapped from"))
                        note(T(f"미국 개별 종목 {len(rt_.replaced)}개만 바꾸고, ETF·해외 종목({', '.join(rt_.kept) or '없음'})과 현금은 그대로 뒀다. "
                               f"후보는 S&P 500·400·600 현재 구성종목에서 섹터 비율대로 1/3을 뽑아 이 기간 내내 거래된 {len(cand_):,}개이고, 섹터는 GICS다(보유 종목은 Yahoo 분류라 경계가 조금 다르다). "
                               "지금 구성종목이라 이 기간에 빠진 종목이 없다(생존 편향). 실제 포트폴리오가 무작위 분포의 위쪽 5% 안에 들어야(p < 0.05) "
                               "종목 고르기에 실력이 있다고 말할 수 있다. 기간이 짧고 종목이 적을수록 그 문턱을 넘기 어렵다.",
                               f"Only the {len(rt_.replaced)} US single stocks are swapped; ETFs and non-US listings "
                               f"({', '.join(rt_.kept) or 'none'}) and cash stay fixed. Candidates are {len(cand_):,} stocks drawn as a sector-proportional third of the current "
                               "S&P 500/400/600 members that traded throughout the period, grouped by GICS sector (holdings use Yahoo's classification, so the borders "
                               "differ slightly). Using current members leaves out stocks that dropped out (survivorship bias). The mix needs to "
                               "sit in the top 5% of the random distribution (p < 0.05) before stock picking can be called skill; short periods "
                               "and few holdings make that bar hard to clear."))

# ════════════════════════════════════════════════════════════════════════
# 읽는 법 / Guide
# ════════════════════════════════════════════════════════════════════════

with tab_help:
    ui.chapter(0)

    def help_table(index_ko, index_en, cols_):
        """cols_: [(열 이름 ko, en, 값 ko 목록, 값 en 목록)]"""
        return pd.DataFrame({T(ck, ce): (vk if ui.LANG == "ko" else ve) for ck, ce, vk, ve in cols_},
                            index=index_ko if ui.LANG == "ko" else index_en)

    h2(T("탭별로 무엇을 보나", "What each tab answers"))
    table(help_table(["전망", "리스크", "VaR", "검증", "스트레스", "기여도", "장중", "유동성"],
                     ["Outlook", "Risk", "VaR", "Backtest", "Stress", "Attribution", "Intraday", "Liquidity"],
                     [("답하는 질문", "Question",
                       ["남은 기간에 비교 지수를 이길 확률과 나쁠 때의 폭, 모형과 과거 실현치의 차이",
                        "하루 손실 규모와 위험을 많이 내는 종목",
                        "VaR가 계산 방법에 따라 얼마나 달라지나", "어떤 VaR 방법이 과거에 실제로 맞았나",
                        "CPI 발표일이나 과거 위기가 다시 오면, 요인 충격을 주면",
                        "지난 기간 벤치보다 앞서거나 뒤진 이유: 섹터 배분, 종목 선택, 요인. 그게 실력인지 운인지",
                        "지금 장중에 비교 지수 대비 어디쯤인가", "한 번에 다 팔 수 있나"],
                       ["Odds of beating the benchmark, how bad it gets, and model versus realized history",
                        "Size of a one-day loss and which holdings drive risk",
                        "How much VaR depends on the method", "Which VaR method was actually right in the past",
                        "What if a CPI surprise or a past crisis repeats; factor shocks",
                        "Why the mix beat or trailed the benchmark: sector allocation, stock selection, factors; skill or luck",
                        "Where the book stands versus the benchmark right now", "Can the book be exited quickly"]),
                      ("먼저 볼 것", "Look first at",
                       ["하위 5%·1%와 분포 비교 표", "종목별 위험 기여 표의 배수", "방법별 VaR 그림의 점 간격",
                        "사후검증 표의 판정 열", "위기 재현 그림의 붉은 막대", "배분·선택 합계와 종목 고유 몫, 무작위 검정의 p값",
                        "VaR 선과의 거리", "청산 소요 열"],
                       ["5th/1st percentiles and the comparison table", "Multiples in the risk contribution table",
                        "Spread of points in the VaR-by-method chart", "Verdict column of the backtest table",
                        "Red bars in the crisis replay", "Allocation vs selection totals, the stock-specific share, the random-test p-value",
                        "Distance to the VaR lines", "Days-to-exit column"])]),
          text=True, index_name=T("탭", "Tab"))
    note(T("첫 화면의 금액은 절대수익(평균 0 가정), 이 탭들의 '벤치 대비'는 비교 지수를 뺀 상대수익이다. "
           "%p는 상대수익의 차이를 뜻한다.",
           "Front-page values are absolute returns with zero drift; 'relative' here means return minus the "
           "benchmark, in percentage points (pp)."))

    h2(T("숫자를 읽을 때 주의할 점", "Caveats"))
    table(help_table(["표본내 편향", "매매 미반영", "재현 가정", "요인 모형 한계", "데이터"],
                     ["In-sample bias", "No trading", "Replay assumption", "Factor model limits", "Data"],
                     [("내용", "Detail",
                       ["최근 성적이 좋아 고른 종목이면 같은 구간으로 잰 정보비율과 평균 수익은 좋게 나온다.",
                        "기간 끝까지 그대로 들고 간다고 가정한다. 손절이 걸리면 실제 하위 꼬리는 더 얕다.",
                        "검증 탭은 현재 비중을 과거 전 기간 들고 있었다고 가정한 재현이다.",
                        "요인 설명력(R²)이 낮으면 가상 충격은 방향 정도만 참고한다. 프리셋 값은 예시다.",
                        "Yahoo Finance 기준. 분봉은 15분쯤 늦을 수 있고, 일봉은 1시간, 실적 일정은 6시간 캐시한다."],
                       ["If the holdings were chosen on recent performance, in-sample information ratio and mean "
                        "return look better than they should.",
                        "Holdings are assumed unchanged to the end. Stop-losses would make the left tail shallower.",
                        "The backtest replays today's weights over the full history.",
                        "When factor R² is low, read hypothetical shocks for direction only. Presets are examples.",
                        "Yahoo Finance. Intraday bars may lag ~15 minutes; daily prices are cached for an hour and "
                        "earnings dates for six."])]),
          text=True, index_name=T("항목", "Item"))

html_('<div class="q-foot">Risk Lab · ' + T("가격 자료: Yahoo Finance", "Prices: Yahoo Finance") +
      ' · <a href="https://github.com/Giana0412/risk-lab" style="color:inherit">GitHub</a></div>')
