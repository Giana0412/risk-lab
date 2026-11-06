"""Risk Lab — a research-report style risk dashboard for a long-only equity book.

Run:  streamlit run app.py          (?lang=en for English, ?lang=ko for Korean)

Built during the Bloomberg Trading Challenge to check a model book every day.
Calculations live in risk.py, data in data.py, report components in ui.py.
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
from ui import pct, pp, money, num, p, h2, keys, figure, table_title, note, table

T = ui.t

# ── 언어: ?lang=en 이면 영어로 연다. 사이드바에서 바꾸면 주소도 바뀐다 ─────────
_q = st.query_params.get("lang", "ko")
if "lang" not in st.session_state:
    st.session_state.lang = "en" if _q == "en" else "ko"
ui.set_lang(st.session_state.lang)

st.set_page_config(page_title=T("리스크 랩", "Risk Lab"), page_icon="▤", layout="wide")
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


def mday(d) -> str:
    return f"{d:%b %-d}" if ui.LANG == "en" else f"{d:%-m월 %-d일}"


# ── 캐시된 데이터 ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=T("가격 받는 중 (2007~)…", "Downloading prices (2007–)…"))
def get_daily(tickers: tuple[str, ...]):
    return data.daily_prices(list(tickers))


@st.cache_data(ttl=6 * 3600, show_spinner=T("실적 일정 받는 중…", "Downloading earnings calendar…"))
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


@st.cache_data(ttl=60, show_spinner=False)
def get_intraday(tickers: tuple[str, ...]):
    return data.intraday_prices(list(tickers))


# ── 사이드바 ────────────────────────────────────────────────────────────

ui.side_head(T("모델 북 리스크 점검", "Model book risk check"), T("리스크 랩", "Risk Lab"))


def _switch_lang():
    st.session_state.lang = "en" if st.session_state._lang_pick == "English" else "ko"
    st.query_params["lang"] = st.session_state.lang


st.sidebar.radio("Language", ["한국어", "English"], index=1 if ui.LANG == "en" else 0, horizontal=True,
                 key="_lang_pick", on_change=_switch_lang, label_visibility="collapsed")

books = data.load_books()
src = data.book_log_path()

ui.side_section(T("북", "Book"))
if books:
    def _book_label(b):
        kind = b.get("kind_en", b.get("kind", "")) if ui.LANG == "en" else b.get("kind", "")
        return f"{b['date']} · {kind}"
    labels = [_book_label(b) for b in books]
    pick = st.sidebar.selectbox("book", range(len(books)), format_func=lambda i: labels[i],
                                label_visibility="collapsed")
    base_book = books[pick]["book"]
    if data.using_sample():
        st.sidebar.caption(T("예시 북이다. 아래 표에서 종목과 비중을 바꿔 자신의 북을 넣을 수 있다.",
                             "Sample book. Edit tickers and weights below to enter your own."))
else:
    pick = None
    base_book = {"SPY": 0.5, "IWM": 0.5}
    st.sidebar.warning(T("북 기록을 찾지 못해 기본 예시를 쓴다.", "No book log found; using a default example."))

edited = st.sidebar.data_editor(
    pd.DataFrame({"티커": list(base_book), "비중": [float(v) * 100 for v in base_book.values()]}),
    num_rows="dynamic", hide_index=True, width="stretch",
    key=f"book-{books[pick]['date'] if books else 'demo'}",
    column_config={
        "티커": st.column_config.TextColumn(T("티커", "Ticker"), width="small"),
        "비중": st.column_config.NumberColumn(T("비중 (%)", "Weight (%)"), format="%.2f",
                                            min_value=0.0, max_value=100.0, step=0.01),
    },
)
weights = {str(t).strip().upper(): float(w) / 100 for t, w in zip(edited["티커"], edited["비중"])
           if isinstance(t, str) and t.strip() and pd.notna(w) and w > 0}

wsum = sum(weights.values())
over = [t for t, w in weights.items() if w > 0.20 + 1e-9]
if wsum > 1 + 1e-9:
    st.sidebar.error(T(f"비중 합 {wsum:.1%} — 레버리지 금지 제약 위반",
                       f"Weights sum to {wsum:.1%} — breaches the no-leverage constraint"))
if over:
    st.sidebar.error(T(f"종목당 20% 상한 초과: {', '.join(over)}", f"Above the 20% single-name cap: {', '.join(over)}"))
if wsum <= 1 + 1e-9 and not over and weights:
    st.sidebar.caption(T(f"합계 {wsum:.1%} · 현금 {1 - wsum:.1%} · 제약 충족",
                         f"Total {wsum:.1%} · cash {1 - wsum:.1%} · constraints met"))

ui.side_section(T("설정", "Settings"))
end_date = st.sidebar.date_input(T("평가 종료일", "Evaluation end date"),
                                 (pd.Timestamp.today() + pd.Timedelta(days=28)).date(),
                                 help=T("이 날까지 북을 그대로 보유한다고 가정한다 (1장 전망).",
                                        "The book is assumed to be held unchanged until this date (Chapter 1)."))
bench = st.sidebar.text_input(T("벤치마크", "Benchmark"), "ACWI",
                              help=T("상대수익의 기준. Bloomberg Trading Challenge 평가 지수(WLS)에 가까운 ACWI가 기본.",
                                     "Reference for relative return. Defaults to ACWI, the closest tradable proxy "
                                     "for the Bloomberg Trading Challenge index (WLS).")).upper()
with st.sidebar.expander(T("고급", "Advanced")):
    lookback = st.select_slider(T("추정 창 (거래일)", "Estimation window (trading days)"), [125, 250, 500, 750],
                                value=250, help=T("2·3장 계산에 쓰는 과거 구간 길이", "History used in Chapters 2–3"))
    horizon = st.radio(T("2·3장 보유기간", "Horizon for Chapters 2–3"), [1, 5, 10], index=0, horizontal=True,
                       format_func=lambda x: T(f"{x}일", f"{x}d"),
                       help=T("h일 VaR = 1일 × √h (i.i.d. 가정)", "h-day VaR = 1-day × √h (i.i.d. assumption)"))
    nav = st.number_input(T("운용자산 (USD)", "Assets under management (USD)"), value=1_000_000, step=100_000)
    if src:
        st.caption(T("북 출처: ", "Book source: ") + f"`{src.name}`")

if not weights:
    st.info(T("사이드바에 종목과 비중을 넣는다.", "Enter tickers and weights in the sidebar."))
    st.stop()

# ── 데이터 준비 ─────────────────────────────────────────────────────────

all_tk = tuple(sorted(set(weights) | {bench} | set(data.FACTOR_TICKERS)))
close, volume = get_daily(all_tk)
missing = [t for t in weights if t not in close or close[t].dropna().empty]
if missing:
    st.warning(T(f"가격을 못 받은 종목은 빼고 계산한다: {', '.join(missing)}",
                 f"No prices for these tickers; excluded: {', '.join(missing)}"))
    weights = {t: w for t, w in weights.items() if t not in missing}
if bench not in close or not weights:
    st.error(T("벤치마크 또는 북 가격이 없다.", "No prices for the benchmark or the book."))
    st.stop()

rets_all = close.pct_change(fill_method=None)
hold = list(weights)
cols = list(dict.fromkeys(hold + [bench]))  # 벤치를 북에도 담으면 열이 중복된다
win = rets_all[cols].dropna().iloc[-lookback:]
if len(win) < 60:
    st.error(T(f"공통 표본이 {len(win)}일뿐이다. 최근 상장 종목이 있으면 추정 창을 줄인다.",
               f"Only {len(win)} common days of data. Shorten the window if a holding listed recently."))
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

remaining = data.trading_days(pd.Timestamp(asof) + pd.Timedelta(days=1), pd.Timestamp(end_date))

# ── 머리 ────────────────────────────────────────────────────────────────

cash = 1 - sum(weights.values())
ui.masthead(T("리스크 랩 · 모델 북 리스크 점검", "Risk Lab · Model book risk check"),
            T(f"{asof:%Y년 %-m월 %-d일} 종가 기준", f"As of close, {asof:%B %-d, %Y}"),
            T(f"벤치마크 {bench} · {len(hold)}종목 · 현금 {cash:.1%} · 평가 종료까지 {len(remaining)}거래일 · "
              f"추정 구간 {len(win)}거래일 · 사용 방법은 부록",
              f"Benchmark {bench} · {len(hold)} holdings · cash {cash:.1%} · {len(remaining)} trading days to "
              f"evaluation end · {len(win)}-day estimation window · see Appendix for usage"))
if len(win) < lookback:
    st.caption(T(f"공통 표본이 {len(win)}일로 요청한 {lookback}일보다 짧다 (최근 상장 종목).",
                 f"Common sample is {len(win)} days, shorter than the requested {lookback} (recent listing)."))

tab_out, tab_sum, tab_var, tab_bt, tab_st, tab_live, tab_liq, tab_help = st.tabs([
    T("1. 전망", "1. Outlook"), T("2. 리스크 개요", "2. Risk"), T("3. VaR 방법론", "3. VaR"),
    T("4. 모형 검증", "4. Backtest"), T("5. 스트레스 테스트", "5. Stress"), T("6. 장중", "6. Intraday"),
    T("7. 유동성", "7. Liquidity"), T("부록. 사용 방법", "Appendix")])

# ════════════════════════════════════════════════════════════════════════
# 1. 전망 / Outlook
# ════════════════════════════════════════════════════════════════════════

with tab_out:
    ui.chapter(1)
    if len(remaining) == 0:
        p(T("남은 거래일이 없다. 사이드바의 평가 종료일을 확인한다.",
            "No trading days remain. Check the evaluation end date in the sidebar."))
    else:
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
        excess_ex = risk.portfolio_returns(R_ex, weights) - bret
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

        H = len(remaining)
        out = risk.simulate_outcome(excess_ex, H, jumps_h)
        s_tot, s_base = out.stats("total"), out.stats("base")
        g = risk.garch_fit(excess_ex)
        long_cols = rets_all[cols].dropna().iloc[-1000:]
        ex_long = risk.portfolio_returns(long_cols[hold], weights) - long_cols[bench]
        s_real = risk.horizon_stats(risk.realized_horizon(ex_long, H)) if len(ex_long) > H + 60 else None
        vr = risk.variance_ratio(ex_long, H)
        p_win, q05, q01 = s_tot["P(상대 > +0%)"], s_tot["5%"], -s_tot["VaR 99%"]

        h2(T(f"평가 종료까지 {H}거래일, 벤치를 이길 확률 {p_win:.0%}",
             f"{H} trading days to go: {p_win:.0%} probability of beating the benchmark"))
        lean_ko = ("벤치에 뒤처지는 쪽으로 기운다" if p_win < 0.45 else
                   "어느 쪽으로도 기울지 않는다" if p_win < 0.55 else "벤치를 앞서는 쪽으로 기운다")
        lean_en = ("leans toward underperformance" if p_win < 0.45 else
                   "is roughly balanced" if p_win < 0.55 else "leans toward outperformance")
        p(T(f"현재 북을 {mday(remaining[0])}부터 {mday(remaining[-1])}까지 그대로 보유한다고 가정하면, "
            f"{bench} 대비 상대수익이 플러스일 확률은 <b>{p_win:.0%}</b>로 추정되며 분포는 {lean_ko}. "
            f"하위 5% 시나리오는 <b>{pp(q05)}</b>, 하위 1% 시나리오는 <b>{pp(q01)}</b>이다. "
            f"운용자산 기준으로 하위 5%는 약 {money(q05 * nav)}에 해당한다.",
            f"If the current book is held unchanged from {mday(remaining[0])} to {mday(remaining[-1])}, the "
            f"probability of a positive return relative to {bench} is estimated at <b>{p_win:.0%}</b>, and the "
            f"distribution {lean_en}. The 5th-percentile scenario is <b>{pp(q05)}</b> and the 1st-percentile "
            f"scenario is <b>{pp(q01)}</b>. On the current asset base, the 5th percentile is about "
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
        "방법 간 적합성은 4장에서 검증한다.",
        f"For the same book and data, the {horizon}-day 99% relative VaR ranges from <b>{pp(-rel99.min())}</b> "
        f"({METHOD_NAMES[lo_m]}) to <b>{pp(-rel99.max())}</b> ({METHOD_NAMES[hi_m]}). When methods that weight "
        "recent data (EWMA, FHS-GARCH) give larger values, recent volatility is above the sample average. "
        "Chapter 4 tests which method fits."))

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
    lq = risk.liquidity(weights, close[hold].iloc[-1], adv, nav, part)
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
          fmt={c_pos: money, c_adv: lambda v: f"{money(v / 1e6)}M" if not math.isnan(v) else "–",
               c_sh: lambda v: f"{v:.3%}", c_days: lambda v: f"{v:.3f}"},
          index_name=T("종목", "Ticker"))
    note(T("일평균 거래대금은 최근 20거래일 평균. 체결 비용과 슬리피지는 반영하지 않았다.",
           "Average daily value over the last 20 trading days. Transaction costs and slippage are not modelled."))

# ════════════════════════════════════════════════════════════════════════
# 부록. 사용 방법 / Appendix. How to use
# ════════════════════════════════════════════════════════════════════════

with tab_help:
    ui.chapter(0)

    def help_table(index_ko, index_en, cols_):
        """cols_: [(열 이름 ko, en, 값 ko 목록, 값 en 목록)]"""
        df = pd.DataFrame({T(ck, ce): (vk if ui.LANG == "ko" else ve) for ck, ce, vk, ve in cols_},
                          index=index_ko if ui.LANG == "ko" else index_en)
        return df

    h2(T("실행", "Running"))
    p(T("저장소 폴더에서 <code>pip install -r requirements.txt</code> 후 <code>streamlit run app.py</code>를 "
        "실행하고 브라우저에서 localhost:8501을 연다. 저장소 폴더 안에서 실행해야 지면 색과 글꼴 설정"
        "(.streamlit/config.toml)이 적용된다. 처음 열 때 2007년부터의 가격을 받느라 20초 안팎이 걸리고, "
        "이후에는 캐시를 쓴다. 주소에 <code>?lang=en</code>을 붙이면 영어로 연다.",
        "From the repository folder, run <code>pip install -r requirements.txt</code> and then "
        "<code>streamlit run app.py</code>, and open localhost:8501. Run it from the repository folder so the "
        "page colours and fonts in .streamlit/config.toml apply. The first load takes about 20 seconds to "
        "download prices back to 2007; later loads use the cache. Add <code>?lang=ko</code> to the URL for Korean."))

    h2(T("북 고르기와 고치기", "Choosing and editing a book"))
    p(T("왼쪽 사이드바의 <b>북</b> 목록은 북 기록 파일에서 읽는다. 환경변수 <code>BOOK_LOG</code>가 가리키는 JSONL "
        "파일이 있으면 그것을, 없으면 저장소의 예시 북(sample_books.jsonl)을 쓴다. 날짜를 고르면 그날 북의 종목과 "
        "비중이 아래 표에 채워진다. 표는 직접 고칠 수 있다.",
        "The <b>Book</b> list in the sidebar is read from a book log. If the environment variable "
        "<code>BOOK_LOG</code> points to a JSONL file, that file is used; otherwise the sample books in the "
        "repository (sample_books.jsonl). Choosing a date fills the table below with that book's tickers and "
        "weights, which you can edit directly."))
    table(help_table(["비중 바꾸기", "종목 추가", "종목 삭제", "원래 북으로 되돌리기"],
                     ["Change a weight", "Add a holding", "Remove a holding", "Restore the original book"],
                     [("방법", "How", ["비중 칸을 두 번 눌러 숫자를 입력한다. 단위는 %다.",
                                       "표 맨 아래 빈 줄에 티커와 비중을 넣는다. 야후 파이낸스 티커를 쓴다.",
                                       "줄 왼쪽 칸을 눌러 선택한 뒤 Delete 키를 누른다.",
                                       "다른 날짜를 골랐다가 다시 고르거나 페이지를 새로 고친다."],
                       ["Double-click the weight cell and type a number, in percent.",
                        "Type a ticker and weight in the empty bottom row. Use Yahoo Finance tickers.",
                        "Click the cell at the left of the row to select it, then press Delete.",
                        "Select another date and back, or reload the page."])]),
          text=True, index_name=T("할 일", "Task"))
    note(T("표 아래 줄에 비중 합계와 현금이 표시된다. 합계가 100%를 넘거나 한 종목이 20%를 넘으면 운용 제약"
           "(레버리지 금지, 종목당 20%) 위반 경고가 뜬다. 고친 비중은 화면에서만 쓰이고 기록 파일은 바뀌지 않는다.",
           "The line below the table shows total weight and cash. A warning appears if the total exceeds 100% or "
           "a holding exceeds 20% (no leverage, 20% single-name cap). Edits apply to the page only; the log file "
           "is not changed."), source=False)

    h2(T("설정", "Settings"))
    table(help_table(["평가 종료일", "벤치마크", "추정 창 (고급)", "보유기간 (고급)", "운용자산 (고급)"],
                     ["Evaluation end date", "Benchmark", "Estimation window (Advanced)", "Horizon (Advanced)",
                      "Assets under management (Advanced)"],
                     [("뜻", "Meaning",
                       ["1장 전망의 시뮬레이션 기간을 정한다. 기본값은 오늘부터 4주 뒤.",
                        "상대수익을 잴 기준. Bloomberg Trading Challenge 평가 지수(WLS)에 가까운 ACWI가 기본이다.",
                        "2·3장 계산에 쓰는 과거 구간 길이. 짧을수록 최근 상황을, 길수록 평균적인 상황을 반영한다.",
                        "2·3장 VaR의 기간. 1일 외에는 √기간을 곱한다(3장 FHS-GARCH는 경로 시뮬레이션).",
                        "금액 환산에 쓰는 규모. 기본값은 100만 달러."],
                       ["Sets the simulation period for Chapter 1. Defaults to four weeks from today.",
                        "Reference for relative return. Defaults to ACWI, close to the Bloomberg Trading Challenge "
                        "index (WLS).",
                        "History used in Chapters 2–3. Shorter reflects recent conditions; longer, average conditions.",
                        "VaR horizon in Chapters 2–3. Beyond one day, scaled by √horizon (FHS-GARCH in Chapter 3 "
                        "simulates paths instead).",
                        "Used to convert percentages to dollars. Defaults to USD 1 million."])]),
          text=True, index_name=T("항목", "Setting"))

    h2(T("장별로 무엇을 보나", "What each chapter answers"))
    table(help_table(["1. 전망", "2. 리스크 개요", "3. VaR 방법론", "4. 모형 검증", "5. 스트레스 테스트", "6. 장중",
                      "7. 유동성"],
                     ["1. Outlook", "2. Risk", "3. VaR", "4. Backtest", "5. Stress", "6. Intraday", "7. Liquidity"],
                     [("답하는 질문", "Question",
                       ["남은 기간에 벤치를 이길 확률과 나쁠 때의 폭", "하루 손실 규모와 위험을 많이 내는 종목",
                        "VaR가 계산 방법에 따라 얼마나 달라지나", "어떤 VaR 방법이 과거에 실제로 맞았나",
                        "CPI 발표일이나 과거 위기가 다시 오면", "지금 장중에 벤치 대비 어디쯤인가", "한 번에 다 팔 수 있나"],
                       ["Odds of beating the benchmark by the end date, and how bad it gets",
                        "Size of a one-day loss and which holdings drive risk",
                        "How much VaR depends on the method", "Which VaR method was actually right in the past",
                        "What if a CPI surprise or a past crisis repeats", "Where the book stands versus the benchmark now",
                        "Can the book be exited quickly"]),
                      ("먼저 볼 것", "Look first at",
                       ["제목의 확률, 하위 5%·1%", "종목별 위험 기여 표의 배수", "그림 3-1의 점 간격", "표 4-1의 판정 열",
                        "그림 5-1, 5-2의 붉은 막대", "제목의 벤치 대비 수치와 VaR 선", "청산 소요 열"],
                       ["The probability in the heading; 5th and 1st percentiles", "Multiples in the risk contribution table",
                        "Spread of points in Figure 3-1", "Verdict column in Table 4-1",
                        "Red bars in Figures 5-1 and 5-2", "Relative figure in the heading and the VaR lines",
                        "Days-to-exit column"])]),
          text=True, index_name=T("장", "Chapter"))

    h2(T("쓰는 순서", "Suggested routine"))
    p(T("<b>매일 아침.</b> 사이드바에서 가장 최근 날짜의 북을 고르고 1장을 본다. 벤치를 이길 확률과 하위 5%가 "
        "전날보다 크게 나빠졌는지, 실적 일정 표에 '평가 기간: 포함'인 종목이 생겼는지 확인한다.",
        "<b>Each morning.</b> Select the latest book and read Chapter 1. Check whether the probability of beating "
        "the benchmark or the 5th percentile has deteriorated, and whether any holding now reports earnings "
        "within the window."))
    p(T("<b>큰 발표 전날.</b> CPI·FOMC 같은 일정 전에는 5장의 CPI 발표일 재현과 가상 충격을 본다. 특정 시나리오에서 "
        "벤치 대비 손실이 큰 종목이 있으면 2장의 위험 기여 표에서 그 종목의 배수를 함께 본다.",
        "<b>Before a major release.</b> Ahead of CPI or FOMC, read the CPI replay and hypothetical shocks in "
        "Chapter 5. If one holding drives a large relative loss, check its multiple in Chapter 2."))
    p(T("<b>장중.</b> 6장에서 '60초마다 갱신'을 켜 둔다. 벤치 대비 선이 붉은 점선(VaR 99%) 아래로 내려가면 "
        "추정 구간 기준으로 100거래일에 한 번 나올 손실이다.",
        "<b>During the session.</b> Turn on 'Refresh every 60 seconds' in Chapter 6. A relative line below the "
        "red dotted VaR 99% marks a loss expected about once in 100 trading days."))
    p(T("<b>비중을 바꿔 보고 싶을 때.</b> 사이드바 표에서 비중을 고치면 모든 장이 다시 계산된다. 실제 주문 전에 "
        "1장의 확률과 하위 5%가 어떻게 움직이는지 비교하는 용도로 쓴다.",
        "<b>Trying new weights.</b> Editing weights in the sidebar recomputes every chapter. Use it before "
        "trading to compare how the Chapter 1 probability and 5th percentile move."))

    h2(T("새 데이터 받기", "Refreshing data"))
    table(help_table(["일봉 가격", "실적 일정", "장중 분봉", "모형 검증 결과"],
                     ["Daily prices", "Earnings calendar", "Intraday bars", "Backtest results"],
                     [("보관 시간", "Cached for", ["1시간", "6시간", "60초", "1시간"], ["1 hour", "6 hours", "60 seconds", "1 hour"]),
                      ("강제로 새로 받으려면", "To force a refresh",
                       ["오른쪽 위 메뉴 → Clear cache, 또는 키보드 C"] * 3 + ["같음"],
                       ["Top-right menu → Clear cache, or press C"] * 3 + ["Same"])]),
          text=True, index_name=T("데이터", "Data"))
    note(T("장중에 열면 아직 끝나지 않은 당일 봉은 자동으로 제외한다. 화면을 다시 계산만 하려면 키보드 R.",
           "During the session, today's unfinished daily bar is excluded automatically. Press R to rerun only."),
         source=False)

    h2(T("숫자를 읽을 때 주의할 점", "Caveats"))
    p(T("모든 수치는 과거 데이터로 만든 추정이다. 특히 다음을 염두에 둔다.",
        "All figures are estimates from historical data. In particular:"))
    table(help_table(["표본내 편향", "손절 미반영", "재현 가정", "요인 모형 한계", "데이터 지연"],
                     ["In-sample bias", "No stop-losses", "Replay assumption", "Factor model limits", "Data delay"],
                     [("내용", "Detail",
                       ["북을 최근 성적으로 골랐기 때문에 같은 구간으로 잰 정보비율과 평균 수익은 좋게 나온다.",
                        "1장은 북을 끝까지 그대로 들고 간다고 가정한다. 손절이 걸리면 실제 하위 꼬리는 더 얕다.",
                        "4장은 현재 북을 과거 전 기간 들고 있었다고 가정한 재현이다. 북이 실제로 바뀐 이력이 아니다.",
                        "5장 가상 충격의 요인 설명력(R²)이 낮으면 방향 정도만 참고한다. 프리셋 값은 예시다.",
                        "6장 분봉은 yfinance 기준이며 15분 정도 늦을 수 있다."],
                       ["If the book was selected on recent performance, the information ratio and mean return "
                        "measured over the same window look better than they should.",
                        "Chapter 1 assumes the book is held to the end. Stop-losses would make the left tail shallower.",
                        "Chapter 4 replays the current book over the full history; it is not the book's actual record.",
                        "When factor R² is low, read Chapter 5 shocks for direction only. Preset values are examples.",
                        "Chapter 6 bars come from yfinance and may lag by about 15 minutes."])]),
          text=True, index_name=T("항목", "Item"))

    h2(T("문제가 생겼을 때", "Troubleshooting"))
    table(help_table(["가격을 못 받은 종목 경고", "실적 일정을 받지 못했다는 문장", "장중 분봉이 없다는 문장",
                      "북 목록이 비어 있음", "지면 색·글꼴이 기본값으로 보임"],
                     ["Missing-price warning", "Earnings calendar unavailable", "No intraday bars",
                      "Empty book list", "Default colours and fonts"],
                     [("원인과 조치", "Cause and fix",
                       ["티커 철자를 확인한다. 야후 파이낸스 표기(예: 한국 종목은 005930.KS)를 쓴다.",
                        "일시적 차단일 수 있다. 몇 분 뒤 C로 캐시를 지우고 다시 연다. 그동안 해당 종목의 실적 위험은 빠진다.",
                        "장 시작 전·마감 직후·휴장일에는 정상이다.",
                        "BOOK_LOG 환경변수로 북 기록 파일을 지정한다. 없으면 예시 북으로 열린다.",
                        "저장소 폴더 밖에서 실행했다. 저장소 폴더로 이동한 뒤 다시 실행한다."],
                       ["Check the ticker. Use Yahoo Finance symbols (e.g. 005930.KS for Korean listings).",
                        "Likely a temporary block. Wait a few minutes, press C to clear the cache, and reload. Until then "
                        "that holding's earnings risk is excluded.",
                        "Normal before the open, right after the close, and on holidays.",
                        "Point the BOOK_LOG environment variable to a log file. Without it, the sample books load.",
                        "The app was started outside the repository folder. Start it from the repository folder."])]),
          text=True, index_name=T("증상", "Symptom"))
