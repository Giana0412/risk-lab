"""리스크 랩 — 모델 북을 실무 리스크 지표로 보는 개인용 대시보드.

실행:  streamlit run app.py

Bloomberg Trading Challenge 기간에 모델 북을 점검하려고 만든 개인 도구.
계산은 risk.py, 데이터는 data.py, 화면 부품은 ui.py.
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

st.set_page_config(page_title="리스크 랩", page_icon="▤", layout="wide")
ui.inject_css()
ui.chapter(0)

PCT = "{:.2%}"


# ── 캐시된 데이터 ───────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner="가격 받는 중 (2007~)…")
def get_daily(tickers: tuple[str, ...]):
    return data.daily_prices(list(tickers))


@st.cache_data(ttl=6 * 3600, show_spinner="실적 일정 받는 중…")
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


@st.cache_data(ttl=3600, show_spinner="VaR 백테스트 중…")
def cached_backtest(ser: pd.Series, alpha: float, window: int):
    return risk.backtest_var(ser, alpha, window)


@st.cache_data(ttl=60, show_spinner=False)
def get_intraday(tickers: tuple[str, ...]):
    return data.intraday_prices(list(tickers))


# ── 사이드바 ────────────────────────────────────────────────────────────

ui.side_head("모델 북 리스크 점검", "리스크 랩")
books = data.load_books()
src = data.book_log_path()

ui.side_section("북")
if books:
    labels = [f"{b['date']} · {b.get('kind', '')}" for b in books]
    pick = st.sidebar.selectbox("모델 북", range(len(books)), format_func=lambda i: labels[i],
                                label_visibility="collapsed")
    base_book = books[pick]["book"]
    if data.using_sample():
        st.sidebar.caption("예시 북이다. 아래 표에서 종목과 비중을 바꿔 자신의 북을 넣을 수 있다.")
else:
    pick = None
    base_book = {"SPY": 0.5, "IWM": 0.5}
    st.sidebar.warning("북 기록을 찾지 못해 기본 예시를 쓴다.")

edited = st.sidebar.data_editor(
    pd.DataFrame({"티커": list(base_book), "비중": [float(v) * 100 for v in base_book.values()]}),
    num_rows="dynamic", hide_index=True, width="stretch",
    key=f"book-{books[pick]['date'] if books else 'demo'}",
    column_config={
        "티커": st.column_config.TextColumn(width="small"),
        "비중": st.column_config.NumberColumn("비중 (%)", format="%.2f", min_value=0.0, max_value=100.0, step=0.01),
    },
)
weights = {str(t).strip().upper(): float(w) / 100 for t, w in zip(edited["티커"], edited["비중"])
           if isinstance(t, str) and t.strip() and pd.notna(w) and w > 0}

wsum = sum(weights.values())
over = [t for t, w in weights.items() if w > 0.20 + 1e-9]
if wsum > 1 + 1e-9:
    st.sidebar.error(f"비중 합 {wsum:.1%} — 레버리지 금지 규정 위반")
if over:
    st.sidebar.error(f"종목당 20% 상한 초과: {', '.join(over)}")
if wsum <= 1 + 1e-9 and not over and weights:
    st.sidebar.caption(f"합계 {wsum:.1%} · 현금 {1 - wsum:.1%} · 규정 충족")

ui.side_section("설정")
end_date = st.sidebar.date_input("평가 종료일", (pd.Timestamp.today() + pd.Timedelta(days=28)).date(),
                                 help="이 날까지 북을 그대로 보유한다고 가정한다 (1장 전망).")
bench = st.sidebar.text_input("벤치마크", "ACWI",
                              help="상대수익의 기준. Bloomberg Trading Challenge 평가 지수(WLS)에 가까운 ACWI가 기본.").upper()
with st.sidebar.expander("고급"):
    lookback = st.select_slider("추정 창 (거래일)", [125, 250, 500, 750], value=250,
                                help="요약·VaR 비교에 쓰는 과거 구간 길이")
    horizon = st.radio("요약·VaR 비교의 보유기간", [1, 5, 10], index=0, horizontal=True,
                       format_func=lambda x: f"{x}일", help="h일 VaR = 1일 × √h (i.i.d. 가정)")
    nav = st.number_input("운용자산 (USD)", value=1_000_000, step=100_000)
    if src:
        st.caption(f"북 출처: `{src}`")

if not weights:
    st.info("사이드바에 종목과 비중을 넣어라.")
    st.stop()

# ── 데이터 준비 ─────────────────────────────────────────────────────────

all_tk = tuple(sorted(set(weights) | {bench} | set(data.FACTOR_TICKERS)))
close, volume = get_daily(all_tk)
missing = [t for t in weights if t not in close or close[t].dropna().empty]
if missing:
    st.warning(f"가격을 못 받은 종목은 빼고 계산한다: {', '.join(missing)}")
    weights = {t: w for t, w in weights.items() if t not in missing}
if bench not in close or not weights:
    st.error("벤치마크 또는 북 가격이 없다.")
    st.stop()

rets_all = close.pct_change(fill_method=None)
hold = list(weights)
cols = list(dict.fromkeys(hold + [bench]))  # 벤치를 북에도 담으면 열이 중복된다
win = rets_all[cols].dropna().iloc[-lookback:]
if len(win) < 60:
    st.error(f"공통 표본이 {len(win)}일뿐이다 — 최근 상장 종목이 있으면 추정 창을 줄여라.")
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

ui.masthead("리스크 랩 · 모델 북 리스크 점검",
            f"{asof:%Y년 %-m월 %-d일} 종가 기준",
            f"벤치마크 {bench} · {len(hold)}종목 · 현금 {1 - sum(weights.values()):.1%} · "
            f"평가 종료까지 {len(remaining)}거래일 · 추정 구간 {len(win)}거래일 · 사용 방법은 부록")
if len(win) < lookback:
    st.caption(f"공통 표본이 {len(win)}일로 요청한 {lookback}일보다 짧다 (최근 상장 종목).")

tab_out, tab_sum, tab_var, tab_bt, tab_st, tab_live, tab_liq, tab_help = st.tabs(
    ["1. 전망", "2. 리스크 개요", "3. VaR 방법론", "4. 모형 검증", "5. 스트레스 테스트", "6. 장중", "7. 유동성",
     "부록. 사용 방법"])

# ════════════════════════════════════════════════════════════════════════
# 1. 전망
# ════════════════════════════════════════════════════════════════════════

with tab_out:
    ui.chapter(1)
    if len(remaining) == 0:
        p("남은 거래일이 없다. 사이드바의 평가 종료일을 확인한다.")
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
                "종목": t, "비중": weights[t],
                "다음 발표": f"{next_ann[t]:%Y-%m-%d %H:%M}" if next_ann.get(t) is not None else "미정",
                "평가 기간": "포함" if in_h else "",
                "평균 |반응|": js.abs().mean() if len(js) else np.nan,
                "최저": js.min() if len(js) else np.nan,
                "최고": js.max() if len(js) else np.nan,
                "표본": len(js),
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

        h2(f"평가 종료까지 {H}거래일, 벤치를 이길 확률 {p_win:.0%}")
        lean = ("벤치에 뒤처지는 쪽으로 기운다" if p_win < 0.45 else
                "어느 쪽으로도 기울지 않는다" if p_win < 0.55 else "벤치를 앞서는 쪽으로 기운다")
        p(f"현재 북을 {remaining[0]:%-m월 %-d일}부터 {remaining[-1]:%-m월 %-d일}까지 그대로 보유한다고 가정하면, "
          f"{bench} 대비 상대수익이 플러스일 확률은 <b>{p_win:.0%}</b>로 추정되며 분포는 {lean}. "
          f"하위 5% 시나리오는 <b>{pp(q05)}</b>, 하위 1% 시나리오는 <b>{pp(q01)}</b>이다. "
          f"운용자산 기준으로 하위 5%는 약 {money(q05 * nav)}에 해당한다.")
        if jumps_h:
            p(f"평가 기간 중 실적 발표가 예정된 종목은 {', '.join(jumps_h)}이며, 해당 종목의 과거 실적일 반응을 "
              f"분포에 반영했다.")
        else:
            p("평가 기간 중 실적 발표가 예정된 보유 종목은 없다.")
        if jumps_1:
            o1 = risk.simulate_outcome(excess_ex, 1, jumps_1)
            p(f"<b>다음 거래일({remaining[0]:%-m월 %-d일})에 {', '.join(jumps_1)}의 실적 반응이 예정되어 있다.</b> "
              f"이 경우 하루 하위 1% 상대손실은 {pp(-risk.var_fhs(excess_ex, 0.99))}에서 "
              f"<b>{pp(-o1.stats()['VaR 99%'])}</b>로 커진다.")
        for t, n in thin:
            p(f"{t}는 평가 기간 중 실적 발표가 있으나 과거 표본이 {n}건에 그쳐 분포에 반영하지 못했다. "
              "실제 위험은 위 수치보다 크다.")
        if earn_failed:
            p("다음 종목은 실적 일정을 받지 못해 실적 위험이 빠져 있다: " +
              ", ".join(f"{t} ({ui.esc(e)})" for t, e in earn_failed.items()) + ".")

        keys([("벤치를 이길 확률", f"{p_win:.0%}", False),
              ("중앙값", pp(s_tot["중앙"]), s_tot["중앙"] < 0),
              ("하위 5%", pp(q05), True),
              ("하위 1%", pp(q01), True)])

        figure(f"남은 {H}거래일 상대수익 분포", f"북 − {bench}, 시뮬레이션 경로 수")
        cnt, edges = np.histogram(out.total, bins=70)
        mids = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(go.Bar(
            x=mids, y=cnt, marker_color=[ui.BAD if m < q05 else ui.BOOK for m in mids], marker_line_width=0,
            width=(edges[1] - edges[0]) * 0.9, customdata=np.stack([edges[:-1], edges[1:]], axis=1),
            hovertemplate="%{customdata[0]:+.1%} ~ %{customdata[1]:+.1%}: %{y:,}<extra></extra>"))
        fig.add_vline(x=0, line_color=ui.INK, line_width=1)
        fig.add_annotation(x=q05, y=1, yref="paper", text=f"하위 5% {pp(q05)}", showarrow=False,
                           xanchor="right", yanchor="top", font=dict(size=11, color=ui.BAD))
        lo, hi = np.quantile(out.total, [0.002, 0.998])
        fig.update_xaxes(range=[lo, hi])
        fig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(ui.layout(fig, 260, "x", legend=False), width="stretch")
        note("평소 거래일은 실적일을 제외한 초과수익에 GARCH(1,1)를 적합하고 표준화 잔차를 복원추출(FHS)해 "
             f"2만 개 경로를 생성했다. 실적 반응은 종목별 과거 실적일 초과수익에서 추출했다. "
             "손절과 리밸런싱은 반영하지 않았다. 붉은 막대는 하위 5% 구간.")

        h2("모형 추정치와 과거 실현치의 차이")
        vols = {"4년": float(ex_long.std()), "1년": float(ex_long.iloc[-250:].std()),
                "3개월": float(ex_long.iloc[-60:].std()), "1개월": float(ex_long.iloc[-20:].std())}
        hot = g.sigma_next > 1.2 * vols["4년"]
        if s_real:
            p(f"같은 북을 지난 {len(ex_long)}거래일 동안 보유했을 때 실제 {H}일 상대수익의 하위 5%는 "
              f"<b>{pp(s_real['5%'])}</b>로, 모형 추정치({pp(q05)})보다 "
              f"{'좁다' if s_real['5%'] > q05 else '넓다'}. "
              + (f"일간 상대 변동성이 4년 평균 {vols['4년']:.2%}에서 최근 1개월 {vols['1개월']:.2%}로 높아진 "
                 f"상태이며, 모형은 이 수준이 점차 낮아진다고 가정한다(다음 거래일 추정 {g.sigma_next:.2%}). "
                 "변동성이 유지되면 모형 쪽이, 평균으로 돌아가면 과거 실현치 쪽이 현실에 가깝다."
                 if hot else "현재 변동성은 평소 수준이어서 두 추정치의 차이는 크지 않다."))
        rows = [("벤치를 이길 확률", "P(상대 > +0%)", "p"), ("+2%p 이상 앞설 확률", "P(상대 > +2%)", "p"),
                ("2%p 넘게 뒤처질 확률", "P(상대 < −2%)", "p"), ("중앙값", "중앙", "v"),
                ("하위 5%", "5%", "v"), ("하위 1%", "VaR 99%", "neg")]
        def cell(sd, k, kind):
            return f"{sd[k]:.0%}" if kind == "p" else pp(-sd[k] if kind == "neg" else sd[k])
        comp = {"모형": [cell(s_tot, k, kd) for _, k, kd in rows]}
        if s_real:
            comp[f"과거 실현 ({len(ex_long)}일)"] = [cell(s_real, k, kd) for _, k, kd in rows]
        if jumps_h:
            comp["모형 (실적 제외)"] = [cell(s_base, k, kd) for _, k, kd in rows]
        table_title(f"{H}거래일 상대수익 분포 비교")
        table(pd.DataFrame(comp, index=[r[0] for r in rows]))
        extra = ""
        if not math.isnan(vr) and (vr < 0.85 or vr > 1.15):
            extra = (f" 분산비 {vr:.2f}: 초과수익이 " +
                     ("되돌림을 보여 모형이 꼬리를 과대평가할 수 있다." if vr < 0.85
                      else "추세를 보여 모형이 꼬리를 과소평가할 수 있다."))
        note("과거 실현은 겹치는 구간으로 계산했으며 실적일을 포함한다." + extra)

        h2("보유 종목 실적 일정")
        jr = pd.DataFrame(jump_rows).set_index("종목").sort_values("평가 기간", ascending=False)
        table_title("실적 발표 일정과 과거 실적일 반응", "반응 = 반응일 종목 수익률 − β × 벤치 수익률")
        table(jr, fmt={"비중": lambda v: f"{v:.1%}", "평균 |반응|": lambda v: pct(v, digits=1),
                       "최저": lambda v: pct(v, True, 1), "최고": lambda v: pct(v, True, 1),
                       "표본": lambda v: f"{int(v)}"},
              neg_cols=("최저",), index_name="종목")
        note("발표 시각이 개장(09:30 ET) 전이면 당일, 이후면 다음 거래일을 반응일로 본다. "
             "시각이 00:00으로 기록된 경우 두 날 중 반응이 큰 날을 택했다. 다음 발표 시각은 미 동부 기준.")

# ════════════════════════════════════════════════════════════════════════
# 2. 리스크 개요
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

    h2("손실 규모와 위험 집중")
    p(f"과거 {len(win)}거래일의 분포를 그대로 적용하면, {horizon}일 기준 99% 신뢰수준에서 북의 손실은 "
      f"<b>{pct(v_abs, digits=1)}</b>({money(v_abs * nav)}), 벤치 대비 손실은 <b>{pp(-v_rel)}</b>를 넘지 않는다. "
      f"그 너머 하위 2.5% 구간의 평균 상대손실(ES)은 {pp(-es_rel)}이다. "
      f"위험 기여가 가장 큰 종목은 {top}로, 비중 {cv.loc[top, '비중']:.0%}에 비해 "
      f"전체 위험의 {cv.loc[top, '기여 비율']:.0%}를 차지한다.")
    keys([(f"VaR 99% · 북", pct(v_abs, digits=1), False), (f"VaR 99% · 벤치 대비", pp(-v_rel), True),
          ("ES 97.5% · 벤치 대비", pp(-es_rel), True), ("최대낙폭", pct(s["최대낙폭"], digits=1), True)])
    note(f"역사적 시뮬레이션, {horizon}일. 최대낙폭은 북 자체 기준.")

    table_title("벤치 대비 특성")
    te = s["추적오차(연)"]
    feat = pd.DataFrame({"값": [num(s["베타(벤치)"]), pct(te, digits=1), num(s["정보비율"]),
                               num(conc["유효 종목 수"], 1), pct(s["연 변동성"], digits=1),
                               num(s["소르티노(rf=0)"]), num(s["칼마"]), num(s["왜도"]), num(s["초과첨도"])],
                         "해석": ["벤치가 1% 움직일 때 북의 평균 반응",
                                f"일간 약 ±{te / math.sqrt(252):.2%} 수준의 이탈",
                                "초과수익 ÷ 추적오차 (표본내, 상향 편향 가능)",
                                f"{len(hold)}종목을 비중으로 환산한 분산 정도 (1/HHI)",
                                "북 자체", "하방 변동성 대비 수익", "연수익 ÷ 최대낙폭",
                                "음수면 큰 손실이 큰 이익보다 잦다", "0이면 정규분포, 클수록 두꺼운 꼬리"]},
                        index=["베타", "추적오차 (연)", "정보비율", "유효 종목 수", "연 변동성",
                               "소르티노", "칼마", "왜도", "초과첨도"])
    table(feat, index_name="지표")
    note("무위험금리는 0으로 두었다.")

    table_title("종목별 위험 기여", "정규 VaR 99%의 오일러 분해. 배수 = 위험 기여 ÷ 비중")
    cv_show = pd.DataFrame({"비중": cv["비중"], "위험 기여": cv["기여 비율"],
                            "배수": cv["기여 비율"] / cv["비중"], "연 변동성": cv["개별 변동성(연)"]})
    table(cv_show, fmt={"비중": lambda v: f"{v:.1%}", "위험 기여": lambda v: f"{v:.1%}",
                        "배수": lambda v: f"{v:.2f}", "연 변동성": lambda v: f"{v:.0%}"},
          index_name="종목", em_rows=(top,))
    note("배수가 1보다 크면 비중에 비해 위험을 많이 낸다. 변동성이 높거나 다른 종목과 함께 움직이는 경우다.")

# ════════════════════════════════════════════════════════════════════════
# 3. VaR 방법론
# ════════════════════════════════════════════════════════════════════════

with tab_var:
    ui.chapter(3)
    rows_v = {}
    for label, series in [("북", port), ("벤치 대비", excess)]:
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
    names = {"역사적": "역사적 시뮬레이션", "정규(분산-공분산)": "분산-공분산 (정규)",
             "코니시-피셔": "코니시-피셔", "EWMA λ=0.94": "EWMA (λ=0.94)", "FHS-GARCH": "FHS-GARCH",
             "몬테카를로 정규": "몬테카를로 (정규)", "몬테카를로 t": "몬테카를로 (t, ν=5)",
             "ES 역사적": "ES · 역사적", "ES 정규": "ES · 정규"}
    var_methods = [m for m in rows_v["북"].index if not m.startswith("ES")]
    rel99 = rows_v["벤치 대비"]["VaR 99%"].reindex(var_methods).dropna()
    lo_m, hi_m = rel99.idxmin(), rel99.idxmax()

    h2("방법에 따라 VaR는 최대 {:.1f}배 달라진다".format(rel99.max() / rel99.min()))
    p(f"동일한 북과 데이터에 대해 벤치 대비 {horizon}일 99% VaR는 {names[lo_m]}에서 <b>{pp(-rel99.min())}</b>, "
      f"{names[hi_m]}에서 <b>{pp(-rel99.max())}</b>로 추정된다. 최근 데이터에 가중을 두는 방법(EWMA, FHS-GARCH)일수록 "
      "값이 크다면 최근 변동성이 표본 평균보다 높다는 뜻이다. 방법 간 적합성은 4장에서 검증한다.")

    figure(f"방법별 {horizon}일 99% VaR", "오른쪽일수록 손실을 크게 추정")
    fig = go.Figure()
    for label, color, sym in [("북", ui.MUTED, "circle-open"), ("벤치 대비", ui.BOOK, "circle")]:
        v = rows_v[label]["VaR 99%"].reindex(var_methods)
        fig.add_scatter(x=v.values, y=[names[m] for m in v.index], mode="markers", name=label,
                        marker=dict(size=10, color=color, symbol=sym, line=dict(color=color, width=1.5)),
                        hovertemplate="%{y}<br>" + label + " %{x:.2%}<extra></extra>")
    fig.update_yaxes(autorange="reversed", showgrid=True)
    fig.update_xaxes(tickformat=".1%")
    st.plotly_chart(ui.layout(fig, 300), width="stretch")
    note("몬테카를로는 개별 종목 공분산으로 계산하므로 북에만 해당한다.")

    table_title(f"방법별 VaR와 ES ({horizon}일)")
    both = pd.concat({k: v.rename(index=names) for k, v in rows_v.items()}, axis=1)
    both.columns = [f"{a_} {b_.replace('VaR ', '')}" for a_, b_ in both.columns]
    table(both, fmt={c: (lambda v: pct(v)) for c in both.columns}, index_name="방법")
    note("손실을 양수로 표기했다.")

    figure("북 변동성: EWMA와 60일 이동평균", "연율화")
    long = risk.portfolio_returns(rets_all[hold].dropna().iloc[-max(lookback, 500):], weights)
    ew = risk.ewma_vol(long) * math.sqrt(252)
    roll = long.rolling(60).std() * math.sqrt(252)
    fig = go.Figure()
    fig.add_scatter(x=roll.index, y=roll, name="60일", line=dict(color=ui.MUTED, width=1.5))
    fig.add_scatter(x=ew.index, y=ew, name="EWMA", line=dict(color=ui.BOOK, width=1.8))
    fig.update_layout(hovermode="x unified")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(ui.layout(fig, 260), width="stretch")
    note("EWMA가 60일선 위에 있으면 최근 변동성이 구간 평균보다 높다.")

# ════════════════════════════════════════════════════════════════════════
# 4. 모형 검증
# ════════════════════════════════════════════════════════════════════════

with tab_bt:
    ui.chapter(4)
    c1, c2, c3 = st.columns(3)
    bt_alpha = c1.radio("신뢰수준", [0.95, 0.99], index=1, horizontal=True, format_func=lambda x: f"{x:.0%}")
    bt_target = c2.radio("대상", ["벤치 대비", "북"], horizontal=True)
    bt_window = c3.select_slider("추정 창 (일)", [125, 250, 500], value=250)
    long_R = rets_all[cols].dropna()
    if len(long_R) < bt_window + 60:
        p(f"공통 표본이 {len(long_R)}일로 검증에 필요한 길이(창 + 60일)에 못 미친다.")
    else:
        lp = risk.portfolio_returns(long_R[hold], weights)
        ser = (lp - long_R[bench]) if bt_target == "벤치 대비" else lp
        ser = ser.iloc[-(bt_window + 750):]
        bt = cached_backtest(ser, bt_alpha, bt_window)
        sm = bt.summary.copy()
        ok = (sm["Kupiec p"] >= 0.05) & (sm["독립성 p"] >= 0.05)
        passed = sm.index[ok].tolist()
        best = (sm["위반"] - sm["기대"]).abs().idxmin()
        n_days = int(sm["표본일"].iloc[0])

        h2("사후검증: " + (f"{len(passed)}개 방법이 두 검정을 통과" if passed else "두 검정을 모두 통과한 방법 없음"))
        p(f"매 거래일 직전 {bt_window}일만으로 다음 날 {bt_alpha:.0%} VaR를 추정하고 실제 손익과 비교했다"
          f"(검증 기간 {n_days}일). 초과 횟수의 적정성(Kupiec)과 초과의 독립성(Christoffersen)을 "
          f"5% 유의수준에서 검정했다. "
          + (f"두 검정을 모두 통과한 방법은 <b>{', '.join(passed)}</b>이다. " if passed else
             "<b>두 검정을 모두 통과한 방법은 없다.</b> ")
          + f"초과 횟수가 기대치에 가장 가까운 방법은 {best}이다.")
        view = pd.DataFrame({
            "초과": sm["위반"].astype(int), "기대": sm["기대"],
            "Kupiec p": sm["Kupiec p"], "독립성 p": sm["독립성 p"],
            "판정": ["적합" if v else "부적합" for v in ok],
            "바젤 구간": [x.split(" ")[-1] if " " in x else x for x in sm["바젤(250일 환산)"]],
        }, index=sm.index)
        table_title(f"VaR 사후검증 ({bt_target}, {bt_alpha:.0%})")
        table(view, fmt={"초과": lambda v: f"{int(v)}", "기대": lambda v: f"{v:.1f}",
                         "Kupiec p": lambda v: f"{v:.3f}", "독립성 p": lambda v: f"{v:.3f}"},
              index_name="방법", em_rows=tuple(passed))
        note("판정은 두 p값이 모두 0.05 이상일 때 적합. 바젤 구간은 250일 환산 초과 횟수 기준(99%에만 적용). "
             "현재 북을 과거 전 기간 보유했다고 가정한 재현이다.")

        m = st.selectbox("그림에 표시할 방법", list(bt.forecasts.columns),
                         index=list(bt.forecasts.columns).index(best))
        hits = bt.realized < -bt.forecasts[m]
        figure(f"일간 수익률과 VaR 한도: {m}", f"초과 {int(hits.sum())}회")
        fig = go.Figure()
        fig.add_scatter(x=bt.realized.index, y=bt.realized, mode="lines", name="일간 수익률",
                        line=dict(color="#b9b7b0", width=0.8))
        fig.add_scatter(x=bt.forecasts.index, y=-bt.forecasts[m], mode="lines", name="VaR 한도",
                        line=dict(color=ui.BOOK, width=1.6))
        fig.add_scatter(x=bt.realized.index[hits], y=bt.realized[hits], mode="markers", name="초과",
                        marker=dict(color=ui.BAD, size=6))
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(ui.layout(fig, 300, "y"), width="stretch")
        note("VaR 한도는 손실 방향(음수)으로 표시했다.")

# ════════════════════════════════════════════════════════════════════════
# 5. 스트레스 테스트
# ════════════════════════════════════════════════════════════════════════

with tab_st:
    ui.chapter(5)
    cpi = risk.stress_days(close, weights, bench, risk.CPI_DAYS, betas_bench)
    hist = risk.stress_replay(close, weights, bench, risk.HISTORICAL_SCENARIOS, betas_bench)

    def short(i: str) -> str:
        return i.split(" (")[0]

    def rel_bars(df: pd.DataFrame, labels: list[str], height: int):
        d = df["상대(북−벤치)"]
        fig = go.Figure(go.Bar(
            y=labels, x=d.values, orientation="h",
            marker_color=[ui.NEG_BAR if v < 0 else ui.POS_BAR for v in d.values],
            text=[pp(v) for v in d.values], textposition="outside", cliponaxis=False,
            textfont=dict(size=11),
            customdata=np.stack([df["북"], df["벤치"]], axis=1),
            hovertemplate="%{y}<br>북 %{customdata[0]:+.1%} · 벤치 %{customdata[1]:+.1%}<extra></extra>"))
        fig.update_yaxes(autorange="reversed", type="category", showgrid=False)
        span = float(np.abs(d).max()) * 1.4
        fig.update_xaxes(range=[-span, span], tickformat=".0%", zeroline=True, zerolinecolor=ui.INK,
                         zerolinewidth=1)
        return ui.layout(fig, height, legend=False)

    def stress_tbl(df: pd.DataFrame, labels: list[str]):
        t = pd.DataFrame({"북": df["북"].values, "벤치": df["벤치"].values, "벤치 대비": df["상대(북−벤치)"].values,
                          "최대 손실 종목": df["최악 기여"].values, "대리 비중": df["대리 비중"].values}, index=labels)
        table(t, fmt={"북": lambda v: pct(v, True, 1), "벤치": lambda v: pct(v, True, 1),
                      "벤치 대비": pp, "대리 비중": lambda v: f"{v:.0%}" if v else "–"},
              neg_cols=("벤치 대비",), index_name="시나리오")

    if not cpi.empty and not hist.empty:
        wc, wh = cpi["상대(북−벤치)"].idxmin(), hist["상대(북−벤치)"].idxmin()
        beat = int((hist["상대(북−벤치)"] > 0).sum())
        h2(f"과거 위기 {len(hist)}건 중 {beat}건에서 벤치보다 선방")
        p(f"현재 비중을 과거 위기 구간에 적용하면 {len(hist)}건 중 {beat}건에서 북의 하락폭이 {bench}보다 작았다. "
          f"가장 불리한 구간은 {short(wh)}로 벤치 대비 <b>{pp(hist.loc[wh, '상대(북−벤치)'])}</b>이다. "
          f"과거 CPI 발표일 하루 기준으로는 {wc.split(' ')[0]}가 가장 불리했다"
          f"({pp(cpi.loc[wc, '상대(북−벤치)'])}).")

    if not cpi.empty:
        lab = [f"{k.split(' ')[0]}  {k.split('(')[1].split('·')[-1].strip(' )')}" for k in cpi.index]
        figure("CPI 발표일의 벤치 대비 수익", "현재 비중을 과거 발표일 하루에 적용")
        st.plotly_chart(rel_bars(cpi, lab, 230), width="stretch")
        table_title("CPI 발표일 재현")
        stress_tbl(cpi, [k.split(" ")[0] for k in cpi.index])
        note("CPI 발표를 앞두고 시나리오를 점검하는 용도다.")

    if not hist.empty:
        figure("과거 위기 구간의 벤치 대비 수익", "구간 누적")
        st.plotly_chart(rel_bars(hist, [short(i) for i in hist.index], 380), width="stretch")
        table_title("과거 위기 재현")
        stress_tbl(hist, [short(i) for i in hist.index])
        proxied = [short(i) for i in hist.index[hist["대리 비중"] > 0.2]]
        note("대리 비중은 당시 상장 전이어서 벤치 베타로 대신한 비중이다."
             + (f" 대리 비중이 20%를 넘는 구간({', '.join(proxied)})은 해석에 유의한다." if proxied else ""))

    h2("가상 충격")
    p("최근 2년 일간 수익률을 시장, 소형주 스프레드, 장기채, 유가, 달러의 다섯 요인에 회귀한 베타로 "
      "충격의 영향을 선형 근사한다. 고유위험은 0으로 둔다.")
    F = data.factor_returns(close).iloc[-500:]
    fb = risk.factor_betas(rets_all[cols].iloc[-500:], F)
    presets = {
        "CPI 상회 (금리 상승, 성장주 약세)": {"시장(ACWI)": -0.025, "소형주 스프레드(IWM−SPY)": -0.01,
                                      "장기채(TLT)": -0.025, "유가(USO)": 0.0, "달러(UUP)": 0.008},
        "CPI 하회 (소형주 강세)": {"시장(ACWI)": 0.015, "소형주 스프레드(IWM−SPY)": 0.025,
                              "장기채(TLT)": 0.02, "유가(USO)": 0.0, "달러(UUP)": -0.006},
        "경기침체 우려": {"시장(ACWI)": -0.08, "소형주 스프레드(IWM−SPY)": -0.03,
                    "장기채(TLT)": 0.05, "유가(USO)": -0.10, "달러(UUP)": 0.02},
        "유가 급등": {"시장(ACWI)": -0.02, "소형주 스프레드(IWM−SPY)": 0.0,
                  "장기채(TLT)": -0.01, "유가(USO)": 0.15, "달러(UUP)": 0.005},
        "직접 입력": None,
    }
    pre = st.selectbox("시나리오", list(presets))
    shocks = {}
    shock_cols = st.columns(len(data.FACTOR_LABELS))
    for col, lab_ in zip(shock_cols, data.FACTOR_LABELS):
        default = presets[pre][lab_] if presets[pre] else 0.0
        shocks[lab_] = col.number_input(f"{lab_} (%)", value=float(default) * 100, step=0.5, format="%.1f",
                                        key=f"shock-{pre}-{lab_}") / 100
    per, book_move, bench_move = risk.factor_shock(fb, weights, bench, shocks)
    keys([("북", pct(book_move, True, 1), book_move < 0), (bench, pct(bench_move, True, 1), bench_move < 0),
          ("벤치 대비", pp(book_move - bench_move), book_move - bench_move < 0)])
    r2 = fb.drop(index=bench, errors="ignore")["R2"]
    note("프리셋 값은 예시 가정이다. 종목별 설명력(R²) 중앙값 " + f"{r2.median():.0%}"
         + (" — 설명력이 낮아 방향 참고용으로만 본다." if r2.median() < 0.3 else "."))
    table_title("요인 베타")
    table(fb, fmt={c: (lambda v: num(v)) for c in fb.columns}, index_name="종목")
    note("최근 500거래일 일간 수익률 OLS.")

# ════════════════════════════════════════════════════════════════════════
# 6. 장중
# ════════════════════════════════════════════════════════════════════════

with tab_live:
    ui.chapter(6)
    refresh = st.toggle("60초마다 갱신", value=False,
                        help="yfinance 분봉은 약 15분 지연될 수 있다. 미 동부 09:30–16:00에만 의미가 있다.")

    @st.fragment(run_every=timedelta(seconds=60) if refresh else None)
    def live():
        ui.chapter(6)  # 조각만 다시 돌 때도 6장 번호로
        tk = tuple(sorted(set(hold) | {bench}))
        ip = get_intraday(tk)
        if ip.empty:
            p("분봉 데이터가 없다. 장 마감 직후나 휴장일에는 정상이다.")
            return
        session = ip.index[-1].date()
        today = ip[ip.index.date == session].ffill()
        prev = close[list(tk)].loc[:pd.Timestamp(session) - pd.Timedelta(days=1)].ffill().iloc[-1]
        bad = [t for t in tk if pd.isna(prev.get(t)) or t not in today or today[t].dropna().empty]
        if bad:
            p(f"전일 종가나 당일 분봉이 없는 종목이 있어 계산하지 않는다: {', '.join(bad)}.")
            return
        today = today.bfill()
        rel = today / prev - 1
        w = pd.Series(weights)
        book_path = (rel[hold] * w).sum(axis=1, min_count=len(hold))
        rel_path = book_path - rel[bench]
        now_book, now_bench = float(book_path.iloc[-1]), float(rel[bench].iloc[-1])
        now_rel = now_book - now_bench
        v95, v99 = risk.var_historical(excess, 0.95), risk.var_historical(excess, 0.99)
        level = ("99% VaR를 넘었다" if now_rel < -v99 else "95% VaR를 넘었다" if now_rel < -v95
                 else "95% VaR 이내다")
        h2(f"{session:%-m월 %-d일} {ip.index[-1]:%H:%M} ET 기준 벤치 대비 {pp(now_rel)}")
        p(f"북은 전일 종가 대비 {pct(now_book, True)}, {bench}는 {pct(now_bench, True)}로 "
          f"벤치 대비 {pp(now_rel)}({money(now_rel * nav)})이다. 일간 상대 손실 기준으로 {level}"
          f"(95% {pp(-v95)}, 99% {pp(-v99)}).")
        figure("장중 벤치 대비 수익", "전일 종가 대비, 5분봉")
        fig = go.Figure()
        fig.add_hline(y=0, line_color=ui.INK, line_width=1)
        for v, lab_, colr in [(v95, "VaR 95%", ui.MUTED), (v99, "VaR 99%", ui.BAD)]:
            fig.add_hline(y=-v, line_dash="dot", line_color=colr, line_width=1,
                          annotation_text=lab_, annotation_position="bottom right",
                          annotation_font=dict(color=colr, size=11))
        fig.add_scatter(x=rel_path.index, y=rel_path, line=dict(color=ui.BOOK, width=1.8),
                        hovertemplate="%{x|%H:%M} %{y:+.2%}<extra></extra>")
        fig.update_yaxes(range=[min(float(rel_path.min()), -v99) * 1.2, max(float(rel_path.max()), v95) * 1.2])
        st.plotly_chart(ui.layout(fig, 280, "y", legend=False), width="stretch")
        note("yfinance 분봉 기준이며 지연될 수 있다.")
        tbl = pd.DataFrame({"당일": rel[hold].iloc[-1], "비중": w, "기여": rel[hold].iloc[-1] * w}).sort_values("기여")
        table_title("종목별 당일 기여")
        table(tbl, fmt={"당일": lambda v: pct(v, True), "비중": lambda v: f"{v:.1%}", "기여": pp},
              neg_cols=("당일", "기여"), index_name="종목")

    live()

# ════════════════════════════════════════════════════════════════════════
# 7. 유동성
# ════════════════════════════════════════════════════════════════════════

with tab_liq:
    ui.chapter(7)
    part = st.slider("체결 가능 비율 (일평균 거래대금 대비)", 5, 30, 20, 5, format="%d%%") / 100
    adv = volume[hold].iloc[-20:].mean()
    lq = risk.liquidity(weights, close[hold].iloc[-1], adv, nav, part)
    worst = lq["청산일"].idxmax()
    h2(f"전 종목 청산에 최대 {lq.loc[worst, '청산일']:.3f}일")
    p(f"일평균 거래대금의 {part:.0%}까지 체결할 수 있다고 가정하면, 청산에 가장 오래 걸리는 {worst}도 "
      f"{lq.loc[worst, '청산일']:.3f}일이면 정리된다. "
      + ("현재 운용 규모에서 유동성 위험은 무시할 수준이다." if lq["청산일"].max() < 0.5
         else "하루 이상 걸리는 종목이 있어 청산 계획이 필요하다."))
    table_title("종목별 청산 소요")
    table(pd.DataFrame({"보유 금액": lq["보유($)"], "일평균 거래대금": lq["ADV($)"],
                        "거래대금 대비": lq["ADV 대비"], "청산 소요 (일)": lq["청산일"]}),
          fmt={"보유 금액": money, "일평균 거래대금": lambda v: f"{money(v / 1e6)}M" if not math.isnan(v) else "–",
               "거래대금 대비": lambda v: f"{v:.3%}", "청산 소요 (일)": lambda v: f"{v:.3f}"},
          index_name="종목")
    note("일평균 거래대금은 최근 20거래일 평균. 체결 비용과 슬리피지는 반영하지 않았다.")

# ════════════════════════════════════════════════════════════════════════
# 부록. 사용 방법
# ════════════════════════════════════════════════════════════════════════

with tab_help:
    ui.chapter(0)
    h2("실행")
    p("저장소 폴더에서 <code>pip install -r requirements.txt</code> 후 <code>streamlit run app.py</code>를 "
      "실행하고 브라우저에서 localhost:8501을 연다. 저장소 폴더 안에서 실행해야 지면 색과 글꼴 설정"
      "(.streamlit/config.toml)이 적용된다. 처음 열 때 2007년부터의 가격을 받느라 20초 안팎이 걸리고, "
      "이후에는 캐시를 쓴다.")

    h2("북 고르기와 고치기")
    p("왼쪽 사이드바의 <b>북</b> 목록은 북 기록 파일에서 읽는다. 환경변수 <code>BOOK_LOG</code>가 가리키는 JSONL "
      "파일이 있으면 그것을, 없으면 저장소의 예시 북(sample_books.jsonl)을 쓴다. 날짜를 고르면 그날 북의 종목과 "
      "비중이 아래 표에 채워진다. 표는 직접 고칠 수 있다.")
    table(pd.DataFrame({"방법": [
        "비중 칸을 두 번 눌러 숫자를 입력한다. 단위는 %다.",
        "표 맨 아래 빈 줄에 티커와 비중을 넣는다. 야후 파이낸스 티커를 쓴다.",
        "줄 왼쪽 칸을 눌러 선택한 뒤 Delete 키를 누른다.",
        "다른 날짜를 골랐다가 다시 고르거나 페이지를 새로 고친다."]},
        index=["비중 바꾸기", "종목 추가", "종목 삭제", "원래 북으로 되돌리기"]), text=True, index_name="할 일")
    note("표 아래 줄에 비중 합계와 현금이 표시된다. 합계가 100%를 넘거나(레버리지 금지) 한 종목이 20%를 넘으면 "
         "운용 제약(레버리지 금지, 종목당 20%) 위반 경고가 뜬다. 고친 비중은 화면에서만 쓰이고 기록 파일은 바뀌지 않는다.", source=False)

    h2("설정")
    table(pd.DataFrame({"뜻": [
        "1장 전망의 시뮬레이션 기간을 정한다. 기본값은 오늘부터 4주 뒤.",
        "상대수익을 잴 기준. Bloomberg Trading Challenge 평가 지수(WLS)에 가까운 ACWI가 기본이다.",
        "2·3장 계산에 쓰는 과거 구간 길이. 짧을수록 최근 상황을, 길수록 평균적인 상황을 반영한다.",
        "2·3장 VaR의 기간. 1일 외에는 √기간을 곱한다(3장 FHS-GARCH는 경로 시뮬레이션).",
        "금액 환산에 쓰는 규모. 기본값은 100만 달러."]},
        index=["평가 종료일", "벤치마크", "추정 창 (고급)", "보유기간 (고급)", "운용자산 (고급)"]),
        text=True, index_name="항목")

    h2("장별로 무엇을 보나")
    table(pd.DataFrame({
        "답하는 질문": ["남은 기간에 벤치를 이길 확률과 나쁠 때의 폭", "하루 손실 규모와 위험을 많이 내는 종목",
                     "VaR가 계산 방법에 따라 얼마나 달라지나", "어떤 VaR 방법이 과거에 실제로 맞았나",
                     "CPI 발표일이나 과거 위기가 다시 오면", "지금 장중에 벤치 대비 어디쯤인가",
                     "한 번에 다 팔 수 있나"],
        "먼저 볼 것": ["제목의 확률, 하위 5%·1%", "종목별 위험 기여 표의 배수", "그림 3-1의 점 간격",
                    "표 4-1의 판정 열", "그림 5-1, 5-2의 붉은 막대", "제목의 벤치 대비 수치와 VaR 선",
                    "청산 소요 열"]},
        index=["1. 전망", "2. 리스크 개요", "3. VaR 방법론", "4. 모형 검증", "5. 스트레스 테스트",
               "6. 장중", "7. 유동성"]), text=True, index_name="장")

    h2("쓰는 순서")
    p("<b>매일 아침.</b> 사이드바에서 가장 최근 날짜의 북을 고르고 1장을 본다. 벤치를 이길 확률과 하위 5%가 "
      "전날보다 크게 나빠졌는지, 실적 일정 표에 '평가 기간: 포함'인 종목이 생겼는지 확인한다.")
    p("<b>큰 발표 전날.</b> CPI·FOMC 같은 일정 전에는 5장의 CPI 발표일 재현과 가상 충격을 본다. 특정 시나리오에서 "
      "벤치 대비 손실이 큰 종목이 있으면 2장의 위험 기여 표에서 그 종목의 배수를 함께 본다.")
    p("<b>장중.</b> 6장에서 '60초마다 갱신'을 켜 둔다. 벤치 대비 선이 붉은 점선(VaR 99%) 아래로 내려가면 "
      "추정 구간 기준으로 100거래일에 한 번 나올 손실이다.")
    p("<b>비중을 바꿔 보고 싶을 때.</b> 사이드바 표에서 비중을 고치면 모든 장이 다시 계산된다. 실제 주문 전에 "
      "1장의 확률과 하위 5%가 어떻게 움직이는지 비교하는 용도로 쓴다.")

    h2("새 데이터 받기")
    table(pd.DataFrame({"보관 시간": ["1시간", "6시간", "60초", "1시간"],
                        "강제로 새로 받으려면": ["오른쪽 위 메뉴 → Clear cache, 또는 키보드 C"] * 3 + ["같음"]},
                       index=["일봉 가격", "실적 일정", "장중 분봉", "모형 검증 결과"]), text=True, index_name="데이터")
    note("장중에 열면 아직 끝나지 않은 당일 봉은 자동으로 제외한다. 화면을 다시 계산만 하려면 키보드 R.", source=False)

    h2("숫자를 읽을 때 주의할 점")
    p("모든 수치는 과거 데이터로 만든 추정이다. 특히 다음을 염두에 둔다.")
    table(pd.DataFrame({"내용": [
        "북을 최근 성적으로 골랐기 때문에 같은 구간으로 잰 정보비율과 평균 수익은 좋게 나온다.",
        "1장은 북을 끝까지 그대로 들고 간다고 가정한다. 손절이 걸리면 실제 하위 꼬리는 더 얕다.",
        "4장은 현재 북을 과거 전 기간 들고 있었다고 가정한 재현이다. 북이 실제로 바뀐 이력이 아니다.",
        "5장 가상 충격의 요인 설명력(R²)이 낮으면 방향 정도만 참고한다. 프리셋 값은 예시다.",
        "6장 분봉은 yfinance 기준이며 15분 정도 늦을 수 있다."]},
        index=["표본내 편향", "손절 미반영", "재현 가정", "요인 모형 한계", "데이터 지연"]), text=True, index_name="항목")

    h2("문제가 생겼을 때")
    table(pd.DataFrame({"원인과 조치": [
        "티커 철자를 확인한다. 야후 파이낸스 표기(예: 한국 종목은 005930.KS)를 쓴다.",
        "일시적 차단일 수 있다. 몇 분 뒤 C로 캐시를 지우고 다시 연다. 그동안 해당 종목의 실적 위험은 빠진다.",
        "장 시작 전·마감 직후·휴장일에는 정상이다.",
        "BOOK_LOG 환경변수로 북 기록 파일을 지정한다. 없으면 예시 북으로 열린다.",
        "저장소 폴더 밖에서 실행했다. 저장소 폴더로 이동한 뒤 다시 실행한다."]},
        index=["가격을 못 받은 종목 경고", "실적 일정을 받지 못했다는 문장", "장중 분봉이 없다는 문장",
               "북 목록이 비어 있음", "지면 색·글꼴이 기본값으로 보임"]), text=True, index_name="증상")
