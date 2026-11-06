"""화면 부품 — 리서치 리포트 형식. 계산은 없다.

형식: 명조 제목, 산세리프 본문, 좁은 본문 폭. 강조는 굵기와 괘선으로만 하고
상자·배경색·이모지는 쓰지 않는다. 차트와 표는 '그림 n.' / '표 n.' 번호를 달고
아래에 주와 자료 출처를 붙인다.
"""
from __future__ import annotations

import html
import math
from typing import Callable

import pandas as pd
import streamlit as st

INK = "#1d1d1b"
INK2 = "#5c5b57"
RULE = "#1d1d1b"
HAIR = "#d9d7d0"
BOOK = "#1f4e8c"      # 북 — 진한 남색
BENCH = "#b8862b"     # 벤치 — 황토
MUTED = "#9a9993"
BAD = "#b0302a"       # 하락 꼬리·손실
NEG_BAR = "#b0302a"
POS_BAR = "#1f4e8c"
SOURCE = "자료: yfinance, 자체 계산"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@500;700&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.min.css');

html, body, .stApp, .stMarkdown, p, li, label, input, textarea, button {
  font-family: 'Pretendard', -apple-system, 'Apple SD Gothic Neo', sans-serif;
}
[data-testid="stIconMaterial"], .material-symbols-rounded { font-family: 'Material Symbols Rounded' !important; }
.stApp { background: #fbfaf7; }
[data-testid="stSidebar"] { background: #f2f0ea; border-right: 1px solid #d9d7d0; }
.block-container { max-width: 920px; padding-top: 2.4rem; padding-bottom: 5rem; }
header[data-testid="stHeader"] { background: transparent; }

/* 탭 → 목차처럼 */
.stTabs [data-baseweb="tab-list"] { gap: 22px; border-bottom: 1px solid #1d1d1b; }
.stTabs [data-baseweb="tab"] { padding: 6px 0; }
.stTabs [data-baseweb="tab"] p { font-size: .92rem; font-weight: 500; color: #5c5b57; }
.stTabs [aria-selected="true"] p { color: #1d1d1b; font-weight: 700; }
.stTabs [data-baseweb="tab-highlight"] { background-color: #1d1d1b; height: 2px; }

.rp-kicker { font-size: .78rem; letter-spacing: .08em; color: #5c5b57; text-transform: uppercase;
  border-top: 3px solid #1d1d1b; padding-top: 8px; }
.rp-title { font-family: 'Noto Serif KR', serif; font-weight: 700; font-size: 2.05rem; line-height: 1.25;
  color: #1d1d1b; margin: 6px 0 8px; letter-spacing: -0.01em; }
.rp-byline { font-size: .86rem; color: #5c5b57; padding-bottom: 14px; }

.rp-h2 { font-family: 'Noto Serif KR', serif; font-weight: 700; font-size: 1.32rem; color: #1d1d1b;
  margin: 26px 0 8px; }
.rp-p { font-size: 1.0rem; line-height: 1.75; color: #1d1d1b; margin: 0 0 12px; max-width: 46em; }
.rp-p b { font-weight: 700; }
.rp-num { font-variant-numeric: tabular-nums; }

.rp-fig { margin: 22px 0 4px; }
.rp-fig .t { font-size: .92rem; font-weight: 700; color: #1d1d1b; }
.rp-fig .s { font-size: .84rem; color: #5c5b57; margin-top: 2px; }
.rp-note { font-size: .78rem; color: #5c5b57; line-height: 1.55; border-top: 1px solid #d9d7d0;
  padding-top: 6px; margin: 2px 0 18px; }

table.rp { border-collapse: collapse; width: 100%; font-size: .88rem; font-variant-numeric: tabular-nums;
  margin: 6px 0 2px; color: #1d1d1b; }
table.rp thead th { font-weight: 600; text-align: right; padding: 6px 8px; border-top: 1.5px solid #1d1d1b;
  border-bottom: 1px solid #1d1d1b; color: #1d1d1b; vertical-align: bottom; }
table.rp thead th:first-child, table.rp td:first-child { text-align: left; }
table.rp td { padding: 5px 8px; text-align: right; border-bottom: 1px solid #ebe9e3; }
table.rp tbody tr:last-child td { border-bottom: 1.5px solid #1d1d1b; }
table.rp td.neg { color: #b0302a; }
table.rp tr.em td { font-weight: 700; }
table.rp.txt th, table.rp.txt td { text-align: left; line-height: 1.55; }
table.rp.txt td:first-child { white-space: nowrap; padding-right: 18px; }
.rp-p code { background: none; color: #1d1d1b; font-size: .9em; padding: 0 2px; border-bottom: 1px solid #d9d7d0; }

/* 핵심 수치: 2열 정의 목록 */
.rp-keys { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); border-top: 1.5px solid #1d1d1b;
  border-bottom: 1px solid #1d1d1b; margin: 10px 0 4px; }
.rp-keys > div { padding: 10px 12px 10px 0; }
.rp-keys > div + div { border-left: 1px solid #d9d7d0; padding-left: 12px; }
.rp-keys .k { font-size: .78rem; color: #5c5b57; }
.rp-keys .v { font-family: 'Noto Serif KR', serif; font-size: 1.55rem; font-weight: 700; color: #1d1d1b;
  font-variant-numeric: tabular-nums; margin-top: 2px; }
.rp-keys .v.neg { color: #b0302a; }
@media (max-width: 700px) { .rp-keys { grid-template-columns: repeat(2, minmax(0,1fr)) !important; } }

/* ── 사이드바: 본문과 같은 지면 — 채운 상자 대신 밑줄 입력, 괘선 구획 ── */
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1.6rem; }
.sb-head { border-top: 3px solid #1d1d1b; padding-top: 8px; margin-bottom: 18px; }
.sb-head .k { font-size: .7rem; letter-spacing: .08em; color: #5c5b57; }
.sb-head .t { font-family: 'Noto Serif KR', serif; font-weight: 700; font-size: 1.35rem; color: #1d1d1b; }
.sb-sec { font-size: .72rem; font-weight: 700; letter-spacing: .08em; color: #1d1d1b;
  border-bottom: 1px solid #1d1d1b; padding-bottom: 4px; margin: 18px 0 8px; }
[data-testid="stSidebar"] label p { font-size: .8rem; color: #5c5b57; font-weight: 500; }
[data-testid="stSidebar"] [data-baseweb="input"],
[data-testid="stSidebar"] [data-baseweb="base-input"],
[data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: transparent !important; border: none !important; border-bottom: 1px solid #9a9993 !important;
  border-radius: 0 !important; box-shadow: none !important;
}
[data-testid="stSidebar"] [data-baseweb="input"]:focus-within,
[data-testid="stSidebar"] [data-baseweb="select"] > div:focus-within { border-bottom: 1.5px solid #1d1d1b !important; }
[data-testid="stSidebar"] input { background: transparent !important; padding-left: 0 !important;
  font-variant-numeric: tabular-nums; }
[data-testid="stSidebar"] [data-baseweb="select"] > div > div { padding-left: 0 !important; }
[data-testid="stSidebar"] [data-testid="stNumberInput"] button { background: transparent !important; border: none !important; }
[data-testid="stSidebar"] [data-testid="stExpander"] details {
  border: none !important; border-top: 1px solid #d9d7d0 !important; border-radius: 0 !important; background: transparent; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary { padding-left: 0; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary p { font-size: .8rem; color: #5c5b57; }
[data-testid="stSidebar"] [data-testid="stDataFrame"], [data-testid="stSidebar"] [data-testid="stDataFrameResizable"] {
  border-top: 1.5px solid #1d1d1b; border-bottom: 1px solid #1d1d1b; border-radius: 0; }
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { font-size: .76rem; color: #5c5b57; }
[data-testid="stSidebar"] [data-testid="stAlert"] { border-radius: 0; }
[data-testid="stSidebar"] [data-testid="stExpander"] summary,
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover { background: transparent !important; }
[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] {
  border: none !important; border-radius: 0 !important; background: transparent !important; }
[data-testid="stSidebar"] code { background: none !important; color: #5c5b57 !important; padding: 0 !important;
  font-size: .72rem; word-break: break-all; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def esc(s) -> str:
    """HTML 이스케이프 + '$' 엔티티화 — 마크다운이 $…$ 를 수식으로 바꾸지 않게."""
    return html.escape(str(s)).replace("$", "&#36;")


def _nan(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def money(x: float) -> str:
    if _nan(x):
        return "–"
    return f"{'−' if x < 0 else ''}&#36;{abs(x):,.0f}"


def pct(x: float, signed: bool = False, digits: int = 2) -> str:
    if _nan(x):
        return "–"
    return (f"{x:+.{digits}%}" if signed else f"{x:.{digits}%}").replace("-", "−")


def pp(x: float, digits: int = 1) -> str:
    if _nan(x):
        return "–"
    return f"{x * 100:+.{digits}f}%p".replace("-", "−")


def num(x: float, digits: int = 2) -> str:
    return "–" if _nan(x) else f"{x:.{digits}f}".replace("-", "−")


# ── 블록 ────────────────────────────────────────────────────────────────

def masthead(kicker: str, title: str, byline: str):
    st.markdown(f'<div class="rp-kicker">{esc(kicker)}</div><div class="rp-title">{esc(title)}</div>'
                f'<div class="rp-byline">{esc(byline)}</div>', unsafe_allow_html=True)


def h2(text: str):
    st.markdown(f'<div class="rp-h2">{esc(text)}</div>', unsafe_allow_html=True)


def p(html_text: str):
    """본문 문단. 인자는 이미 HTML 이다 (숫자는 호출부에서 <b> 로 감싼다)."""
    st.markdown(f'<p class="rp-p">{html_text}</p>', unsafe_allow_html=True)


def keys(items: list[tuple[str, str, bool]]):
    """핵심 수치 줄: (라벨, 값, 음수강조)."""
    cells = "".join(f'<div><div class="k">{esc(k)}</div><div class="v{" neg" if neg else ""}">{v}</div></div>'
                    for k, v, neg in items)
    st.markdown(f'<div class="rp-keys" style="grid-template-columns:repeat({len(items)},minmax(0,1fr))">'
                f'{cells}</div>', unsafe_allow_html=True)


class Counter:
    """그림·표 번호는 장별로 매긴다 (그림 4-1). Streamlit 은 탭을 한 번에 다 그리므로
    전역 번호를 쓰면 4장 첫 표가 '표 6' 이 된다 — 장이 바뀔 때 0 으로 돌린다."""
    def __init__(self):
        self.chap = 0
        self.fig = 0
        self.tab = 0


N = Counter()


def chapter(n: int):
    N.chap, N.fig, N.tab = n, 0, 0


def _label(kind: str, k: int) -> str:
    return f"{kind} {N.chap}-{k}." if N.chap else f"{kind} {k}."


def figure(title: str, sub: str = ""):
    N.fig += 1
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="rp-fig"><div class="t">{_label("그림", N.fig)} {esc(title)}</div>{s}</div>',
                unsafe_allow_html=True)


def table_title(title: str, sub: str = ""):
    N.tab += 1
    s = f'<div class="s">{esc(sub)}</div>' if sub else ""
    st.markdown(f'<div class="rp-fig"><div class="t">{_label("표", N.tab)} {esc(title)}</div>{s}</div>',
                unsafe_allow_html=True)


def note(text: str, source: bool = True):
    src = f"<br>{esc(SOURCE)}" if source else ""
    st.markdown(f'<div class="rp-note">주: {esc(text)}{src}</div>', unsafe_allow_html=True)


def table(df: pd.DataFrame, fmt: dict[str, Callable] | None = None, neg_cols: tuple = (),
          index_name: str | None = None, em_rows: tuple = (), text: bool = False):
    """괘선만 있는 정적 표. fmt: 열 → 값을 문자열로 바꾸는 함수. neg_cols 의 음수는 붉게."""
    fmt = fmt or {}
    head = f"<th>{esc(index_name if index_name is not None else (df.index.name or ''))}</th>" + \
        "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for idx, row in df.iterrows():
        tds = [f"<td>{esc(idx)}</td>"]
        for c in df.columns:
            v = row[c]
            s = fmt[c](v) if c in fmt else ("–" if _nan(v) else esc(v))
            cls = ' class="neg"' if c in neg_cols and isinstance(v, (int, float)) and not _nan(v) and v < 0 else ""
            tds.append(f"<td{cls}>{s}</td>")
        em = ' class="em"' if idx in em_rows else ""
        body.append(f"<tr{em}>{''.join(tds)}</tr>")
    st.markdown(f'<table class="rp{" txt" if text else ""}"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>',
                unsafe_allow_html=True)


def layout(fig, height: int = 300, pct_axis: str | None = None, legend: bool = True):
    """차트: 흰 바탕, 가로 격자만, 축선 검정, 범례는 위 왼쪽."""
    fig.update_layout(
        height=height, margin=dict(t=28 if legend else 8, b=8, l=8, r=8),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Pretendard, -apple-system, sans-serif", size=12, color=INK),
        legend=dict(orientation="h", y=1.1, x=0, title=None, font=dict(size=12)) if legend else None,
        showlegend=legend, hoverlabel=dict(font_size=12, bgcolor="white"),
        bargap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor=INK, linewidth=1, ticks="outside", tickcolor=INK, ticklen=3,
                     zeroline=False)
    fig.update_yaxes(gridcolor="#e6e4dd", linecolor=INK, linewidth=0, zeroline=False)
    if pct_axis == "x":
        fig.update_xaxes(tickformat=".0%")
    elif pct_axis == "y":
        fig.update_yaxes(tickformat=".1%")
    return fig


def side_head(kicker: str, title: str):
    st.sidebar.markdown(f'<div class="sb-head"><div class="k">{esc(kicker)}</div>'
                        f'<div class="t">{esc(title)}</div></div>', unsafe_allow_html=True)


def side_section(text: str):
    st.sidebar.markdown(f'<div class="sb-sec">{esc(text)}</div>', unsafe_allow_html=True)
