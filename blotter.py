"""매매 이력(Bloomberg TMSG 내보내기) → 매매 단위별 일간 비중·수익률.

실제로 사고판 포트폴리오의 성과 기여도를 내려면 '고정 비중으로 사서 보유'가 아니라
**날마다 바뀐 실제 비중**이 필요하다. 이 파일이 그걸 만든다. 계산(기여도)은 risk.py 의
*_w 함수가 한다.

── 입력 ─────────────────────────────────────────────────────────────────────
TMSG(Trade Idea) 목록을 Excel/텍스트로 내보낸 것. 열 이름으로 찾는다:
  Idea Status · Ticker · Last Price · Gross Exposure · P&L - OTD · Relative P&L - OTD ·
  Absolute Total Return - OTD · Close Time · Close Price · Open Time · Shares
시각은 **터미널 현지 시각**(기본 Asia/Hong_Kong)이다. 22:30 HKT = 09:30 EST.

── 규약 ─────────────────────────────────────────────────────────────────────
· 한 줄 = 매매 단위(lot). 같은 종목을 두 번 사면 두 줄이고, 열도 따로 둔다.
· 진입가는 내보내기에 없다. 달러 종목은 (기준가 − 손익/주식수), 다른 통화는
  기준가 ÷ (1 + 절대수익률)로 되짚는다. 기준가 = 청산가(청산) 또는 현재가(보유).
· 체결 세션: 현지 시각이 그 거래소 마감 전이면 그날, 뒤면 다음 거래일.
· 날짜 축은 벤치마크(미국) 거래일. 아시아·유럽 종목은 그날까지의 마지막 현지 종가로 본다.
· 일간 손익(달러):  진입일 = 수량·종가/환율 − 원가,  보유일 = 평가액 변화,
                    청산일 = 수량·청산가/환율 − 전일 평가액.
  비중 w_t = 그날 시작 시점 노출 ÷ 전일 순자산, 수익률 r_t = 손익 ÷ 시작 노출.
  그러면 Σ w·r = 그날 포트폴리오 수익률이 정확히 된다.
· 순자산 = 운용 원금(대회 기본 1,000,000달러) + 누적 손익. 현금 = 1 − Σw.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Bloomberg 거래소 코드 → (야후 접미사, 통화, 시간대, 마감 시각, 야후 환율 티커: 1달러당 현지통화)
EXCH = {
    "US": ("", "USD", "America/New_York", dtime(16, 0), None),
    "KS": (".KS", "KRW", "Asia/Seoul", dtime(15, 30), "KRW=X"),
    "KQ": (".KQ", "KRW", "Asia/Seoul", dtime(15, 30), "KRW=X"),
    "TT": (".TW", "TWD", "Asia/Taipei", dtime(13, 30), "TWD=X"),
    "NA": (".AS", "EUR", "Europe/Amsterdam", dtime(17, 30), "EUR=X"),
    "LN": (".L", "GBP", "Europe/London", dtime(16, 30), "GBP=X"),
    "HK": (".HK", "HKD", "Asia/Hong_Kong", dtime(16, 0), "HKD=X"),
    "JT": (".T", "JPY", "Asia/Tokyo", dtime(15, 0), "JPY=X"),
}

_TS = r"\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}"


def yahoo(bbg: str) -> tuple[str, str]:
    """'2330 TT' → ('2330.TW', 'TT'),  'BRK/B US' → ('BRK-B', 'US')."""
    sym, ex = bbg.rsplit(" ", 1)
    if ex not in EXCH:
        raise ValueError(f"모르는 거래소 코드: {bbg}")
    return sym.replace("/", "-") + EXCH[ex][0], ex


def _num(x):
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return float("nan")


def parse_tmsg(text: str) -> pd.DataFrame:
    """TMSG 내보내기 텍스트(탭 또는 공백 여러 칸 구분) → 매매 단위 표.

    빈 칸이 공백으로 뭉개진 붙여넣기도 읽도록 **열 위치가 아니라 모양**으로 자른다:
    상태 · 티커(2단어) · Long/Short · 숫자들 · 시각들 · (청산가) · 수량."""
    rows = []
    pat = re.compile(rf"(Open|Closed|Closing)\s+.*?\s(\S+ [A-Z]{{2}})\s+(Long|Short)\s+(.*?)\s+({_TS})\s*(.*)$")
    for line in text.splitlines():
        m = pat.search(line)
        if not m:
            continue
        status, tk, side, nums, last_upd, rest = m.groups()
        x = nums.split()
        ts = re.findall(_TS, rest)
        tail = re.sub(_TS, " ", rest).split()
        pnl, rel, absr = _num(x[-4]), _num(x[-3]), _num(x[-2])
        if status == "Closed":           # 현재가(=청산가) 손익 상대손익 수익률 상대수익률 | 청산시각 청산가 진입시각 수량
            last, gross = _num(x[0]), float("nan")
            close_ts, open_ts = ts[0], ts[1]
            close_px, shares = _num(tail[0]), _num(tail[-1])
        else:                            # Open/Closing: [현재가] 노출 손익 … | 진입시각 수량
            gross = _num(x[-5])
            last = _num(x[-6]) if len(x) >= 6 else float("nan")
            close_ts, open_ts, close_px = None, ts[0], float("nan")
            shares = _num(tail[-1])
        rows.append(dict(status=status, bbg=tk, side=side, last=last, gross=gross, pnl=pnl, rel_pnl=rel,
                         abs_ret=absr, open_ts=open_ts, close_ts=close_ts, close_px=close_px, shares=shares))
    if not rows:
        raise ValueError("TMSG 줄을 하나도 못 읽었다 — 'Open/Closed  …  티커  Long  …' 형식인지 확인")
    df = pd.DataFrame(rows)
    if (df["side"] != "Long").any():
        raise ValueError("숏 포지션은 아직 지원하지 않는다")
    df[["yf", "ex"]] = df["bbg"].apply(lambda b: pd.Series(yahoo(b)))
    df["ccy"] = df["ex"].map(lambda e: EXCH[e][1])
    if df["last"].isna().any():           # 'Closing' 처럼 현재가가 빈 줄 → 노출 ÷ 수량 (달러 종목만 성립)
        m = df["last"].isna() & (df["ccy"] == "USD")
        df.loc[m, "last"] = df.loc[m, "gross"] / df.loc[m, "shares"]
    ref = df["close_px"].where(df["status"] == "Closed", df["last"])
    usd = df["ccy"] == "USD"
    df["entry_px"] = np.where(usd, ref - df["pnl"] / df["shares"], ref / (1 + df["abs_ret"] / 100))
    df["lot"] = [f"{b.split()[0]}#{i}" for i, b in enumerate(df["bbg"])]
    return df.iloc[::-1].reset_index(drop=True)      # 오래된 매매부터


def session(ts: str, ex: str, days: pd.DatetimeIndex, local_tz: str) -> pd.Timestamp | None:
    """터미널 시각 → 그 거래소의 체결 세션(거래일). 마감 뒤면 다음 거래일."""
    tz, close = EXCH[ex][2], EXCH[ex][3]
    t = pd.Timestamp(ts.replace("/", "-")).tz_localize(local_tz).tz_convert(tz)
    d = pd.Timestamp(t.date())
    if t.time() >= close:
        d = d + pd.Timedelta(days=1)
    after = days[days >= d]
    return after[0] if len(after) else None


@dataclass
class Book:
    W: pd.DataFrame        # 구간 × 매매 단위 비중 (+ '현금')
    R: pd.DataFrame        # 구간 × 매매 단위 수익률(달러)
    pnl: pd.DataFrame      # 구간 × 매매 단위 손익(달러)
    nav: pd.Series         # 구간 끝 순자산
    lots: pd.DataFrame     # 매매 단위 표 (+ 진입·청산 세션, 재구성 손익)
    ticker_of: dict        # 매매 단위 → 야후 티커


def stamps(grid: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """하루를 두 구간으로: 전일 종가→시가(09:30 에 끝), 시가→종가(16:00 에 끝)."""
    return pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in grid for h in (9.5, 16)])


def split_returns(close: pd.Series, open_: pd.Series, grid: pd.DatetimeIndex) -> pd.Series:
    """미국 상장 종목(벤치·ETF)의 두 구간 수익률. 인덱스 = stamps(grid)."""
    c = close.reindex(close.index.union(grid)).ffill()
    o = open_.reindex(grid)
    out = []
    for d in grid:
        prev = c[c.index < d].iloc[-1] if (c.index < d).any() else np.nan
        out += [o[d] / prev - 1, c[d] / o[d] - 1]
    return pd.Series(out, index=stamps(grid)).fillna(0.0)


def _local(ts: str, ex: str, local_tz: str):
    return pd.Timestamp(ts.replace("/", "-")).tz_localize(local_tz).tz_convert(EXCH[ex][2])


def build(lots: pd.DataFrame, close: pd.DataFrame, opens: pd.DataFrame, grid: pd.DatetimeIndex,
          notional: float = 1_000_000, local_tz: str = "Asia/Hong_Kong") -> Book:
    """lots: parse_tmsg 결과. close/opens: 야후 종가·시가(현지통화, close 에 환율 열 포함).
    grid: 날짜 축(미국 거래일).

    🔴 하루를 **두 구간**으로 나눈다. 장 시작에 팔고 같은 날 새로 사면, 하루 단위로는 판 것과 산 것이
    둘 다 '그날 시작 비중'에 잡혀 비중 합이 100%를 크게 넘는다(실제 이력에서 170%대) — 현금이 음수가 되어 배분 효과가
    엉터리가 된다. 시가에서 끊으면 판 매매는 밤사이 구간에만, 산 매매는 장중 구간에만 들어간다.
    · 미국 종목: 밤사이 = 전일 종가→시가, 장중 = 시가→종가. 시가 이전(현지 09:30 이하) 청산은 시가 청산.
      장중 진입은 진입가→종가(그날 시가→진입 시점 움직임은 포트폴리오에 없었다).
    · 아시아·유럽 종목: 미국 장이 열리기 전에 그날 장이 끝나므로 **하루 전체를 밤사이 구간**에 넣는다.
    · 진입 원가(달러) = 기준 평가액 − Bloomberg 손익. 기준 평가액 = 수량 × 청산가(또는 현재가) ÷ 그날 환율.
      비달러 종목은 Bloomberg 수익률의 통화 기준이 불분명해서(현지통화로 되짚으면 유럽 종목 하나가 천 달러 넘게
      어긋났다) 손익 자체를 맞추는 쪽으로 정했다. 그 대신 진입가 검증은 미국 종목으로 한다."""
    lots = lots.copy()
    T = stamps(grid)
    pnl = pd.DataFrame(0.0, index=T, columns=lots["lot"])
    start_exp = pd.DataFrame(0.0, index=T, columns=lots["lot"])
    s_open, s_close, costs = [], [], []
    for _, L in lots.iterrows():
        ex, sh = L["ex"], L["shares"]
        px = close[L["yf"]].dropna()
        op = opens[L["yf"]].reindex(px.index) if L["yf"] in opens else px
        fx_t = EXCH[ex][4]
        fx = close[fx_t].dropna() if fx_t else pd.Series(1.0, index=px.index)
        ffx = lambda d: float(fx[fx.index <= d].iloc[-1]) if (fx.index <= d).any() else float(fx.iloc[0])
        d0 = session(L["open_ts"], ex, px.index, local_tz)
        d1 = session(L["close_ts"], ex, px.index, local_tz) if L["close_ts"] else None
        s_open.append(d0)
        s_close.append(d1)
        # 진입 원가(달러) — Bloomberg 손익과 맞춘다
        if L["status"] == "Closed":
            ref_val = sh * L["close_px"] / ffx(d1)
        else:
            ref_val = sh * L["last"] / ffx(px.index[-1])
        cost = ref_val - L["pnl"]
        costs.append(cost)
        if d0 is None or not (grid >= d0).any():
            continue                                    # 날짜 축 끝 뒤에 들어온 매매
        us = ex == "US"
        exit_at_open = bool(L["close_ts"]) and us and _local(L["close_ts"], ex, local_tz).time() <= dtime(9, 31)
        prev = None                                     # 직전 구간 끝 평가액
        for d in grid[grid >= d0]:
            if d1 is not None and d > d1:
                break
            f = ffx(d)
            last_c = px[px.index <= d]
            c_val = sh * float(last_c.iloc[-1]) / f
            am, pm = d + pd.Timedelta(hours=9.5), d + pd.Timedelta(hours=16)
            exit_val = sh * L["close_px"] / f if (d1 is not None and d == d1) else None
            if not us:                                  # 하루 전체가 밤사이 구간
                begin = cost if prev is None else prev
                end = exit_val if exit_val is not None else c_val
                pnl.at[am, L["lot"]], start_exp.at[am, L["lot"]] = end - begin, begin
                prev = end
                if exit_val is not None:
                    break
                continue
            o_val = sh * float(op.get(d, np.nan)) / f
            if prev is not None:                        # 밤사이: 전일 종가 → 시가 (또는 시가 청산)
                end = exit_val if (exit_val is not None and exit_at_open) else o_val
                pnl.at[am, L["lot"]], start_exp.at[am, L["lot"]] = end - prev, prev
                if exit_val is not None and exit_at_open:
                    break
                begin = o_val
            else:
                begin = cost                            # 오늘 진입 → 장중 구간부터
            end = exit_val if exit_val is not None else c_val
            pnl.at[pm, L["lot"]], start_exp.at[pm, L["lot"]] = end - begin, begin
            prev = end
            if exit_val is not None:
                break
    lots["s_open"], lots["s_close"], lots["cost_usd"] = s_open, s_close, costs
    lots["pnl_rebuilt"] = pnl.sum().reindex(lots["lot"]).to_numpy()
    nav = notional + pnl.sum(axis=1).cumsum()
    nav_prev = nav.shift(1).fillna(notional)
    W = start_exp.div(nav_prev, axis=0)
    R = (pnl / start_exp).where(start_exp > 0, 0.0)
    W["현금"] = 1.0 - W.sum(axis=1)
    return Book(W=W, R=R, pnl=pnl, nav=nav, lots=lots, ticker_of=dict(zip(lots["lot"], lots["yf"])))
