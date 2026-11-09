"""데이터 — 북과 가격(yfinance). 계산은 risk.py 에 있다."""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import yfinance as yf

# 북 기록(JSONL, 한 줄에 {"date", "kind", "book": {티커: 비중}}).
# BOOK_LOG 환경변수가 가리키는 파일이 있으면 그것을, 없으면 저장소의 예시 북을 쓴다.
SAMPLE_BOOKS = Path(__file__).resolve().parent / "sample_books.jsonl"

# 팩터 프록시 — 전부 거래 가능한 ETF 라 yfinance 로 받는다.
# 크기 팩터는 IWM−SPY 수익률 차(롱숏 스프레드)로 만든다.
FACTOR_TICKERS = ["ACWI", "SPY", "IWM", "TLT", "USO", "UUP"]
FACTOR_LABELS = {
    "시장(ACWI)": "ACWI",
    "소형주 스프레드(IWM−SPY)": None,
    "장기채(TLT)": "TLT",
    "유가(USO)": "USO",
    "달러(UUP)": "UUP",
}


# NYSE 휴장·조기폐장 (pandas 의 미국 연방 휴일 달력은 NYSE 와 다르다 — 성금요일 포함,
# 콜럼버스·재향군인의 날 제외). 출처: NYSE 공지. 2027 년 이후는 손으로 늘려야 한다.
NYSE_HOLIDAYS = pd.to_datetime([
    "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25",
    "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26", "2027-05-31", "2027-06-18",
    "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24",
])
NYSE_EARLY_CLOSE = {pd.Timestamp("2026-11-27"): (13, 0), pd.Timestamp("2026-12-24"): (13, 0)}


def trading_days(start, end) -> pd.DatetimeIndex:
    return pd.bdate_range(start, end, freq="C", holidays=list(NYSE_HOLIDAYS))


def book_log_path() -> Path | None:
    env = os.environ.get("BOOK_LOG")
    if env and Path(env).expanduser().exists():
        return Path(env).expanduser()
    return SAMPLE_BOOKS if SAMPLE_BOOKS.exists() else None


def using_sample() -> bool:
    return book_log_path() == SAMPLE_BOOKS


def load_books() -> list[dict]:
    """북 기록, 최신이 앞."""
    p = book_log_path()
    if not p:
        return []
    out = []
    for line in p.read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("book"):
            out.append(r)
    return sorted(out, key=lambda r: r.get("date", ""), reverse=True)


def _close_volume(df: pd.DataFrame, tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    if isinstance(df.columns, pd.MultiIndex):
        close, vol = df["Close"], df["Volume"]
    else:  # 티커 하나일 때 yfinance 는 단층 컬럼을 준다
        close = df[["Close"]].rename(columns={"Close": tickers[0]})
        vol = df[["Volume"]].rename(columns={"Volume": tickers[0]})
    return close.sort_index(), vol.sort_index()


def daily_prices(tickers: list[str], start: str = "2007-06-01") -> tuple[pd.DataFrame, pd.DataFrame]:
    """수정종가·거래량. 스트레스 시나리오(2008~)를 재생하려고 길게 받는다."""
    tk = sorted(set(tickers))
    df = yf.download(tk, start=start, auto_adjust=True, progress=False, threads=True)
    close, vol = _close_volume(df, tk)
    close.index = pd.to_datetime(close.index).tz_localize(None)
    vol.index = close.index
    # 장중에 받으면 yfinance 가 '아직 안 끝난 오늘'을 일봉 한 줄로 끼워 준다. 그 줄을 종가로
    # 쓰면 VaR 표본에 반쪽짜리 하루가 들어가고 실시간 탭의 '전일 종가'도 어긋난다.
    now = pd.Timestamp.now(tz="America/New_York")
    ch, cm = NYSE_EARLY_CLOSE.get(pd.Timestamp(now.date()), (16, 0))
    if len(close) and close.index[-1].date() == now.date() and (now.hour, now.minute) < (ch, cm + 15):
        close, vol = close.iloc[:-1], vol.iloc[:-1]
    return close, vol


def earnings_dates_one(t: str, limit: int = 24) -> list[pd.Timestamp]:
    """한 종목의 실적 발표 시각 (과거 ~6년 + 예정). 실패하면 **예외를 그대로 올린다** —
    빈 목록으로 삼키면 '실적 없음'과 '못 받음'이 구분되지 않아 점프 층이 조용히 꺼진다.
    ETF 처럼 정말로 실적이 없는 종목은 빈 목록이다."""
    df = yf.Ticker(t).get_earnings_dates(limit=limit)
    return sorted(pd.Timestamp(x) for x in df.index) if df is not None else []


def intraday_prices(tickers: list[str], interval: str = "5m") -> pd.DataFrame:
    """최근 2세션 분봉 — 오늘 세션만 잘라 쓰고, 전일 종가는 일봉에서 가져온다."""
    tk = sorted(set(tickers))
    df = yf.download(tk, period="2d", interval=interval, auto_adjust=True,
                     progress=False, threads=True, prepost=False)
    if df.empty:
        return pd.DataFrame()
    close, _ = _close_volume(df, tk)
    close.index = pd.to_datetime(close.index).tz_convert("America/New_York")
    return close


def factor_returns(close: pd.DataFrame) -> pd.DataFrame:
    # 자기 거래일 기준 수익률 — 표에 환율 같은 다른 달력의 열이 섞여 있으면 휴장 다음 날이 NaN 이 된다
    r = close[FACTOR_TICKERS].apply(lambda s: s.dropna().pct_change(fill_method=None)).reindex(close.index)
    f = pd.DataFrame(index=r.index)
    for label, t in FACTOR_LABELS.items():
        f[label] = r["IWM"] - r["SPY"] if t is None else r[t]
    return f


# ── 성과 기여도용: 섹터 ─────────────────────────────────────────────────
# 종목 섹터는 Yahoo 분류(Morningstar 기반)를 쓴다. 섹터 ETF 는 GICS 를 따르므로 경계가
# 약간 다르다(예: 일부 결제·인터넷 기업). 벤치 섹터 비중도 Yahoo 펀드 자료의 **현재** 값이라
# 기간 내내 같다고 가정한다 — 그 차이는 '벤치 재구성 오차'로 드러난다(risk.brinson_attribution).
SECTOR_KEY = {
    "Technology": "technology", "Financial Services": "financial_services", "Healthcare": "healthcare",
    "Consumer Cyclical": "consumer_cyclical", "Consumer Defensive": "consumer_defensive",
    "Energy": "energy", "Industrials": "industrials", "Basic Materials": "basic_materials",
    "Utilities": "utilities", "Real Estate": "realestate", "Communication Services": "communication_services",
}
SECTOR_NAME = {  # key → (한국어, English)
    "technology": ("기술", "Technology"), "financial_services": ("금융", "Financials"),
    "healthcare": ("헬스케어", "Health care"), "consumer_cyclical": ("경기소비재", "Consumer discretionary"),
    "consumer_defensive": ("필수소비재", "Consumer staples"), "energy": ("에너지", "Energy"),
    "industrials": ("산업재", "Industrials"), "basic_materials": ("소재", "Materials"),
    "utilities": ("유틸리티", "Utilities"), "realestate": ("부동산", "Real estate"),
    "communication_services": ("커뮤니케이션", "Communication"),
    "미분류": ("미분류", "Unclassified"), "현금": ("현금", "Cash"),
}
# 미국 벤치(SPY·QQQ 등)는 SPDR 섹터 ETF, 전 세계 벤치(ACWI)는 iShares 글로벌 섹터 ETF
SECTOR_ETF_US = {
    "technology": "XLK", "financial_services": "XLF", "healthcare": "XLV", "consumer_cyclical": "XLY",
    "consumer_defensive": "XLP", "energy": "XLE", "industrials": "XLI", "basic_materials": "XLB",
    "utilities": "XLU", "realestate": "XLRE", "communication_services": "XLC",
}
SECTOR_ETF_GLOBAL = {
    "technology": "IXN", "financial_services": "IXG", "healthcare": "IXJ", "consumer_cyclical": "RXI",
    "consumer_defensive": "KXI", "energy": "IXC", "industrials": "EXI", "basic_materials": "MXI",
    "utilities": "JXI", "realestate": "REET", "communication_services": "IXP",
}
GLOBAL_BENCHES = {"ACWI", "VT", "URTH"}


def sector_etfs(bench: str) -> dict[str, str]:
    return SECTOR_ETF_GLOBAL if bench in GLOBAL_BENCHES else SECTOR_ETF_US


def stock_sector(t: str) -> str | None:
    """Yahoo 섹터 → 키. ETF·실패는 None (기여도에서 '미분류'로 간다)."""
    try:
        return SECTOR_KEY.get(yf.Ticker(t).info.get("sector") or "")
    except Exception:
        return None


def bench_sector_weights(bench: str) -> dict[str, float]:
    """벤치(ETF)의 현재 섹터 비중. 펀드가 아니거나 못 받으면 빈 dict."""
    try:
        sw = yf.Ticker(bench).funds_data.sector_weightings or {}
        return {k: float(v) for k, v in sw.items() if float(v) > 0}
    except Exception:
        return {}
