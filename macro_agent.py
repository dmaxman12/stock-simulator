"""Macro environment agent — assesses broad market conditions.

Uses four equally-weighted indicators (no ticker needed — macro is market-wide):
  1. WTI crude oil (CL=F)         — rising = inflationary = bearish
  2. 10-year US Treasury yield (^TNX) — rising = tighter conditions = bearish
  3. US Dollar Index (DX-Y.NYB)   — rising = earnings headwind = bearish
  4. US initial jobless claims (FRED ICSA) — rising = weakening economy = bearish
"""

import math
from datetime import datetime, timedelta

import pandas as pd
import requests
import yfinance as yf


# Scale factors for tanh normalization (tuned per indicator)
_SCALE_WTI = 0.10       # 10% move in oil is a full signal
_SCALE_UST = 0.50       # 50 bps move in yields is a full signal
_SCALE_DXY = 0.05       # 5% move in DXY is a full signal
_SCALE_CLAIMS = 0.15    # 15% move in claims is a full signal

_LOOKBACK_DAYS = 20     # ~1 month of trading days
_LOOKBACK_WEEKS = 4     # ~1 month of weekly data


def _momentum_signal(current, previous, scale, invert=True):
    """Convert a % change to [-1, 1] via tanh.

    invert=True means rising value is bearish (returns negative signal).
    """
    if previous == 0:
        return 0.0
    pct_change = (current - previous) / abs(previous)
    raw = math.tanh(pct_change / scale)
    return -raw if invert else raw


def _flatten_columns(df):
    """Flatten MultiIndex columns from yfinance if present."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


def fetch_wti():
    """Fetch WTI crude oil data, compute 20-day momentum signal. Returns [-1, 1]."""
    df = yf.download("CL=F", period="3mo", progress=False)
    if df.empty or len(df) < _LOOKBACK_DAYS + 1:
        return None
    df = _flatten_columns(df)
    close = df["Close"]
    current = float(close.iloc[-1])
    previous = float(close.iloc[-(_LOOKBACK_DAYS + 1)])
    return round(_momentum_signal(current, previous, _SCALE_WTI), 4)


def fetch_ust_10y():
    """Fetch 10-year Treasury yield, compute 20-day momentum signal. Returns [-1, 1]."""
    df = yf.download("^TNX", period="3mo", progress=False)
    if df.empty or len(df) < _LOOKBACK_DAYS + 1:
        return None
    df = _flatten_columns(df)
    close = df["Close"]
    current = float(close.iloc[-1])
    previous = float(close.iloc[-(_LOOKBACK_DAYS + 1)])
    # Yields are in percentage points; convert change to bps-like scale
    # _momentum_signal uses % change, but for yields we want basis-point sensitivity,
    # so we use the raw % change with the bps-calibrated scale factor
    change_bps = (current - previous) / 100  # rough normalization
    raw = math.tanh(change_bps / (_SCALE_UST / 100))
    return round(-raw, 4)  # invert: rising yields = bearish


def fetch_dxy():
    """Fetch US Dollar Index data, compute 20-day momentum signal. Returns [-1, 1]."""
    df = yf.download("DX-Y.NYB", period="3mo", progress=False)
    if df.empty or len(df) < _LOOKBACK_DAYS + 1:
        return None
    df = _flatten_columns(df)
    close = df["Close"]
    current = float(close.iloc[-1])
    previous = float(close.iloc[-(_LOOKBACK_DAYS + 1)])
    return round(_momentum_signal(current, previous, _SCALE_DXY), 4)


def fetch_claims():
    """Fetch US initial jobless claims from FRED public CSV, compute 4-week momentum. Returns [-1, 1]."""
    start = (datetime.now() - timedelta(days=120)).strftime("%Y-%m-%d")
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id=ICSA&cosd={start}"
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException:
        return None

    from io import StringIO
    df = pd.read_csv(StringIO(resp.text))
    # FRED CSV uses "observation_date" or "DATE" depending on endpoint
    date_col = "observation_date" if "observation_date" in df.columns else "DATE"
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.dropna(subset=["ICSA"])
    if len(df) < _LOOKBACK_WEEKS + 1:
        return None

    current = float(df["ICSA"].iloc[-1])
    previous = float(df["ICSA"].iloc[-(_LOOKBACK_WEEKS + 1)])
    return round(_momentum_signal(current, previous, _SCALE_CLAIMS), 4)


def analyze():
    """Public interface. Returns macro signal [-1, 1]. No ticker needed."""
    result = analyze_detailed()
    return result["score"]


def analyze_detailed():
    """Returns detailed breakdown of macro signals.

    Returns dict with keys: score, wti, ust_10y, dxy, claims, raw_data.
    If an indicator fails, it is excluded and remaining indicators are reweighted equally.
    """
    indicators = {}
    raw_data = {}

    for name, fetcher in [("wti", fetch_wti), ("ust_10y", fetch_ust_10y),
                          ("dxy", fetch_dxy), ("claims", fetch_claims)]:
        try:
            val = fetcher()
        except Exception:
            val = None
        if val is not None:
            indicators[name] = val
        raw_data[name] = val

    # Equal weight among available indicators
    if indicators:
        weight = 1.0 / len(indicators)
        score = round(sum(weight * v for v in indicators.values()), 4)
    else:
        score = 0.0

    return {
        "score": score,
        "wti": indicators.get("wti", 0.0),
        "ust_10y": indicators.get("ust_10y", 0.0),
        "dxy": indicators.get("dxy", 0.0),
        "claims": indicators.get("claims", 0.0),
        "raw_data": raw_data,
    }
