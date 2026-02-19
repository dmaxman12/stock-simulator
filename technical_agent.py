import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta


def analyze(ticker: str, start_date: str, end_date: str) -> dict[str, float]:
    """Compute a weighted technical signal for each trading day in [start_date, end_date].

    Returns {date_string: signal} where signal is in [-1, 1].
    Positive = bullish, negative = bearish.
    """
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")

    # Fetch extra 80 calendar days for indicator warmup
    fetch_start = (start_dt - timedelta(days=80)).strftime("%Y-%m-%d")
    fetch_end = (end_dt + timedelta(days=1)).strftime("%Y-%m-%d")

    df = yf.download(ticker, start=fetch_start, end=fetch_end, progress=False)
    if df.empty:
        return {}

    # Flatten multi-level columns if present (yfinance sometimes returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]

    # --- SMA crossover (20/50) ---
    sma20 = close.rolling(window=20).mean()
    sma50 = close.rolling(window=50).mean()
    sma_signal = ((sma20 - sma50) / close).clip(-1, 1)

    # --- EMA crossover (12/26) ---
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    ema_signal = ((ema12 - ema26) / close).clip(-1, 1)

    # --- RSI (14-period) ---
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(window=14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_signal = ((50 - rsi) / 50).clip(-1, 1)

    # --- MACD (12, 26, 9) ---
    macd_line = ema12 - ema26
    macd_signal_line = macd_line.ewm(span=9, adjust=False).mean()
    histogram = macd_line - macd_signal_line
    macd_signal = (histogram / close * 100).clip(-1, 1)

    # Weighted combination
    combined = (
        0.20 * sma_signal
        + 0.20 * ema_signal
        + 0.25 * rsi_signal
        + 0.35 * macd_signal
    )

    # Filter to requested date range
    mask = (df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))
    result_series = combined[mask].dropna()

    return {dt.strftime("%Y-%m-%d"): round(float(val), 4) for dt, val in result_series.items()}
