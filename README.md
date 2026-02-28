# Stock Trading Simulator

Multi-agent stock trading simulator that combines four independent signal sources to generate daily buy/sell signals for any ticker.

## Agents

- **Technical Agent** — SMA (20/50), EMA (12/26), RSI (14), MACD (12,26,9) via Yahoo Finance. Produces a daily signal per trading day in the requested date range.
- **News Agent** — Headlines via yfinance scored by Claude API. Returns a single sentiment signal. Requires `ANTHROPIC_API_KEY`.
- **Honesty Agent** — Analyzes executive earnings-call language for consistency, hedging, and guidance accuracy via Claude API. Requires `ANTHROPIC_API_KEY`.
- **Macro Agent** — Market-wide macro environment signal (no ticker needed). Four equally-weighted indicators, each normalized to [-1, 1] via tanh:
  - WTI crude oil (`CL=F`) — rising oil = inflationary = bearish
  - 10-year US Treasury yield (`^TNX`) — rising yields = tighter conditions = bearish
  - US Dollar Index (`DX-Y.NYB`) — rising dollar = earnings headwind = bearish
  - US initial jobless claims (FRED `ICSA`) — rising claims = weakening economy = bearish

  Supports both point-in-time (`analyze()`) and historical (`analyze_historical()`) modes for backtesting. Gracefully degrades if any indicator is unavailable.

## Setup

```bash
pip3 install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"  # optional, for news + honesty agents
```

## Usage

```bash
python3 controller.py AAPL 2024-10-01 2024-10-31
python3 controller.py AAPL 2021-03-01 2026-02-28 --macro-weight 0.20
```

| Flag | Default | Description |
|------|---------|-------------|
| `--tech-weight` | 0.40 | Technical signal weight |
| `--news-weight` | 0.25 | News sentiment weight |
| `--honesty-weight` | 0.20 | Executive honesty weight |
| `--macro-weight` | 0.15 | Macro environment weight |
| `--output` | results.txt | Output CSV path |

## Output

CSV with columns: `date, ticker, tech_signal, news_signal, honesty_signal, macro_signal, combined_signal`

All signals are in [-1, 1]. Positive = bullish, negative = bearish.

## Data Sources

| Source | API Key Required | Agent |
|--------|:---:|--------|
| Yahoo Finance (yfinance) | No | Technical, Macro |
| FRED public CSV | No | Macro (jobless claims) |
| Claude API | Yes | News, Honesty |
