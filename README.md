# Stock Trading Simulator

Multi-agent stock trading simulator that combines technical analysis with news sentiment to generate trading signals.

## Agents

- **Technical Agent** — computes weighted signals from SMA (20/50), EMA (12/26), RSI (14), and MACD (12, 26, 9) using Yahoo Finance price data
- **News Agent** — fetches recent headlines via yfinance and scores sentiment using the Claude API. Falls back to neutral (0.0) if no API key is set

## Setup

```bash
pip3 install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"  # optional, for news sentiment
```

## Usage

```bash
python3 controller.py AAPL 2024-10-01 2024-10-31
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--tech-weight` | 0.6 | Weight for technical signals |
| `--news-weight` | 0.4 | Weight for news sentiment |
| `--output` | results.txt | Output file path |

## Output

CSV file with columns: `date, ticker, tech_signal, news_signal, combined_signal`

All signals are in the range [-1, 1] where positive = bullish and negative = bearish.
