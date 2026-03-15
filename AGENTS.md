# Agent Logic Reference

## Technical Agent (`technical_agent.py`)

**Input:** ticker, start_date, end_date
**Output:** `{date: signal}` where signal is in [-1, 1]

Fetches daily closes from Yahoo Finance with 80 extra calendar days before start_date for indicator warmup.

### Indicators

| Indicator | Parameters | Weight | Type |
|-----------|-----------|--------|------|
| SMA Crossover | 20/50-day | 20% | Trend-following |
| EMA Crossover | 12/26-day | 20% | Trend-following |
| RSI | 14-period | 25% | Mean-reverting |
| MACD | 12, 26, 9 | 35% | Momentum |

**SMA Crossover (20/50)** — `(SMA20 - SMA50) / price`, clipped to [-1, 1]. Positive when short-term average is above long-term, indicating an uptrend.

**EMA Crossover (12/26)** — `(EMA12 - EMA26) / price`, clipped to [-1, 1]. Same logic as SMA but uses exponential moving averages that react faster to recent price changes.

**RSI (14-period)** — `(50 - RSI) / 50`, clipped to [-1, 1]. Contrarian: goes negative when RSI > 50 (overbought), positive when RSI < 50 (oversold). This is the primary reason the agent underperforms in sustained uptrends — it sells into strength.

**MACD (12, 26, 9)** — `histogram / price * 100`, clipped to [-1, 1]. The MACD histogram measures the difference between the MACD line (EMA12 - EMA26) and its 9-period signal line. Positive histogram = bullish momentum building.

### Combined Signal

```
signal = 0.20 * SMA + 0.20 * EMA + 0.25 * RSI + 0.35 * MACD
```

### Known Characteristics
- Conservative in uptrends due to RSI mean-reversion (25% weight)
- SMA/EMA crossover signals are divided by price, producing small values
- Low drawdowns but tends to lag buy-and-hold in bull markets
- Better at risk management than return generation

---

## News Agent (`news_agent.py`)

**Input:** ticker
**Output:** single float in [-1, 1]

Point-in-time only — yfinance always returns the most recent headlines regardless of date parameter.

### Process
1. Fetch up to 20 recent headlines via `yfinance.Ticker.news`
2. Send headlines to Claude API (claude-sonnet-4-20250514) with prompt to score sentiment
3. Claude returns a single float: -1.0 (very bearish) to 1.0 (very bullish)

### Dependencies
- `ANTHROPIC_API_KEY` environment variable required
- Returns 0.0 (neutral) if no API key or no headlines found

---

## Honesty Agent (`honesty_agent.py`)

**Input:** ticker
**Output:** dict with `score`, `consistency`, `hedging`, `guidance_accuracy`, `flagged_statements`

Point-in-time only. Assesses executive credibility by cross-referencing public statements against financial data.

### Data Sources
1. **Executive statements** — news headlines and summaries via yfinance (up to 25)
2. **SEC filings** — MD&A section from most recent 10-K/10-Q via EDGAR API
3. **Financials** — revenue, earnings, margins, earnings surprises via yfinance

### Assessment Dimensions

| Dimension | Range | Positive means | Negative means |
|-----------|-------|---------------|----------------|
| Consistency | [-1, 1] | Transparent, consistent messaging | Contradictory, shifting narratives |
| Hedging | [-1, 1] | Direct, specific language | Excessive hedging, vague promises |
| Guidance Accuracy | [-1, 1] | Conservative guidance, beats estimates | Over-promising vs actuals |

### Process
1. Fetch all three data sources
2. Send combined context to Claude API (claude-sonnet-4-20250514) as a forensic financial analyst prompt
3. Claude returns JSON with scores and flagged statements
4. Overall score is clamped to [-1, 1]

### Dependencies
- `ANTHROPIC_API_KEY` environment variable required
- Returns all zeros if no API key or insufficient data

---

## Macro Agent (`macro_agent.py`)

**Input:** none (market-wide), optional start_date/end_date for historical mode
**Output:** single float or `{date: signal}` in [-1, 1]

No ticker needed — assesses broad market conditions.

### Indicators

| Indicator | Source | Scale Factor | Interpretation |
|-----------|--------|-------------|----------------|
| WTI Crude Oil (CL=F) | Yahoo Finance | 0.10 (10%) | Rising oil = inflationary = bearish |
| 10Y Treasury Yield (^TNX) | Yahoo Finance | 0.50 (50 bps) | Rising yields = tighter conditions = bearish |
| US Dollar Index (DX-Y.NYB) | Yahoo Finance | 0.05 (5%) | Rising dollar = earnings headwind = bearish |
| Initial Jobless Claims (ICSA) | FRED CSV | 0.15 (15%) | Rising claims = weakening economy = bearish |

All four indicators are inverted: rising values produce negative (bearish) signals.

### Signal Computation
1. Compute 20-day (or 4-week for claims) percentage change
2. Normalize via `tanh(pct_change / scale_factor)` to [-1, 1]
3. Invert sign (rising = bearish)
4. Equal-weight average of available indicators

The scale factors control sensitivity — e.g., a 10% move in oil produces a near-full signal (tanh ≈ ±0.76), while smaller moves produce proportionally smaller signals.

### Modes
- **Point-in-time** (`analyze()`) — latest values only, single combined score
- **Historical** (`analyze_historical(start, end)`) — daily signals for backtesting, with weekly claims forward-filled to daily

### Graceful Degradation
If any indicator is unavailable, remaining indicators are reweighted equally.

---

## Controller (`controller.py`)

Combines all four agents with configurable weights:

```
combined = tech_weight * technical + news_weight * news + honesty_weight * honesty + macro_weight * macro
```

| Agent | Default Weight |
|-------|---------------|
| Technical | 40% |
| News | 25% |
| Honesty | 20% |
| Macro | 15% |

Technical and macro produce daily signals. News and honesty produce a single signal that is held constant across all dates.
