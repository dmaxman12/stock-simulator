#!/usr/bin/env python3
"""Standalone CLI for testing and backtesting individual agents.

Usage:
    python3 test_agent.py technical AAPL --start 2024-10-01 --end 2024-10-31
    python3 test_agent.py technical AAPL --start 2024-01-01 --end 2024-12-31 --backtest
    python3 test_agent.py news AAPL
    python3 test_agent.py news AAPL --backtest --days 30
    python3 test_agent.py honesty AAPL
    python3 test_agent.py honesty AAPL --backtest --days 30
    python3 test_agent.py macro
    python3 test_agent.py macro --start 2024-01-01 --end 2024-12-31 --backtest
    python3 test_agent.py news AAPL --json
"""

import argparse
import json
import math
import sys
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

import technical_agent
import news_agent
import honesty_agent
import macro_agent


# ---------------------------------------------------------------------------
# Backtesting engine
# ---------------------------------------------------------------------------

def fetch_prices(ticker, start_date, end_date):
    """Fetch daily close prices. Returns {date_string: price}."""
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    if df.empty:
        return {}
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return {dt.strftime("%Y-%m-%d"): float(df["Close"].loc[dt]) for dt in df.index}


def run_backtest(signals, prices, label="Agent"):
    """Simulate trading where position size = signal value each day.

    signals: {date: float} signal in [-1, 1]
    prices:  {date: float} close prices

    Prints performance summary and returns results dict.
    """
    dates = sorted(d for d in signals if d in prices)
    if len(dates) < 2:
        print("Not enough overlapping dates for backtest.")
        return None

    # Daily returns
    daily_returns = []
    agent_returns = []
    equity = 1.0
    peak = 1.0
    max_dd = 0.0
    wins = 0
    losses = 0

    for i in range(1, len(dates)):
        prev_date = dates[i - 1]
        curr_date = dates[i]
        price_return = (prices[curr_date] - prices[prev_date]) / prices[prev_date]
        position = signals[prev_date]  # signal from previous day drives today's position
        agent_ret = position * price_return

        daily_returns.append(price_return)
        agent_returns.append(agent_ret)

        equity *= (1 + agent_ret)
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak
        max_dd = max(max_dd, drawdown)

        if agent_ret > 0:
            wins += 1
        elif agent_ret < 0:
            losses += 1

    total_return = equity - 1.0
    bh_equity = prices[dates[-1]] / prices[dates[0]]
    bh_return = bh_equity - 1.0

    n = len(agent_returns)
    avg = sum(agent_returns) / n
    var = sum((r - avg) ** 2 for r in agent_returns) / n
    std = math.sqrt(var) if var > 0 else 0.0001
    sharpe = (avg / std) * math.sqrt(252) if std > 0 else 0.0

    total_trades = wins + losses

    print(f"\n{'='*50}")
    print(f"BACKTEST: {label}")
    print(f"{'='*50}")
    print(f"  Period:          {dates[0]} to {dates[-1]} ({n} days)")
    print(f"  Agent return:    {total_return:+.2%}")
    print(f"  Buy & hold:      {bh_return:+.2%}")
    print(f"  Excess return:   {total_return - bh_return:+.2%}")
    print(f"  Sharpe ratio:    {sharpe:.2f}")
    print(f"  Max drawdown:    {max_dd:.2%}")
    print(f"  Win rate:        {wins}/{total_trades} ({wins/total_trades:.0%})" if total_trades > 0 else "  Win rate:        N/A")
    print(f"  Avg daily ret:   {avg:+.4%}")
    print(f"{'='*50}")

    return {
        "period": f"{dates[0]} to {dates[-1]}",
        "trading_days": n,
        "agent_return": round(total_return, 6),
        "buy_hold_return": round(bh_return, 6),
        "excess_return": round(total_return - bh_return, 6),
        "sharpe": round(sharpe, 4),
        "max_drawdown": round(max_dd, 6),
        "win_rate": round(wins / total_trades, 4) if total_trades > 0 else None,
        "avg_daily_return": round(avg, 6),
    }


def forward_backtest(ticker, signal, days, label="Agent"):
    """For point-in-time agents: apply a constant signal over the next N days
    and see what would have happened.

    Returns results dict.
    """
    end = datetime.now()
    start = end - timedelta(days=days + 5)  # small buffer for weekends
    prices = fetch_prices(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    if not prices:
        print("Could not fetch price data.")
        return None

    sorted_dates = sorted(prices.keys())
    # Take the last `days` trading days
    if len(sorted_dates) > days:
        sorted_dates = sorted_dates[-days:]

    signals = {d: signal for d in sorted_dates}
    print(f"\n  Applied constant signal {signal:+.4f} over last {len(sorted_dates)} trading days")
    return run_backtest(signals, prices, label=label)


# ---------------------------------------------------------------------------
# Agent runners
# ---------------------------------------------------------------------------

def run_technical(args):
    if not args.start or not args.end:
        print("Error: --start and --end are required for technical agent.", file=sys.stderr)
        sys.exit(1)

    ticker = args.ticker.upper()
    print(f"Technical Agent — {ticker} ({args.start} to {args.end})\n")

    signals = technical_agent.analyze(ticker, args.start, args.end)
    if not signals:
        print("No data found.")
        return

    if args.backtest:
        prices = fetch_prices(ticker, args.start, args.end)
        result = run_backtest(signals, prices, label=f"Technical — {ticker}")
        if args.json and result:
            print(json.dumps(result, indent=2))
        return

    if args.json:
        print(json.dumps(signals, indent=2))
        return

    print(f"{'Date':<12} {'Signal':>8}")
    print("-" * 22)
    for date in sorted(signals):
        val = signals[date]
        bar = "+" * int(abs(val) * 20) if val >= 0 else "-" * int(abs(val) * 20)
        print(f"{date:<12} {val:>8.4f}  {bar}")

    vals = list(signals.values())
    print(f"\n{len(vals)} trading days")
    print(f"  Avg: {sum(vals)/len(vals):.4f}  Min: {min(vals):.4f}  Max: {max(vals):.4f}")


def run_news(args):
    ticker = args.ticker.upper()
    print(f"News Agent — {ticker}\n")

    headlines = news_agent.search_news(ticker)
    if not headlines:
        print("No headlines found.")
    else:
        print(f"Headlines ({len(headlines)}):")
        for h in headlines:
            print(f"  • {h}")

    print()
    score = news_agent.analyze(ticker)

    label = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
    print(f"Sentiment: {score:+.4f} ({label})")

    if args.backtest:
        days = args.days or 30
        result = forward_backtest(ticker, score, days, label=f"News — {ticker}")
        if args.json and result:
            print(json.dumps({"signal": score, "headlines": headlines, **result}, indent=2))
        return

    if args.json:
        print(json.dumps({"ticker": ticker, "headlines": headlines, "score": score}, indent=2))


def run_honesty(args):
    ticker = args.ticker.upper()
    print(f"Honesty Agent — {ticker}\n")

    # Show intermediate data
    print("Fetching executive statements...")
    statements = honesty_agent.fetch_executive_statements(ticker)
    print(f"  Found {len(statements)} statement(s)")
    for s in statements[:5]:
        print(f"  • {s[:120]}{'...' if len(s) > 120 else ''}")
    if len(statements) > 5:
        print(f"  ... and {len(statements) - 5} more")

    print("\nFetching SEC filings...")
    filings = honesty_agent.fetch_sec_filings(ticker)
    print(f"  {'Got ' + str(len(filings)) + ' chars' if filings else 'None found'}")

    print("\nFetching financials...")
    financials = honesty_agent.fetch_financials(ticker)
    if financials:
        for k, v in financials.items():
            if v is not None and k != "earnings_surprises":
                print(f"  {k}: {v}")
        surprises = financials.get("earnings_surprises", [])
        if surprises:
            print(f"  earnings_surprises: {surprises}")

    print("\nRunning veracity assessment...")
    result = honesty_agent.assess_veracity(ticker, statements, filings, financials)

    print(f"\n  Overall score:      {result['score']:+.4f}")
    print(f"  Consistency:        {result['consistency']:+.4f}")
    print(f"  Hedging:            {result['hedging']:+.4f}")
    print(f"  Guidance accuracy:  {result['guidance_accuracy']:+.4f}")
    if result["flagged_statements"]:
        print(f"\n  Flagged ({len(result['flagged_statements'])}):")
        for f in result["flagged_statements"]:
            print(f"    ⚠ {f}")

    if args.backtest:
        days = args.days or 30
        bt = forward_backtest(ticker, result["score"], days, label=f"Honesty — {ticker}")
        if args.json and bt:
            print(json.dumps({"signal": result, **bt}, indent=2))
        return

    if args.json:
        print(json.dumps({"ticker": ticker, "statements_count": len(statements), **result}, indent=2))


def run_macro(args):
    historical = args.start and args.end

    if historical:
        print(f"Macro Agent — Historical ({args.start} to {args.end})\n")
        signals = macro_agent.analyze_historical(args.start, args.end)
        if not signals:
            print("No data available.")
            return

        if args.backtest:
            # Macro is market-wide; backtest against SPY as proxy
            print("  Backtesting against SPY (market proxy)")
            prices = fetch_prices("SPY", args.start, args.end)
            result = run_backtest(signals, prices, label="Macro vs SPY")
            if args.json and result:
                print(json.dumps(result, indent=2))
            return

        if args.json:
            print(json.dumps(signals, indent=2))
            return

        print(f"{'Date':<12} {'Signal':>8}")
        print("-" * 22)
        for date in sorted(signals):
            val = signals[date]
            bar = "+" * int(abs(val) * 20) if val >= 0 else "-" * int(abs(val) * 20)
            print(f"{date:<12} {val:>8.4f}  {bar}")

        vals = list(signals.values())
        print(f"\n{len(vals)} trading days")
        print(f"  Avg: {sum(vals)/len(vals):.4f}  Min: {min(vals):.4f}  Max: {max(vals):.4f}")
    else:
        print("Macro Agent — Point-in-Time Snapshot\n")
        result = macro_agent.analyze_detailed()

        if args.backtest:
            days = args.days or 30
            print("  Backtesting against SPY (market proxy)")
            bt = forward_backtest("SPY", result["score"], days, label="Macro vs SPY")
            if args.json and bt:
                print(json.dumps({"signal": result, **bt}, indent=2))
            return

        if args.json:
            print(json.dumps(result, indent=2))
            return

        print(f"  WTI crude oil:     {result['wti']:+.4f}")
        print(f"  10Y Treasury:      {result['ust_10y']:+.4f}")
        print(f"  US Dollar Index:   {result['dxy']:+.4f}")
        print(f"  Jobless claims:    {result['claims']:+.4f}")
        print(f"\n  Combined score:    {result['score']:+.4f}")

        label = "bullish" if result["score"] > 0.1 else "bearish" if result["score"] < -0.1 else "neutral"
        print(f"  Environment:       {label}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test and backtest individual stock-simulator agents")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    parser.add_argument("--backtest", action="store_true", help="Run backtest on agent signals")
    parser.add_argument("--days", type=int, help="Lookback days for forward backtest (news/honesty/macro point-in-time)")
    sub = parser.add_subparsers(dest="agent", required=True)

    # Technical
    p = sub.add_parser("technical", aliases=["tech"], help="Technical analysis agent")
    p.add_argument("ticker", help="Stock ticker (e.g. AAPL)")
    p.add_argument("--start", required=True, help="Start date YYYY-MM-DD")
    p.add_argument("--end", required=True, help="End date YYYY-MM-DD")

    # News
    p = sub.add_parser("news", help="News sentiment agent")
    p.add_argument("ticker", help="Stock ticker (e.g. AAPL)")

    # Honesty
    p = sub.add_parser("honesty", help="Executive honesty agent")
    p.add_argument("ticker", help="Stock ticker (e.g. AAPL)")

    # Macro
    p = sub.add_parser("macro", help="Macro environment agent")
    p.add_argument("--start", help="Start date YYYY-MM-DD (omit for point-in-time)")
    p.add_argument("--end", help="End date YYYY-MM-DD (omit for point-in-time)")

    args = parser.parse_args()

    runners = {
        "technical": run_technical, "tech": run_technical,
        "news": run_news,
        "honesty": run_honesty,
        "macro": run_macro,
    }
    runners[args.agent](args)


if __name__ == "__main__":
    main()
