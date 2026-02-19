#!/usr/bin/env python3
"""Multi-agent stock trading simulator controller.

Usage:
    python controller.py AAPL 2024-10-01 2024-10-31
    python controller.py AAPL 2024-10-01 2024-10-31 --tech-weight 0.7 --news-weight 0.3
    python controller.py AAPL 2024-10-01 2024-10-31 --output my_results.txt
"""

import argparse
import sys

import technical_agent
import news_agent


def main():
    parser = argparse.ArgumentParser(description="Multi-agent stock trading simulator")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--tech-weight", type=float, default=0.6, help="Technical signal weight (default: 0.6)")
    parser.add_argument("--news-weight", type=float, default=0.4, help="News sentiment weight (default: 0.4)")
    parser.add_argument("--output", default="results.txt", help="Output file (default: results.txt)")
    args = parser.parse_args()

    ticker = args.ticker.upper()
    print(f"Analyzing {ticker} from {args.start_date} to {args.end_date}")

    # Technical analysis — one call for entire date range
    print("Running technical analysis...")
    tech_signals = technical_agent.analyze(ticker, args.start_date, args.end_date)
    if not tech_signals:
        print(f"Error: No price data found for {ticker} in the given date range.", file=sys.stderr)
        sys.exit(1)
    print(f"  Got {len(tech_signals)} trading days of technical signals")

    # News sentiment — one call, reused for all dates
    print("Running news sentiment analysis...")
    news_signal = news_agent.analyze(ticker)
    if news_signal == 0.0:
        print("  News signal: 0.0 (neutral — no API key or no headlines)")
    else:
        print(f"  News signal: {news_signal}")

    # Combine and write results
    output_path = args.output
    with open(output_path, "w") as f:
        f.write("date,ticker,tech_signal,news_signal,combined_signal\n")
        for date in sorted(tech_signals):
            tech = tech_signals[date]
            combined = round(args.tech_weight * tech + args.news_weight * news_signal, 4)
            f.write(f"{date},{ticker},{tech},{news_signal},{combined}\n")

    print(f"\nResults written to {output_path} ({len(tech_signals)} rows)")


if __name__ == "__main__":
    main()
