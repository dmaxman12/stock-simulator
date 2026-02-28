#!/usr/bin/env python3
"""Multi-agent stock trading simulator controller.

Usage:
    python controller.py AAPL 2024-10-01 2024-10-31
    python controller.py AAPL 2024-10-01 2024-10-31 --tech-weight 0.4 --news-weight 0.25 --honesty-weight 0.2 --macro-weight 0.15
    python controller.py AAPL 2024-10-01 2024-10-31 --output my_results.txt
"""

import argparse
import sys

import technical_agent
import news_agent
import honesty_agent
import macro_agent


def main():
    parser = argparse.ArgumentParser(description="Multi-agent stock trading simulator")
    parser.add_argument("ticker", help="Stock ticker symbol (e.g. AAPL)")
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--tech-weight", type=float, default=0.40, help="Technical signal weight (default: 0.40)")
    parser.add_argument("--news-weight", type=float, default=0.25, help="News sentiment weight (default: 0.25)")
    parser.add_argument("--honesty-weight", type=float, default=0.20, help="Executive honesty weight (default: 0.20)")
    parser.add_argument("--macro-weight", type=float, default=0.15, help="Macro environment weight (default: 0.15)")
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

    # Executive honesty assessment — one call, reused for all dates
    print("Running executive honesty analysis...")
    honesty_result = honesty_agent.analyze_detailed(ticker)
    honesty_signal = honesty_result["score"]
    if honesty_signal == 0.0:
        print("  Honesty signal: 0.0 (neutral — no API key or insufficient data)")
    else:
        print(f"  Honesty signal: {honesty_signal}")
        print(f"    Consistency: {honesty_result['consistency']}")
        print(f"    Hedging: {honesty_result['hedging']}")
        print(f"    Guidance accuracy: {honesty_result['guidance_accuracy']}")
        if honesty_result["flagged_statements"]:
            print(f"    Flagged: {len(honesty_result['flagged_statements'])} statement(s)")

    # Macro environment assessment — daily historical signals
    print("Running macro environment analysis...")
    macro_signals = macro_agent.analyze_historical(args.start_date, args.end_date)
    if not macro_signals:
        print("  Macro signals: none (no data available)")
    else:
        vals = list(macro_signals.values())
        avg = sum(vals) / len(vals)
        print(f"  Got {len(macro_signals)} days of macro signals")
        print(f"    Average: {avg:.4f}  Min: {min(vals):.4f}  Max: {max(vals):.4f}")

    # Combine and write results
    output_path = args.output
    with open(output_path, "w") as f:
        f.write("date,ticker,tech_signal,news_signal,honesty_signal,macro_signal,combined_signal\n")
        for date in sorted(tech_signals):
            tech = tech_signals[date]
            macro = macro_signals.get(date, 0.0)
            combined = round(
                args.tech_weight * tech
                + args.news_weight * news_signal
                + args.honesty_weight * honesty_signal
                + args.macro_weight * macro,
                4,
            )
            f.write(f"{date},{ticker},{tech},{news_signal},{honesty_signal},{macro},{combined}\n")

    print(f"\nResults written to {output_path} ({len(tech_signals)} rows)")


if __name__ == "__main__":
    main()
