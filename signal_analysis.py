#!/usr/bin/env python3
"""Signal predictive power analysis and walk-forward weight optimization.

Tests the predictive power of each agent (technical, macro) independently via
Information Coefficient (IC) analysis, then finds optimal mixture weights using
walk-forward ridge regression.

Usage:
    python3 signal_analysis.py
"""

import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge

import technical_agent
import macro_agent

# ── Configuration ────────────────────────────────────────────────────────────
PRIMARY_TICKER = "XLI"
CROSS_TICKERS = ["SPY", "QQQ", "XLE"]
START_DATE = "2016-03-01"
END_DATE = "2026-02-28"
HORIZONS = [1, 5, 10, 20]
TRAIN_YEARS = 2
TEST_MONTHS = 6
RIDGE_ALPHA = 1.0
OPT_HORIZON = 5  # forward return horizon for ridge training
FIXED_W_TECH = 0.727  # 0.40 / (0.40 + 0.15), normalized to 2-agent subset
FIXED_W_MACRO = 0.273  # 0.15 / (0.40 + 0.15)


# ── Data Loading ─────────────────────────────────────────────────────────────

def load_signals(ticker, start_date, end_date):
    """Fetch tech + macro signals and equity close prices, aligned on equity calendar."""
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    buffer_end = (end_dt + timedelta(days=45)).strftime("%Y-%m-%d")

    # Equity close prices (master calendar + buffer for forward returns)
    price_df = yf.download(ticker, start=start_date, end=buffer_end, progress=False)
    if isinstance(price_df.columns, pd.MultiIndex):
        price_df.columns = price_df.columns.get_level_values(0)
    close = price_df[["Close"]].rename(columns={"Close": "close"})

    # Agent signals
    tech_dict = technical_agent.analyze(ticker, start_date, end_date)
    macro_dict = macro_agent.analyze_historical(start_date, end_date)

    tech_series = pd.Series(tech_dict, name="tech_signal")
    tech_series.index = pd.to_datetime(tech_series.index)
    macro_series = pd.Series(macro_dict, name="macro_signal")
    macro_series.index = pd.to_datetime(macro_series.index)

    # Align on equity calendar
    df = close.copy()
    df = df.join(tech_series, how="left")
    df = df.join(macro_series, how="left")
    df["macro_signal"] = df["macro_signal"].ffill(limit=2)

    # Coverage report
    sig_mask = df.index <= pd.Timestamp(end_date)
    n_total = sig_mask.sum()
    n_tech = df.loc[sig_mask, "tech_signal"].notna().sum()
    n_macro = df.loc[sig_mask, "macro_signal"].notna().sum()
    print(f"  Coverage: {n_tech}/{n_total} tech, {n_macro}/{n_total} macro")

    return df


def compute_forward_returns(df, horizons=None):
    """Add forward return columns: fwd_1d, fwd_5d, etc."""
    if horizons is None:
        horizons = HORIZONS
    for n in horizons:
        df[f"fwd_{n}d"] = df["close"].shift(-n) / df["close"] - 1
    return df


# ── Part 1: IC Analysis ─────────────────────────────────────────────────────

def compute_ic(signal, fwd_return):
    """Spearman rank correlation between signal and forward return."""
    valid = pd.DataFrame({"s": signal, "r": fwd_return}).dropna()
    if len(valid) < 10:
        return np.nan
    corr, _ = spearmanr(valid["s"], valid["r"])
    return corr


def compute_rolling_monthly_ic(df, signal_col, return_col):
    """Compute IC per calendar month. Returns Series indexed by month."""
    df = df[[signal_col, return_col]].dropna()
    monthly_ic = {}
    for (yr, mo), grp in df.groupby([df.index.year, df.index.month]):
        if len(grp) >= 10:
            ic = compute_ic(grp[signal_col], grp[return_col])
            monthly_ic[pd.Timestamp(yr, mo, 1)] = ic
    return pd.Series(monthly_ic).sort_index()


def compute_quintile_spread(df, signal_col, return_col):
    """Average return of top quintile minus bottom quintile."""
    valid = df[[signal_col, return_col]].dropna()
    if len(valid) < 50:
        return np.nan
    valid = valid.copy()
    valid["q"] = pd.qcut(valid[signal_col], 5, labels=False, duplicates="drop")
    q_means = valid.groupby("q")[return_col].mean()
    return q_means.iloc[-1] - q_means.iloc[0]


def run_ic_analysis(all_data):
    """Run IC analysis for all tickers and agents. Returns nested dict."""
    results = {}
    for ticker, df in all_data.items():
        results[ticker] = {}
        for agent in ["tech_signal", "macro_signal"]:
            agent_stats = {}
            for h in HORIZONS:
                ret_col = f"fwd_{h}d"
                valid = df[[agent, ret_col]].dropna()
                overall_ic = compute_ic(valid[agent], valid[ret_col])
                monthly_ic = compute_rolling_monthly_ic(df, agent, ret_col)
                q_spread = compute_quintile_spread(df, agent, ret_col)

                ic_mean = monthly_ic.mean() if len(monthly_ic) > 0 else np.nan
                ic_std = monthly_ic.std() if len(monthly_ic) > 1 else np.nan
                ic_ir = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan
                hit_rate = (monthly_ic > 0).mean() if len(monthly_ic) > 0 else np.nan

                agent_stats[h] = {
                    "ic": overall_ic,
                    "ic_mean": ic_mean,
                    "ic_std": ic_std,
                    "ic_ir": ic_ir,
                    "hit_rate": hit_rate,
                    "q_spread": q_spread,
                    "n_months": len(monthly_ic),
                }
            results[ticker][agent] = agent_stats
    return results


def print_ic_report(ic_results):
    """Print IC analysis tables."""
    print(f"\n{'='*80}")
    print(f"  PART 1: INFORMATION COEFFICIENT ANALYSIS")
    print(f"{'='*80}")

    for ticker in ic_results:
        for agent in ["tech_signal", "macro_signal"]:
            label = "Technical" if agent == "tech_signal" else "Macro"
            stats = ic_results[ticker][agent]
            print(f"\n  [{ticker}] {label} Agent")
            print(f"  {'Horizon':>8s} │ {'IC':>7s} {'IC Mean':>8s} {'IC IR':>7s} {'Hit Rate':>9s} {'Q5-Q1':>9s} {'Months':>7s}")
            print(f"  {'─'*8}─┼─{'─'*52}")
            for h in HORIZONS:
                s = stats[h]
                print(f"  {h:>5d}-day │ {s['ic']:>+6.3f} {s['ic_mean']:>+7.3f} {s['ic_ir']:>+6.2f}  {s['hit_rate']:>7.1%}  {s['q_spread']:>+8.4f} {s['n_months']:>7d}")

    # Cross-ticker summary at 5-day horizon
    print(f"\n  Cross-Ticker IC Summary (5-day horizon)")
    print(f"  {'Ticker':>6s} │ {'Tech IC':>8s} {'Tech HR':>8s} │ {'Macro IC':>9s} {'Macro HR':>9s}")
    print(f"  {'─'*6}─┼─{'─'*17}─┼─{'─'*19}")
    for ticker in ic_results:
        t = ic_results[ticker]["tech_signal"][5]
        m = ic_results[ticker]["macro_signal"][5]
        print(f"  {ticker:>6s} │ {t['ic']:>+7.3f} {t['hit_rate']:>7.1%}  │ {m['ic']:>+8.3f} {m['hit_rate']:>8.1%}")


# ── Part 2: Walk-Forward Optimization ────────────────────────────────────────

def walk_forward_split(dates, train_years=TRAIN_YEARS, test_months=TEST_MONTHS):
    """Generate rolling train/test splits. Returns list of (train_dates, test_dates)."""
    splits = []
    min_date = dates.min()
    max_date = dates.max()

    train_start = min_date
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(months=test_months)

        if test_end > max_date:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= test_start) & (dates < test_end)

        if train_mask.sum() >= 200 and test_mask.sum() >= 40:
            splits.append((train_mask, test_mask))

        train_start += pd.DateOffset(months=test_months)

    return splits


def fit_ridge_weights(X_train, y_train, alpha=RIDGE_ALPHA):
    """Fit ridge regression, return weight vector."""
    valid = ~(np.isnan(X_train).any(axis=1) | np.isnan(y_train))
    X = X_train[valid]
    y = y_train[valid]
    if len(X) < 20:
        return np.array([FIXED_W_TECH, FIXED_W_MACRO])
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(X, y)
    return model.coef_


def walk_forward_backtest(df):
    """Run walk-forward optimization. Returns results dict."""
    dates = df.index
    splits = walk_forward_split(dates)

    if not splits:
        print("  Warning: not enough data for walk-forward splits")
        return None

    X_cols = ["tech_signal", "macro_signal"]
    ret_col = f"fwd_{OPT_HORIZON}d"

    weight_history = []
    oos_returns = []
    fixed_returns = []
    bh_returns = []
    window_details = []

    for i, (train_mask, test_mask) in enumerate(splits):
        train = df[train_mask].dropna(subset=X_cols + [ret_col])
        test = df[test_mask].dropna(subset=X_cols + ["fwd_1d"])

        X_train = train[X_cols].values
        y_train = train[ret_col].values
        weights = fit_ridge_weights(X_train, y_train)

        train_start = df.index[train_mask][0].strftime("%Y-%m")
        train_end = df.index[train_mask][-1].strftime("%Y-%m")
        test_start = df.index[test_mask][0].strftime("%Y-%m")
        test_end = df.index[test_mask][-1].strftime("%Y-%m")

        # Out-of-sample: signal-proportional returns
        combined = test[X_cols].values @ weights
        oos_daily = combined * test["fwd_1d"].values

        # Fixed-weight baseline
        fixed_combined = test["tech_signal"].values * FIXED_W_TECH + test["macro_signal"].values * FIXED_W_MACRO
        fixed_daily = fixed_combined * test["fwd_1d"].values

        # Buy and hold
        bh_daily = test["fwd_1d"].values

        oos_returns.extend(oos_daily)
        fixed_returns.extend(fixed_daily)
        bh_returns.extend(bh_daily)

        w_total = abs(weights[0]) + abs(weights[1])
        w_tech_pct = abs(weights[0]) / w_total * 100 if w_total > 0 else 50
        w_macro_pct = abs(weights[1]) / w_total * 100 if w_total > 0 else 50

        oos_sharpe = np.mean(oos_daily) / np.std(oos_daily) * np.sqrt(252) if np.std(oos_daily) > 0 else 0

        weight_history.append({
            "window": i + 1,
            "train": f"{train_start} to {train_end}",
            "test": f"{test_start} to {test_end}",
            "w_tech": weights[0],
            "w_macro": weights[1],
            "tech_pct": w_tech_pct,
            "macro_pct": w_macro_pct,
            "oos_sharpe": oos_sharpe,
            "n_test": len(test),
        })

    oos_returns = np.array(oos_returns)
    fixed_returns = np.array(fixed_returns)
    bh_returns = np.array(bh_returns)

    return {
        "oos_returns": oos_returns,
        "fixed_returns": fixed_returns,
        "bh_returns": bh_returns,
        "weight_history": weight_history,
    }


def compute_strategy_stats(returns, label):
    """Compute annualized stats for a return series."""
    if len(returns) == 0:
        return {}
    ann_ret = np.mean(returns) * 252
    ann_vol = np.std(returns) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = np.min(dd)
    return {
        "label": label,
        "ann_ret": ann_ret,
        "ann_vol": ann_vol,
        "sharpe": sharpe,
        "max_dd": max_dd,
    }


def print_walkforward_report(wf_results):
    """Print walk-forward optimization results."""
    print(f"\n{'='*80}")
    print(f"  PART 2: WALK-FORWARD OPTIMIZATION ({PRIMARY_TICKER})")
    print(f"{'='*80}")

    wh = wf_results["weight_history"]
    print(f"\n  {'Win':>4s} │ {'Train Period':>22s} │ {'Test Period':>22s} │ {'w_tech':>7s} {'w_macro':>8s} │ {'OOS Sharpe':>10s}")
    print(f"  {'─'*4}─┼─{'─'*22}─┼─{'─'*22}─┼─{'─'*16}─┼─{'─'*10}")
    for w in wh:
        print(f"  {w['window']:>4d} │ {w['train']:>22s} │ {w['test']:>22s} │ {w['w_tech']:>+6.4f} {w['w_macro']:>+7.4f} │ {w['oos_sharpe']:>+9.2f}")

    # Strategy comparison
    stats = [
        compute_strategy_stats(wf_results["oos_returns"], "Walk-Forward Ridge"),
        compute_strategy_stats(wf_results["fixed_returns"], f"Fixed Weight ({FIXED_W_TECH:.0%}/{FIXED_W_MACRO:.0%})"),
        compute_strategy_stats(wf_results["bh_returns"], "Buy & Hold"),
    ]

    print(f"\n  Strategy Comparison (out-of-sample)")
    print(f"  {'Strategy':<28s} │ {'Ann Ret':>8s} {'Ann Vol':>8s} {'Sharpe':>7s} {'Max DD':>8s}")
    print(f"  {'─'*28}─┼─{'─'*34}")
    for s in stats:
        print(f"  {s['label']:<28s} │ {s['ann_ret']:>+7.1%} {s['ann_vol']:>7.1%} {s['sharpe']:>+6.2f} {s['max_dd']:>+7.1%}")


def print_weight_trajectory(wf_results):
    """Print weight trajectory and summary."""
    print(f"\n{'='*80}")
    print(f"  PART 3: WEIGHT TRAJECTORY")
    print(f"{'='*80}")

    wh = wf_results["weight_history"]
    print(f"\n  {'Window':>6s} │ {'Test Period':>22s} │ {'Tech %':>7s} {'Macro %':>8s}")
    print(f"  {'─'*6}─┼─{'─'*22}─┼─{'─'*16}")
    for w in wh:
        print(f"  {w['window']:>6d} │ {w['test']:>22s} │ {w['tech_pct']:>6.1f}% {w['macro_pct']:>7.1f}%")

    avg_tech = np.mean([w["tech_pct"] for w in wh])
    avg_macro = np.mean([w["macro_pct"] for w in wh])
    print(f"\n  Average optimal split:     Tech {avg_tech:.0f}% / Macro {avg_macro:.0f}%")
    print(f"  Current controller default: Tech {FIXED_W_TECH:.0%} / Macro {FIXED_W_MACRO:.0%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print(f"{'='*80}")
    print(f"  SIGNAL ANALYSIS AND WEIGHT OPTIMIZATION")
    print(f"  Period: {START_DATE} to {END_DATE}")
    print(f"  Primary: {PRIMARY_TICKER} | Cross-validation: {', '.join(CROSS_TICKERS)}")
    print(f"{'='*80}")

    # Load data for all tickers
    all_data = {}
    for ticker in [PRIMARY_TICKER] + CROSS_TICKERS:
        print(f"\nLoading signals for {ticker}...")
        df = load_signals(ticker, START_DATE, END_DATE)
        df = compute_forward_returns(df)
        all_data[ticker] = df
        print(f"  {len(df)} total rows ({df['tech_signal'].notna().sum()} with tech signal)")
        time.sleep(1)

    # Part 1: IC Analysis
    ic_results = run_ic_analysis(all_data)
    print_ic_report(ic_results)

    # Part 2: Walk-Forward (primary ticker only)
    print(f"\nRunning walk-forward optimization on {PRIMARY_TICKER}...")
    wf_results = walk_forward_backtest(all_data[PRIMARY_TICKER])
    if wf_results:
        print_walkforward_report(wf_results)
        print_weight_trajectory(wf_results)


if __name__ == "__main__":
    main()
