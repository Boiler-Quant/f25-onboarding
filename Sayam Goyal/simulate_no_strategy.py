from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- Constants and Setup ---
CACHE_DIR = "data/gamma_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# --- Data Loading Functions ---

def load_cached_markets(months: int = 1) -> List[Dict[str, Any]]:
    """Loads cached market data from a JSON file."""
    path = os.path.join(CACHE_DIR, f"markets_{months}m.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached markets file not found: {path}. "
            f"Run the data caching script for --months {months} first."
        )
    with open(path, "r") as f:
        return json.load(f)


def load_cached_prices_for_token(token_id: str) -> Optional[List[Dict[str, Any]]]:
    """Loads cached price history for a specific token ID."""
    path = os.path.join(CACHE_DIR, f"prices_{token_id}.json")
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


# --- Data Parsing Functions ---

def _parse_timestamp(time_str: Optional[str]) -> Optional[int]:
    """Parses a timestamp string into a Unix timestamp integer."""
    if not time_str:
        return None
    
    # Clean the string to handle different formats
    s = time_str.split(".")[0].replace("+00", "")
    
    for fmt in (None, "%Y-%m-%dT%H:%M:%S"):
        try:
            if fmt is None:
                dt = datetime.fromisoformat(s)
            else:
                dt = datetime.strptime(s, fmt)
            return int(dt.timestamp())
        except (ValueError, TypeError):
            continue
    return None


def parse_markets_to_records(markets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Parses raw market data into structured records for simulation,
    including the market outcome and the price 12 hours before close.
    """
    records: List[Dict[str, Any]] = []
    for market in markets:
        try:
            # 1. Extract token IDs
            clob_token_ids_str = market.get("clobTokenIds")
            if not clob_token_ids_str:
                continue
            token_ids = json.loads(clob_token_ids_str)
            if not isinstance(token_ids, list) or len(token_ids) < 2:
                continue

            # 2. Determine the index for the "YES" outcome
            outcomes_str = market.get("outcomes")
            if not outcomes_str:
                continue
            outcomes = json.loads(outcomes_str)
            yes_index = next((i for i, o in enumerate(outcomes) if isinstance(o, str) and o.lower() == "yes"), 0)

            # 3. Parse the market close time
            closed_ts = _parse_timestamp(market.get("closedTime"))
            if closed_ts is None:
                continue

            # 4. Find the price 12 hours before the close
            yes_token_id = token_ids[yes_index]
            price_history = load_cached_prices_for_token(yes_token_id)
            if not price_history:
                continue

            target_ts = closed_ts - (12 * 3600)  # 12 hours in seconds
            price_at_target = None
            for p in price_history:
                if int(p.get("t", 0)) <= target_ts:
                    price_at_target = p
                else:
                    break
            
            # Default to the earliest price if no price is found before the target time
            chosen_price_point = price_at_target if price_at_target else price_history[0]
            price_12h_before_close = float(chosen_price_point.get("p"))

            # 5. Determine the winning outcome
            outcome_prices_str = market.get("outcomePrices")
            if not outcome_prices_str:
                continue
            final_prices = json.loads(outcome_prices_str)
            winner_index = next((i for i, v in enumerate(final_prices) if str(v) == "1"), None)
            if winner_index is None:
                continue
            
            yes_won = 1 if winner_index == yes_index else 0

            records.append({
                "market_id": market.get("id"),
                "name": market.get("name"),
                "closed_ts": closed_ts,
                "yes_price_12h": price_12h_before_close,
                "yes_won": yes_won,
            })
        except (json.JSONDecodeError, TypeError, ValueError, IndexError, KeyError):
            # Skip any market that has malformed data
            continue
    return records


def get_bucket_label(price: float, binsize: float = 0.05) -> str:
    """Creates a string label for a given price bucket (e.g., '5-10%')."""
    p = max(0.0, min(1.0, float(price)))
    # Ensure index does not go out of bounds for price = 1.0
    max_idx = int(1.0 / binsize) - 1
    idx = min(int(p / binsize), max_idx)
    
    lo = idx * binsize
    hi = lo + binsize
    return f"{int(lo*100)}-{int(hi*100)}%"


# --- Simulation and Analysis ---

def simulate_strategy(
    records: List[Dict[str, Any]], 
    binsize: float = 0.05, 
    min_history: int = 10, 
    stake: float = 1.0
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    cumulative_profit = 0.0
    
    # Track historical data for each price bucket
    bucket_counts: Dict[str, int] = defaultdict(int)
    bucket_yes_wins: Dict[str, int] = defaultdict(int)

    # Process records chronologically
    for r in sorted(records, key=lambda x: x["closed_ts"]):
        p_yes = float(r["yes_price_12h"])
        bucket = get_bucket_label(p_yes, binsize)
        
        # Get historical performance for the current bucket
        past_count = bucket_counts[bucket]
        past_yes_wins = bucket_yes_wins[bucket]
        p_hist_yes = (past_yes_wins / past_count) if past_count > 0 else None

        # --- Decision Logic ---
        decision = False
        profit = 0.0
        expected_profit = 0.0
        
        # Bet on NO if implied probability is higher than historical
        if past_count >= min_history and p_hist_yes is not None and p_yes > p_hist_yes:
            decision = True
            expected_profit = stake * (p_yes - p_hist_yes)
            
            # Profit calculation for a $1 stake on NO
            if r["yes_won"] == 0:  # NO wins
                profit = stake
            else:  # YES wins, NO loses
                profit = -stake
            cumulative_profit += profit

        rows.append({
            "closed_ts": r["closed_ts"],
            "market_id": r["market_id"],
            "name": r.get("name"),
            "bucket": bucket,
            "yes_price_12h": p_yes,
            "p_hist_yes": p_hist_yes if p_hist_yes is not None else np.nan,
            "decision": decision,
            "expected_profit": expected_profit,
            "profit": profit,
            "cumulative_profit": cumulative_profit,
        })

        # Update historical data with the outcome of the current market
        bucket_counts[bucket] += 1
        bucket_yes_wins[bucket] += int(r["yes_won"])

    return pd.DataFrame(rows)


def summarize_and_plot(df: pd.DataFrame, out_prefix: str) -> None:
    if df.empty:
        print("No simulation rows to plot.")
        return

    # Save raw simulation data
    csv_path = out_prefix + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved simulation CSV to {csv_path}")

    df["time"] = df["closed_ts"].apply(datetime.fromtimestamp)
    trades = df[df["decision"]].copy()

    # --- Plotting ---
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Cumulative PnL over time
    plt.figure(figsize=(12, 6))
    plt.plot(df["time"], df["cumulative_profit"], label="Cumulative PnL")
    plt.title("Cumulative Profit and Loss Over Time")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Profit ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_prefix}_cumulative_pnl.png")

    if trades.empty:
        print("No trades were placed; skipping trade-level analysis.")
    else:
        trades = trades.sort_values("closed_ts")
        trades["cum_expected"] = trades["expected_profit"].cumsum()
        trades["cum_realized"] = trades["profit"].cumsum()

        # 2. Expected vs. Realized PnL
        plt.figure(figsize=(12, 6))
        plt.plot(trades["time"], trades["cum_expected"], label="Expected Cumulative PnL")
        plt.plot(trades["time"], trades["cum_realized"], label="Realized Cumulative PnL", linestyle='--')
        plt.title("Cumulative Expected vs. Realized Profit")
        plt.xlabel("Date")
        plt.ylabel("Profit ($)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_expected_vs_realized.png")

        # 3. Drawdown
        peak = trades["cum_realized"].cummax()
        drawdown = peak - trades["cum_realized"]
        plt.figure(figsize=(12, 5))
        plt.fill_between(trades["time"], drawdown, color="r", alpha=0.3)
        plt.plot(trades["time"], drawdown, color="r", alpha=0.8)
        plt.title("Portfolio Drawdown")
        plt.xlabel("Date")
        plt.ylabel("Drawdown ($)")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_drawdown.png")
        
        # 4. Per-trade profit distribution
        plt.figure(figsize=(10, 5))
        plt.hist(trades["profit"].dropna(), bins=20, color="skyblue", edgecolor="black")
        plt.title("Per-Trade Profit Distribution")
        plt.xlabel("Profit ($)")
        plt.ylabel("Number of Trades")
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_profit_histogram.png")

        # 5. Per-bucket performance
        bucket_ev = trades.groupby("bucket").agg(
            n_trades=("profit", "count"),
            avg_real_profit=("profit", "mean"),
            avg_expected_profit=("expected_profit", "mean")
        ).reset_index()

        plt.figure(figsize=(12, 6))
        x = np.arange(len(bucket_ev))
        width = 0.35
        plt.bar(x - width/2, bucket_ev["avg_expected_profit"], width, label="Avg. Expected Profit")
        plt.bar(x + width/2, bucket_ev["avg_real_profit"], width, label="Avg. Realized Profit")
        plt.xticks(x, bucket_ev["bucket"], rotation=45, ha="right")
        plt.ylabel("Average Profit per Trade ($)")
        plt.title("Per-Bucket Expected vs. Realized Profit")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{out_prefix}_bucket_performance.png")

        print("\nAll plots saved.")

    # --- Final Summary Stats ---
    total_trades = len(trades)
    total_profit = df["cumulative_profit"].iloc[-1] if not df.empty else 0.0
    avg_profit = trades["profit"].mean() if total_trades > 0 else 0.0
    win_rate = (trades["profit"] > 0).mean() if total_trades > 0 else 0.0

    print("\n--- Simulation Summary ---")
    print(f"  Total trades placed:      {total_trades}")
    print(f"  Total profit:             ${total_profit:.2f}")
    print(f"  Average profit per trade: ${avg_profit:.4f}")
    print(f"  Win rate:                 {win_rate:.2%}")
    print("--------------------------\n")


# --- Main Execution ---

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate a NO-bet strategy using cached market and price data."
    )
    parser.add_argument("--months", type=int, default=1, help="Number of months of cached data to use.")
    parser.add_argument("--min-history", type=int, default=10, help="Minimum number of markets in a bucket to place a trade.")
    parser.add_argument("--stake", type=float, default=1.0, help="The amount to stake on each trade.")
    parser.add_argument("--binsize", type=float, default=0.05, help="The size of the price probability buckets.")
    args = parser.parse_args()

    print(f"Loading cached markets for {args.months} month(s)...")
    markets = load_cached_markets(months=args.months)
    
    print("Parsing market data into records...")
    records = parse_markets_to_records(markets)
    if not records:
        print("No valid records could be parsed from the cached market data.")
        return
    print(f"Successfully parsed {len(records)} records.")

    print(f"Running simulation with min_history={args.min_history}, stake=${args.stake}, binsize={args.binsize}...")
    df = simulate_strategy(
        records, 
        binsize=args.binsize, 
        min_history=args.min_history, 
        stake=args.stake
    )
    
    out_prefix = os.path.join(CACHE_DIR, f"simulation_m{args.months}_h{args.min_history}_s{args.stake}")
    print(f"Simulation complete. Saving results with prefix: {out_prefix}")
    summarize_and_plot(df, out_prefix=out_prefix)


if __name__ == "__main__":
    main()