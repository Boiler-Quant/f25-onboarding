```
# Hybrid Trading Strategy: Adapting to Market Regimes

## Overview

This repository contains the code for a **hybrid trading strategy** that intelligently adapts to different market conditions. Instead of relying on a single approach, the system dynamically switches between a **trend-following** strategy and a **mean-reversion** strategy.

The core principle is to use the right tool for the right job, ensuring the strategy performs well in both trending and range-bound markets.

## How It Works

The strategy's decision-making process is based on a single key indicator: the **Average True Range (ATR)**.

1.  **Regime Detection:**
    * The strategy first calculates the ratio of the ATR to the asset's closing price (`ATR / Close`).
    * This ratio acts as a volatility filter, helping to identify the current market regime.

2.  **Strategy Execution:**
    * **If `ATR / Close` is above a predefined threshold:** The market is considered **trending and volatile**. The strategy activates a **trend-following** approach using **EMA crossovers**.
        * **Buy Signal:** The short-term EMA crosses above the long-term EMA.
        * **Sell Signal:** The short-term EMA crosses below the long-term EMA.

    * **If `ATR / Close` is below the threshold:** The market is considered **calm and range-bound**. The strategy activates a **mean-reversion** approach using **Bollinger Bands**.
        * **Buy Signal:** Price moves below the lower Bollinger Band.
        * **Sell Signal:** Price moves above the upper Bollinger Band.
        * **Exit:** Positions are closed when the price reverts to the middle band (SMA).

## Key Components

* **Average True Range (ATR):** Used to measure market volatility and determine the current regime.
* **Exponential Moving Averages (EMAs):** Used for the trend-following component.
* **Bollinger Bands:** Used for the mean-reversion component.

## Goal

The primary objective of this hybrid strategy is to maximize performance by:

* Capturing large directional moves during trending periods.
* Profiting from price oscillations during calm, ranging periods.
* Avoiding the common pitfall of using a trend-following strategy in a sideways market or vice-versa.
