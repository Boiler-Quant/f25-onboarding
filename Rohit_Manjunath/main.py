import argparse

import pandas as pd
import numpy as np
import yfinance as yf
from backtesting import Strategy
from backtesting.lib import FractionalBacktest

# Helpers
def _as_series(x):
    if isinstance(x, pd.Series):
        return x
    return pd.Series(np.asarray(x, dtype=float))

def ema(x, span: int):
    s = _as_series(x)
    return s.ewm(span=span, adjust=False).mean().to_numpy()

def rsi(x, period: int = 14):
    s = _as_series(x)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = (-delta).clip(lower=0.0)
    roll_up = up.rolling(period).mean()
    roll_down = down.rolling(period).mean()
    rs = roll_up / (roll_down.replace(0.0, np.nan))
    out = 100.0 - (100.0 / (1.0 + rs))
    return out.to_numpy()

def true_range(h, l, c):
    h = _as_series(h)
    l = _as_series(l)
    c = _as_series(c)

    prev_close = c.shift(1)
    tr = pd.concat([h - l, (h - prev_close).abs(), (l - prev_close).abs()], axis=1).max(axis=1)
    
    return tr

def atr(h, l, c, period: int = 14):
    tr = true_range(h, l, c)
    return tr.rolling(period).mean().to_numpy()

def rolling_mean(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window).mean()

def _body(high, low, open_, close):
    return abs(close - open_)

def _spread(high, low):
    return (high - low)

def _upper_wick(high, open_, close):
    return high - max(open_, close)

def _lower_wick(low, open_, close):
    return min(open_, close) - low

def vsa_long_signal(df, i, vol_win=20, atr_win=14):
    if i < max(vol_win, atr_win) + 1:
        return False

    row = df.iloc[i]
    prev = df.iloc[i-1]

    vol_ma = df['Volume'].rolling(vol_win).mean().iloc[i]
    atr_val = df['ATR'].iloc[i]

    if pd.isna(vol_ma) or pd.isna(atr_val) or atr_val <= 0:
        return False

    o, h, l, c, v = row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
    sp = _spread(h, l)
    ub = _upper_wick(h, o, c)
    lb = _lower_wick(l, o, c)
    body = _body(h, l, o, c)

    # 1) Stopping volume
    #   - Down bar (c < o)
    #   - Wide spread: sp > 1.3 * ATR
    #   - High volume: v > 1.5 * vol_ma
    #   - Close in top half of bar: c > l + 0.5*sp (buying pressure)
    stopping_volume = (c < o) and (sp > 1.3 * atr_val) and (v > 1.5 * vol_ma) and (c > l + 0.5 * sp)

    # 2) Shakeout/Test:
    #   - Long lower wick: lb > 0.55 * sp
    #   - Body not huge: body < 0.6 * sp
    #   - For a "test" we often want relatively lower volume than recent, here < 0.9 * vol_ma
    shakeout = (lb > 0.55 * sp) and (body < 0.6 * sp) and (v < 0.9 * vol_ma)

    return stopping_volume or shakeout

def vsa_short_signal(df, i, vol_win=20, atr_win=14):
    if i < max(vol_win, atr_win) + 1:
        return False

    row = df.iloc[i]

    vol_ma = df['Volume'].rolling(vol_win).mean().iloc[i]
    atr_val = df['ATR'].iloc[i]

    if pd.isna(vol_ma) or pd.isna(atr_val) or atr_val <= 0:
        return False

    o, h, l, c, v = row['Open'], row['High'], row['Low'], row['Close'], row['Volume']
    sp = _spread(h, l)
    ub = _upper_wick(h, o, c)
    body = _body(h, l, o, c)

    # 1) No demand:
    #   - Up bar (c > o)
    #   - Narrow spread: sp < 0.7 * ATR
    #   - Low volume: v < 0.8 * vol_ma
    no_demand = (c > o) and (sp < 0.7 * atr_val) and (v < 0.8 * vol_ma)

    # 2) Upthrust (supply):
    #   - Long upper wick: ub > 0.55 * sp
    #   - Wide spread: sp > 1.2 * ATR
    #   - Close in lower half: c < l + 0.5*sp
    #   - High volume: v > 1.4 * vol_ma
    upthrust = (ub > 0.55 * sp) and (sp > 1.2 * atr_val) and (c < l + 0.5 * sp) and (v > 1.4 * vol_ma)

    return no_demand or upthrust

def vsa_zscore(high, low, volume, window: int = 200, min_periods: int = 50):
    """Z-score of (spread minus expected spread given volume), rolling OLS."""
    h = _as_series(high)
    l = _as_series(low)
    v = _as_series(volume)
    sp = (h - l)

    # Rolling means
    mV = v.rolling(window, min_periods=min_periods).mean()
    mS = sp.rolling(window, min_periods=min_periods).mean()

    # Rolling covariance and variance
    covVS = (v * sp).rolling(window, min_periods=min_periods).mean() - mV * mS
    varV  = (v * v).rolling(window, min_periods=min_periods).mean() - mV * mV
    slope = covVS / varV.replace(0.0, np.nan)
    intercept = mS - slope * mV

    expected = intercept + slope * v
    resid = sp - expected

    # Standardize residual with rolling stats
    r_mean = resid.rolling(window, min_periods=min_periods).mean()
    r_std  = resid.rolling(window, min_periods=min_periods).std().replace(0.0, np.nan)
    z = (resid - r_mean) / r_std
    return z.to_numpy()

# The Strat
class VSAMARSI(Strategy):
    params = dict(
        ema_fast=20,
        ema_slow=50,
        rsi_period=14,
        rsi_long_min=50.0,
        rsi_long_max=70.0,
        rsi_short_max=50.0,
        rsi_short_min=30.0,
        atr_period=14,
        swing_lookback=5,
        risk_reward=2.0,
        use_trailing=False,
        vsa_window=200,
        vsa_z_thresh=2.0,
    )

    def init(self):
        close = self.data.Close
        high  = self.data.High
        low   = self.data.Low
        vol   = self.data.Volume

        ef = self.params['ema_fast']
        es = self.params['ema_slow']
        rp = self.params['rsi_period']
        ap = self.params['atr_period']
        vw = self.params['vsa_window']

        self.ema_fast = self.I(ema, close, ef, name="EMA_fast")
        self.ema_slow = self.I(ema, close, es, name="EMA_slow")
        self.rsi_val  = self.I(rsi, close, rp, name="RSI")
        self.atr_val  = self.I(atr,  high,  low,   close, ap, name="ATR")
        self.vsa_z    = self.I(vsa_zscore, high, low, vol, vw, name="VSA_Z")

        df = self.data.df.copy() 
        df['ATR'] = pd.Series(self.atr_val, index=df.index)
        self._df = df

    def next(self):
        i = len(self.data) - 1
        df = self._df

        price = self.data.Close[-1]
        ema_f = self.ema_fast[-1]
        ema_s = self.ema_slow[-1]
        rsi_v = float(self.rsi_val[-1])

        trend_long  = (price > ema_s) and (ema_f > ema_s)
        trend_short = (price < ema_s) and (ema_f < ema_s)

        vsa_l = vsa_long_signal(df, i)
        vsa_s = vsa_short_signal(df, i)

        z = float(self.vsa_z[-1])
        vsa_extreme = np.isfinite(z) and (abs(z) >= self.params['vsa_z_thresh'])

        # long_ok  = trend_long  and (rsi_v > self.params['rsi_long_min'])  and (rsi_v < self.params['rsi_long_max'])  and vsa_l
        # short_ok = trend_short and (rsi_v < self.params['rsi_short_max']) and (rsi_v > self.params['rsi_short_min']) and vsa_s

        long_ok  = trend_long  and (rsi_v > self.params['rsi_long_min'])  and (rsi_v < self.params['rsi_long_max'])  and vsa_extreme
        short_ok = trend_short and (rsi_v < self.params['rsi_short_max']) and (rsi_v > self.params['rsi_short_min']) and vsa_extreme

        def recent_swing_low(k):
            lows = self.data.Low[-k-1:-1]
            return np.min(lows) if len(lows) else self.data.Low[-1]

        def recent_swing_high(k):
            highs = self.data.High[-k-1:-1]
            return np.max(highs) if len(highs) else self.data.High[-1]

        if long_ok and not self.position.is_long:
            sl = recent_swing_low(self.params['swing_lookback'])
            if sl is not None and sl < price:
                risk = price - sl
                tp = price + self.params['risk_reward'] * risk
                self.position.close()
                self.buy(sl=sl, tp=tp)

        if short_ok and not self.position.is_short:
            sh = recent_swing_high(self.params['swing_lookback'])
            if sh is not None and sh > price:
                risk = sh - price
                tp = price - self.params['risk_reward'] * risk
                self.position.close()
                self.sell(sl=sh, tp=tp)

        if self.params['use_trailing'] and self.position:
            if self.position.is_long and self.data.Close[-1] < self.ema_fast[-1]:
                self.position.close()
            if self.position.is_short and self.data.Close[-1] > self.ema_fast[-1]:
                self.position.close()

# Loading Relevant Data
def load_ohlcv(symbol: str, interval: str = "15m", period: str = "60d") -> pd.DataFrame:
    df = yf.download(tickers=symbol, interval=interval, period=period,
                     progress=False, group_by='ticker', auto_adjust=False, threads=True)
    
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {symbol} {interval=} {period=}.")

    if isinstance(df.columns, pd.MultiIndex):
        lower = [str(c).lower() for c in df.columns.get_level_values(0)]
        upper = [str(c).lower() for c in df.columns.get_level_values(1)]
        ohlc = {'open','high','low','close','volume'}
        if set(lower) & ohlc == ohlc:
            df = df.droplevel(1, axis=1)
        elif set(upper) & ohlc == ohlc:
            df = df.droplevel(0, axis=1)
        elif symbol in df.columns.get_level_values(0):
            df = df.xs(symbol, axis=1, level=0, drop_level=True)
        elif symbol in df.columns.get_level_values(1):
            df = df.xs(symbol, axis=1, level=1, drop_level=True)
        else:
            df.columns = df.columns.get_level_values(-1)
    df = df.rename(columns=str.title)
    df = df[['Open','High','Low','Close','Volume']].dropna()
    if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df = df.sort_index()
    return df

# Run Backtest
def run_backtest(symbol: str, interval: str, period: str, cash: float = 100_000.0, commission: float = 0.01, plot: bool = False):
    df = load_ohlcv(symbol, interval, period)
    bt = FractionalBacktest(df, VSAMARSI, cash=cash, commission=commission, trade_on_close=False, hedging=False, exclusive_orders=True)
    stats = bt.run()
    print(stats)
    if plot:
        bt.plot(open_browser=False)

def main():
    parser = argparse.ArgumentParser(description="Backtest VSA + MA + RSI intraday strategy.")
    parser.add_argument("--symbol", type=str, default="ETH-USD", help="Ticker symbol (default: BTC-USD)")
    parser.add_argument("--interval", type=str, default="15m", help="Chart interval (e.g., 1m,2m,5m,15m,30m,60m,90m,1h,1d)")
    parser.add_argument("--period", type=str, default="60d", help="Lookback period (e.g., 7d,30d,60d,1y,5y,max)")
    parser.add_argument("--cash", type=float, default=100_000.0, help="Starting cash")
    parser.add_argument("--commission", type=float, default=0.0005, help="Commission per trade (fraction)")
    parser.add_argument("--plot", action="store_true", help="Plot results")
    args = parser.parse_args()

    run_backtest(
        symbol=args.symbol,
        interval=args.interval,
        period=args.period,
        cash=args.cash,
        commission=args.commission,
        plot=args.plot,
    )

if __name__ == "__main__":
    main()
