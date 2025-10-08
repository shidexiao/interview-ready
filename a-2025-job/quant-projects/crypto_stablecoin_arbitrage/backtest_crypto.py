# 简化版：用单交易所的K线模拟做市/价差触发（教学用途）
import argparse, pandas as pd, numpy as np
from pathlib import Path

def simple_threshold_backtest(df, up=0.003, down=0.003):
    # 价格越过均线一定阈值时开/平仓（示例策略）
    df = df.copy()
    df['ma'] = df['close'].rolling(50).mean()
    pos = 0
    pnl = 0.0
    equity = [1.0]
    for i in range(50, len(df)):
        px = df['close'].iloc[i]
        ma = df['ma'].iloc[i]
        if pos == 0:
            if (px - ma)/ma > up:
                pos = 1; entry = px
            elif (ma - px)/ma > down:
                pos = -1; entry = px
        else:
            if (pos==1 and px<ma) or (pos==-1 and px>ma):
                pnl += (px-entry)/entry * pos
                pos = 0
        equity.append(1.0 + pnl)
    return pd.Series(equity, index=df.index[-len(equity):])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=False, default=None, help="use your own kline csv with columns: time,open,high,low,close,volume")
    ap.add_argument("--out", default="outputs/backtest_equity.csv")
    args = ap.parse_args()

    if args.csv is None:
        # 生成模拟序列
        idx = pd.date_range("2024-01-01", periods=200, freq="H")
        price = 100*np.exp(np.cumsum(np.random.normal(0, 0.01, size=len(idx))))
        df = pd.DataFrame({"time": idx, "open": price, "high": price*1.005, "low": price*0.995, "close": price, "volume": 1000})
    else:
        df = pd.read_csv(args.csv, parse_dates=['time'])

    eq = simple_threshold_backtest(df)
    Path("outputs").mkdir(exist_ok=True, parents=True)
    eq.to_csv(args.out)
    print("[ok] saved backtest equity curve to", args.out)

if __name__ == "__main__":
    main()
