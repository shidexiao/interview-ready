import argparse, pandas as pd, numpy as np
from pathlib import Path

def compute_factors(df: pd.DataFrame, window: int=126) -> pd.DataFrame:
    df = df.copy()
    df['ret'] = df.groupby('symbol')['close'].pct_change()
    # 动量：过去 window 日累计收益
    df['mom'] = df.groupby('symbol')['close'].apply(lambda x: x.pct_change(periods=window))
    # 低波：过去 window 日收益标准差的负值（越小越好）
    df['vol'] = df.groupby('symbol')['ret'].rolling(window).std().reset_index(level=0, drop=True)
    df['lowvol'] = -df['vol']
    # 标准化
    for col in ['mom','lowvol']:
        df[col] = df.groupby('date')[col].transform(lambda x: (x - x.mean())/ (x.std(ddof=0) + 1e-9))
    # 综合评分（等权）
    df['score'] = df[['mom','lowvol']].mean(axis=1)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", default="data/prices_Ashare.parquet")
    ap.add_argument("--window", type=int, default=126)
    ap.add_argument("--outfile", default="data/factors.parquet")
    args = ap.parse_args()

    if not Path(args.infile).exists():
        alt = "data/prices_US.parquet"
        if Path(alt).exists():
            args.infile = alt
        else:
            raise SystemExit("No price parquet found. Run fetch_data.py first.")

    df = pd.read_parquet(args.infile)
    out = compute_factors(df, args.window)
    out.to_parquet(args.outfile, index=False)
    print("[ok] factors saved to", args.outfile)

if __name__ == "__main__":
    main()
