import os, sys, argparse, pandas as pd, numpy as np
from datetime import datetime
from pathlib import Path

def try_tushare():
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        return None
    try:
        import tushare as ts
        pro = ts.pro_api(token)
        return pro
    except Exception as e:
        print("[warn] Tushare not available:", e)
        return None

def fetch_a_share(pro, start, end):
    # 简化：取沪深300成分作为示例
    print("[info] Fetching CSI300 constituents...")
    idx = pro.index_weight(index_code='399300.SZ', start_date=start.replace("-",""), end_date=end.replace("-",""))
    symbols = sorted(set(idx['con_code'].dropna().tolist()))
    # 拉日线
    all_frames = []
    for ts_code in symbols:
        try:
            df = pro.daily(ts_code=ts_code, start_date=start.replace("-",""), end_date=end.replace("-",""))
            if df.empty: 
                continue
            df['date'] = pd.to_datetime(df['trade_date'])
            df = df.sort_values('date')
            df = df[['date','open','high','low','close','vol']].rename(columns={'vol':'volume'})
            df['symbol'] = ts_code
            all_frames.append(df)
        except Exception as e:
            print("[warn] fetch fail", ts_code, e)
    if not all_frames:
        raise RuntimeError("no A-share data fetched")
    data = pd.concat(all_frames, ignore_index=True)
    return data

def fetch_us_stocks(start, end):
    import yfinance as yf
    # 美股示例篮子
    symbols = ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","AVGO","BRK-B","JPM"]
    frames = []
    for s in symbols:
        try:
            df = yf.download(s, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty: 
                continue
            df = df.rename(columns=str.lower).reset_index().rename(columns={'index':'date'})
            df['symbol'] = s
            frames.append(df[['date','open','high','low','close','volume','symbol']])
        except Exception as e:
            print("[warn] yfinance fail", s, e)
    if not frames:
        raise RuntimeError("no US data fetched")
    return pd.concat(frames, ignore_index=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe", default="CSI300", help="CSI300 or US10")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    pro = try_tushare() if args.universe.upper().startswith("CSI") else None
    if pro is not None:
        data = fetch_a_share(pro, args.start, args.end)
        uniname = "Ashare"
    else:
        print("[info] Using yfinance fallback (US stocks)")
        data = fetch_us_stocks(args.start, args.end)
        uniname = "US"
    # 保存 parquet
    fp = outdir / f"prices_{uniname}.parquet"
    data.to_parquet(fp, index=False)
    print("[ok] saved to", fp)

if __name__ == "__main__":
    main()
