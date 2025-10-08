import asyncio, ccxt.async_support as ccxt, pandas as pd, numpy as np, argparse, time
from pathlib import Path
from datetime import datetime

EXCHANGES = ['binance','okx','bybit']

async def fetch_ticker_safe(ex, symbol):
    try:
        t = await ex.fetch_ticker(symbol)
        return dict(exchange=ex.id, symbol=symbol, bid=t.get('bid'), ask=t.get('ask'), ts=time.time())
    except Exception as e:
        return dict(exchange=ex.id, symbol=symbol, error=str(e), ts=time.time())

async def run_monitor(symbols, threshold, interval, outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    # 初始化交易所
    ex_objs = []
    for name in EXCHANGES:
        cls = getattr(ccxt, name)
        ex = cls({'enableRateLimit': True})
        ex_objs.append(ex)
    try:
        while True:
            rows = []
            for s in symbols:
                tasks = [fetch_ticker_safe(ex, s) for ex in ex_objs]
                res = await asyncio.gather(*tasks, return_exceptions=False)
                rows.extend(res)
            df = pd.DataFrame(rows)
            now = datetime.utcnow().isoformat()
            # 计算跨所可实现价差（买低卖高）
            for s in symbols:
                sdf = df[(df['symbol']==s) & df['bid'].notna() & df['ask'].notna()]
                if len(sdf) < 2: 
                    continue
                best_bid = sdf.loc[sdf['bid'].idxmax()]
                best_ask = sdf.loc[sdf['ask'].idxmin()]
                spread = (best_bid['bid'] - best_ask['ask']) / best_ask['ask']
                if spread >= threshold:
                    line = {
                        'time': now, 'symbol': s, 'buy_ex': best_ask['exchange'], 'buy_px': best_ask['ask'],
                        'sell_ex': best_bid['exchange'], 'sell_px': best_bid['bid'], 'spread': float(spread)
                    }
                    print("[opportunity]", line)
                    out = Path(outdir)/"opportunities.csv"
                    hdr = not out.exists()
                    pd.DataFrame([line]).to_csv(out, mode='a', header=hdr, index=False)
            await asyncio.sleep(interval)
    finally:
        for ex in ex_objs:
            await ex.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="USDT/USDC,BTC/USDT", help="comma-separated symbols")
    ap.add_argument("--threshold", type=float, default=0.001, help="min spread trigger (e.g., 0.001=10bp)")
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    asyncio.run(run_monitor(symbols, args.threshold, args.interval, args.outdir))

if __name__ == "__main__":
    main()
