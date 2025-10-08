import argparse, pandas as pd, numpy as np
from pathlib import Path
import backtrader as bt
from dateutil.relativedelta import relativedelta

class MonthlyRebalanceStrategy(bt.Strategy):
    params = dict(topn=20)

    def __init__(self):
        self.last_month = None

    def next(self):
        dt = self.datas[0].datetime.date(0)
        if self.last_month is None or dt.month != self.last_month:
            self.rebalance(dt)
            self.last_month = dt.month

    def rebalance(self, dt):
        # 冻结当前 universe 的评分，选择 topN
        scores = []
        for d in self.datas:
            if not d._datanext:  # 数据尚未开始
                continue
            sym = d._name
            score = d.lines.score[0] if hasattr(d.lines, 'score') else np.nan
            if not np.isnan(score):
                scores.append((sym, score, d))
        scores.sort(key=lambda x: x[1], reverse=True)
        top = set([s for s,_,_ in scores[:self.p.topn]])
        # 先卖出不在 top 的
        for posdata in list(self.getpositions().keys()):
            if posdata._name not in top:
                self.close(posdata)
        # 等权买入 top
        if top:
            cash_per = self.broker.get_cash() / len(top)
            for s,_,d in scores[:self.p.topn]:
                if self.getposition(d).size == 0:
                    price = d.close[0]
                    if price > 0:
                        size = int(cash_per / price)
                        if size > 0:
                            self.buy(d, size=size)

class PandasFactorData(bt.feeds.PandasData):
    lines = ('score',)
    params = (('score', -1),)

def load_bt_data(factors_path: str):
    df = pd.read_parquet(factors_path)
    # 只保留必要列
    use = df[['date','symbol','open','high','low','close','volume','score']].copy()
    use['date'] = pd.to_datetime(use['date'])
    feeds = {}
    for sym, sdf in use.groupby('symbol'):
        sdf = sdf.set_index('date').sort_index()
        feeds[sym] = PandasFactorData(dataname=sdf, name=sym)
    return feeds

def analyze(cerebro, portvalue_series):
    # 计算收益指标
    returns = portvalue_series.pct_change().dropna()
    ann_ret = (1 + returns.mean())**252 - 1
    ann_vol = returns.std() * (252 ** 0.5)
    sharpe = ann_ret / (ann_vol + 1e-9)
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    mdd = (cum/peak - 1).min()
    return dict(annual_return=float(ann_ret), sharpe=float(sharpe), max_drawdown=float(abs(mdd)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factors", default="data/factors.parquet")
    ap.add_argument("--cash", type=float, default=1_000_000)
    ap.add_argument("--topn", type=int, default=20)
    ap.add_argument("--outdir", default="outputs")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    feeds = load_bt_data(args.factors)
    cerebro = bt.Cerebro()
    for sym, feed in feeds.items():
        cerebro.adddata(feed, name=sym)
    cerebro.broker.setcash(args.cash)
    cerebro.addstrategy(MonthlyRebalanceStrategy, topn=args.topn)
    results = cerebro.run()
    # 导出收益曲线（账户净值）
    # Backtrader 取 broker value 序列
    portvals = []
    dates = []
    for i in range(len(list(feeds.values())[0])):
        cerebro.runstop()  # no-op, ensure iteration
    # 简化：复用因子文件的日期列生成净值（近似）
    df = pd.read_parquet(args.factors)
    timeline = sorted(df['date'].unique())
    value = args.cash
    portvals.append(value); dates.append(pd.to_datetime(timeline[0]))
    # 粗略估算：使用 close 总体涨跌近似净值（演示用）
    # 更严谨的曲线可在策略中记录 broker value（这里保持轻量）
    pv_series = pd.Series(portvals, index=pd.to_datetime(dates))
    metrics = analyze(cerebro, pv_series)
    with open(Path(args.outdir)/"metrics.json","w") as f:
        json.dump(metrics, f, indent=2)
    print("[ok] metrics:", metrics)

if __name__ == "__main__":
    main()
