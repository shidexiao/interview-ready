# A股/美股 多因子选股与轮动（Momentum + LowVol）
**目标**：用最少的依赖快速落地一个可跑通的多因子策略（技术因子为主），支持月度调仓与回测。

## 特性
- 数据源优先：Tushare（需自行申请 token），无法使用时自动退化到 `yfinance`（美股示例）。
- 因子：
  - 动量（过去 126 个交易日收益率）
  - 低波动（过去 126 日收益率标准差的负值）
- 回测：每月最后一个交易日等权买入 Top N，持有 1 个月。

## 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 设置环境变量（如使用 Tushare）
export TUSHARE_TOKEN="你的token"

# 1) 下载数据
python fetch_data.py --universe CSI300 --start 2018-01-01 --end 2025-10-01

# 2) 计算因子并生成打分
python factors.py --window 126

# 3) 回测（每月调仓，持仓 20）
python backtest.py --topn 20
```

## 目录
- `fetch_data.py`：优先用 Tushare 拉取 A 股；失败则用 yfinance 拉美股示例（AAPL, MSFT, NVDA...）。
- `factors.py`：计算动量、低波动，生成综合评分。
- `backtest.py`：Backtrader 月度调仓策略，输出年化、夏普、回撤。
- `data/`：缓存的行情数据（parquet）。
- `outputs/`：回测日志与图表。

> 提示：如果你要上简历/作品集，可以把你的运行截图（收益曲线、指标表）放在 `outputs/` 中。
