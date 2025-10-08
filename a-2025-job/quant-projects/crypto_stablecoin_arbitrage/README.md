# 稳定币/跨交易所套利（USDT/USDC + 可扩展到主流币）

**目标**：搭建一个可运行的实时价差监控与“模拟执行”系统；支持对接 Binance/OKX/Bybit API，后续可无缝切换到实盘或各家 Testnet。

## 快速开始
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 环境变量（按需）
export BINANCE_KEY=xxx
export BINANCE_SECRET=yyy
export OKX_KEY=...
export OKX_SECRET=...
export OKX_PASSPHRASE=...

# 运行实时监控
python monitor_arbitrage.py --symbols USDT/USDC,BTC/USDT --threshold 0.001

# 回看日志与机会快照
ls outputs/
```

## 说明
- 不直接下单，默认“模拟成交”，记录到 `outputs/opportunities.csv`，便于复盘。
- 采用 `asyncio` 并发拉取/订阅，多交易所汇总。
- 阈值：默认 10bp（0.1%），可按滑点与手续费调整。
- 扩展：可接入 Telegram Bot、持仓/余额 API、跨链/跨所资金调度。
