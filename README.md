# ETF All-Weather Strategy

Macro factor-driven ETF all-weather portfolio strategy research and backtesting framework.

（基于宏观因子驱动的 ETF 全天候风险平价投资策略研究项目）

---

## Research Motivation

This project aims to translate macroeconomic signals into a systematic
multi-asset allocation framework using ETFs as implementation tools.

The focus is not on short-term return optimization, but on robustness,
risk diversification, and portfolio behavior across different market regimes.

---

## Research Framework

The research follows a top-down investment process:

- Identify macroeconomic and policy-related signals
- Construct macro and style factor representations
- Translate factor signals into asset allocation decisions
- Build a diversified portfolio using a risk parity framework
- Evaluate strategy behavior under different market environments

ETFs are used purely as implementation instruments rather than investment targets.

---

## Key Features

- Macro factor-driven asset allocation
- All-weather portfolio construction
- Risk parity weighting scheme
- Liquidity and ETF premium/discount constraints
- Robustness testing via Monte Carlo simulations and parameter perturbation

---

## Repository Structure

# ETF All-Weather Risk Parity Strategy (Macro-Driven)
# 基于宏观因子驱动的 ETF 全天候风险平价策略

## 1. Project Overview / 项目概述
本项目构建了一个能够跨越不同宏观周期的稳健配置模型。不同于传统的风险平价，本策略引入了宏观因子对资产相关性进行动态修正，特别针对 A 股市场的政策环境进行了优化。

## 2. Core Methodology / 核心逻辑
* **Macro Regime Identification:** 利用 Python 搭建宏观因子库，识别增长与通胀的 Regime Switching。
* **Risk Parity Framework:** 基于风险贡献均衡原则分配权重，确保组合在极端波动下的鲁棒性。
* **Reality Bias Correction (核心卖点):** * **Liquidity Friction:** 考虑了 ETF 成分股的流动性冲击成本。
    * **Premium/Discount:** 引入折溢价率作为市场情绪的领先/滞后指标进行仓位微调。

## 3. Tech Stack / 技术栈
* **Language:** Python (Pandas, NumPy, Scipy, Matplotlib)
* **Environment:** 自研量化回测引擎，支持 Monte Carlo 模拟与压力测试。

## 4. Backtest Results / 回测表现
(这里可以贴一张你回测生成的净值曲线图)
* **Sharpe Ratio:** X.X
* **Max Drawdown:** XX%

Detailed backtesting results and robustness analysis are included in the research notebooks
and are available for discussion during interviews.
