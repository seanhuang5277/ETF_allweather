# -*- coding: utf-8 -*-
"""config.py

集中维护策略与回测通用常量，避免在多个策略脚本中重复硬编码。
仅放轻量、稳定、跨策略共享的配置；与单策略专属参数（例如 HRP_LINKAGE_METHOD_STR）区分。

使用方式:
    from framework.config import (
        TRADING_DAYS_PER_YEAR_SCALAR,
        COST_PER_SIDE_SCALAR,
        RISK_FREE_RATE_SCALAR_ANNUAL,
        DEFAULT_LOOKBACK_YEARS,
        DEFAULT_MIN_DATA_YEARS,
        MIN_CLEAN_DAYS,
    )
"""
from __future__ import annotations

# 交易与市场时间假设
TRADING_DAYS_PER_YEAR_SCALAR: int = 252

# 成本 & 风险假设
COST_PER_SIDE_SCALAR: float = 0.0005          # 单边换手成本 (5bps=0.05%)
RISK_FREE_RATE_SCALAR_ANNUAL: float = 0.02    # 年化无风险利率 (示例)

# 窗口/数据质量相关默认值（策略可根据需要覆盖）
DEFAULT_LOOKBACK_YEARS: int = 3               # 默认滚动窗口年数
DEFAULT_MIN_DATA_YEARS: int = 1               # 资产纳入所需最少有效年数
MIN_CLEAN_DAYS: int = 100                      # 清洗后最少可用样本天数

# 数值稳定相关
EPSILON_NUMERIC: float = 1e-12                # 数值稳定防除零

__all__ = [
    'TRADING_DAYS_PER_YEAR_SCALAR',
    'COST_PER_SIDE_SCALAR',
    'RISK_FREE_RATE_SCALAR_ANNUAL',
    'DEFAULT_LOOKBACK_YEARS',
    'DEFAULT_MIN_DATA_YEARS',
    'MIN_CLEAN_DAYS',
    'EPSILON_NUMERIC',
]
