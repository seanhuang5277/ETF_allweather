# -*- coding: utf-8 -*-
"""
Framework package: 通用绩效计算与可视化工具。
将策略文件中的通用阶段（绩效、绘图）抽离为可复用模块。
"""

from .performance import (
    compute_portfolio_returns_and_equity,
    calculate_performance_metrics,
)
from .plotting import (
    plot_equity_curve,
    plot_weights_history,
    show_all_block,
)
from .logging_config import setup_logging, get_logger
from .config import (
    TRADING_DAYS_PER_YEAR_SCALAR,
    COST_PER_SIDE_SCALAR,
    RISK_FREE_RATE_SCALAR_ANNUAL,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_DATA_YEARS,
    MIN_CLEAN_DAYS,
    EPSILON_NUMERIC,
)
from .load_data import load_returns_and_aum
from .allocation_utils import (
    to_daily_rebalance_dates,
    choose_num_factors,
    get_risk_contributions_daily,
    solve_risk_parity_weights,
    map_factor_to_asset_weights,
    hrp_allocate_recursive_bisect,
)
from .monte_carlo import (
    MonteCarloSimulator,
    MonteCarloResult,
    StrategyRobustnessValidator,
    plot_monte_carlo_results,
    plot_equity_curve_fan,
    generate_summary_report,
)

__all__ = [
    # performance
    'compute_portfolio_returns_and_equity', 'calculate_performance_metrics',
    # plotting
    'plot_equity_curve', 'plot_weights_history', 'show_all_block',
    # logging
    'setup_logging', 'get_logger',
    # config
    'TRADING_DAYS_PER_YEAR_SCALAR', 'COST_PER_SIDE_SCALAR', 'RISK_FREE_RATE_SCALAR_ANNUAL',
    'DEFAULT_LOOKBACK_YEARS', 'DEFAULT_MIN_DATA_YEARS', 'MIN_CLEAN_DAYS', 'EPSILON_NUMERIC',
    # data
    'load_returns_and_aum',
    # allocation
    'to_daily_rebalance_dates', 'choose_num_factors', 'get_risk_contributions_daily',
    'solve_risk_parity_weights', 'map_factor_to_asset_weights', 'hrp_allocate_recursive_bisect',
    # monte carlo
    'MonteCarloSimulator', 'MonteCarloResult', 'StrategyRobustnessValidator',
    'plot_monte_carlo_results', 'plot_equity_curve_fan', 'generate_summary_report',
]
