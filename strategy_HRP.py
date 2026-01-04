# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_HRP.py
@Author : Sean
@Date   : 2025-11-09

@Description:
Hierarchical Risk Parity (HRP) 策略回测脚本。
核心思想：利用资产间相关性结构的层次聚类 + 准对角化排序 + 递归二分风险分配，
在不显式求协方差矩阵逆的情况下，实现稳定、分散的风险配置。
与传统 MVO/RP 相比，HRP 更抗噪声、更鲁棒，适用于中高维资产集合。
"""

# ==============================================================================
# 阶段 0: 导入核心库
# ==============================================================================
import pandas as pd
import numpy as np
from sklearn.covariance import OAS
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
from framework.performance import compute_portfolio_returns_and_equity, calculate_performance_metrics
from framework.plotting import plot_equity_curve, plot_weights_history
from framework.logging_config import setup_logging, get_logger
from framework.config import (
    TRADING_DAYS_PER_YEAR_SCALAR,
    COST_PER_SIDE_SCALAR,
    RISK_FREE_RATE_SCALAR_ANNUAL,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_DATA_YEARS,
    MIN_CLEAN_DAYS,
    EPSILON_NUMERIC,
)
from framework.load_data import (
    compute_log_returns,
    load_etf_index_returns_by_category,
)
from framework.allocation_utils import hrp_allocate_recursive_bisect

# 初始化统一日志
setup_logging(level="INFO") #  'DEBUG'/'INFO'
logger = get_logger("strategy.HRP") 


# ==============================================================================
# 阶段 1: 数据加载与预处理
# ==============================================================================
"""[已注释] 顶层数据加载逻辑由 run_strategy() 统一处理，以下代码仅保留参考。
# try:
#     # 新实现：使用框架数据加载
#     all_simple_returns_df, all_log_returns_df, market_aum_daily_df, all_assets = load_returns_and_aum(
#         'index_daily_simple_returns.csv',
#         'proxy_etfs_aum_monthly.csv',
#     )
#     n_assets = len(all_assets)
#     logger.info(f"数据加载完成 | 资产数: {n_assets}")
# except FileNotFoundError as e:
#     logger.error(f"文件缺失: {e}")
#     exit()
# except Exception as e:
#     logger.error(f"数据加载错误: {e}")
#     exit()
"""



# ==============================================================================
# 阶段 2: 全局参数与策略设定
# ==============================================================================
STRATEGY_MODE = 'HRP'

# 与旧变量名对齐，引用默认配置值（策略可按需覆盖）
LOOKBACK_YEARS = DEFAULT_LOOKBACK_YEARS
MIN_DATA_YEARS = DEFAULT_MIN_DATA_YEARS

# --- 2.1 HRP 策略专用参数（与 BL_RB 命名习惯对齐） ---
# linkage 方法、是否使用最优叶序、分配口径（'var'|'std'）、可选对角波动地板
HRP_LINKAGE_METHOD_STR = 'ward'
HRP_USE_OPTIMAL_ORDERING_BOOL = True
# 改为用“标准差倒数”进行二分分配（放弃方差倒数）
HRP_ALLOCATION_METRIC_STR = 'std'
HRP_DIAG_VOL_FLOOR_SCALAR = None       # 例如 0.0001 表示日波动地板为 1bp；默认 None 不启用

# ==============================================================================
# 阶段 3: HRP 核心函数定义
# ==============================================================================

## HRP 聚类与递归二分已迁移至 framework.allocation_utils.hrp_allocate_recursive_bisect





# ============================================================================
# 统一接口：run_strategy
# ----------------------------------------------------------------------------
def run_strategy(
    *,
    categories: list[str] | None = None,
    data_dir: str = 'data',
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    # HRP 专属参数
    linkage_method: str = 'ward',
    use_optimal_ordering: bool = True,
    diag_vol_floor: float | None = None,
    debug: bool = False,
    auto_plot: bool = False,
) -> dict:
    """运行 HRP 策略（使用 config 中 F='Y' 的指数集合），返回标准化结果字典。

    - 数据：`data/index_daily_simple_returns.csv` + `config/config_export_etf_and_index_rawdata.csv`
    - 资产：仅使用 F='Y' 且属于所选类别的 Index 列
    - 分配：递归二分时按“标准差倒数”分配（放弃方差倒数）
    """

    # 1) 加载（仅 F='Y'）的 ETF/Index（按类别）
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(
        data_dir=data_dir,
        returns_index_col='日期',
    )

    # 2) 选类别并合并指数为资产池
    use_cats = list(idx_by_cat.keys()) if categories is None else list(categories)
    simple_df = None
    for c in use_cats:
        df = idx_by_cat.get(c)
        if df is None or df.empty:
            continue
        simple_df = df if simple_df is None else simple_df.join(df, how='outer')
    if simple_df is None or simple_df.empty:
        raise ValueError(f"所选类别 {use_cats} 未加载到任何指数资产")
    simple_df = simple_df.sort_index()
    if start or end:
        simple_df = simple_df.loc[start:end]

    # 3) 协方差估计用对数收益
    log_df = compute_log_returns(simple_df)
    assets = log_df.columns.tolist()

    # 4) 月度调仓窗口
    trading_days_per_year = TRADING_DAYS_PER_YEAR_SCALAR
    lookback_window = trading_days_per_year * lookback_years
    min_data_days_required = trading_days_per_year * min_data_years
    rebalance_dates = log_df.groupby(log_df.index.to_period('M')).tail(1).index
    first_valid_rebalance_date = log_df.index[0] + pd.DateOffset(days=min_data_days_required)
    rebalance_dates = rebalance_dates[(rebalance_dates >= first_valid_rebalance_date) & (rebalance_dates <= log_df.index.max())]

    weights_hist = pd.DataFrame(index=rebalance_dates, columns=assets, dtype=float)
    rc_hist = pd.DataFrame(index=rebalance_dates, columns=assets, dtype=float)

    oas_model = OAS(assume_centered=True)
    for date in rebalance_dates:
        start_date = date - pd.DateOffset(days=lookback_window)
        window_df = log_df.loc[start_date:date]
        live_assets = window_df.count()[lambda s: s >= min_data_days_required].index.tolist()
        n_live = len(live_assets)
        if n_live < 2:
            weights_hist.loc[date] = 0.0
            rc_hist.loc[date] = 0.0
            continue
        clean_live_df = window_df[live_assets].dropna()
        if len(clean_live_df) < min_clean_days:
            weights_hist.loc[date] = 0.0
            rc_hist.loc[date] = 0.0
            continue
        try:
            oas_model.fit(clean_live_df)
            sigma = oas_model.covariance_
            w_series = hrp_allocate_recursive_bisect(
                sigma,
                live_assets,
                linkage_method=linkage_method,
                use_optimal_ordering=use_optimal_ordering,
                epsilon=EPSILON_NUMERIC,
                debug=debug,
                allocation_metric='std',
                diag_vol_floor=diag_vol_floor,
            )
            # RC (std 年化)
            w_vec = w_series.values
            mrc = sigma @ w_vec
            trc = w_vec * mrc
            port_var = float(w_vec @ sigma @ w_vec)
            port_std_d = float(np.sqrt(max(port_var, 0.0)))
            rc_std_d = trc / port_std_d if port_std_d > 0 else np.zeros_like(trc)
            rc_std_y = rc_std_d * np.sqrt(trading_days_per_year)

            weights_hist.loc[date] = 0.0
            weights_hist.loc[date, live_assets] = w_series.values
            rc_hist.loc[date] = 0.0
            rc_hist.loc[date, live_assets] = rc_std_y
        except Exception:
            # 回退等权
            fallback = pd.Series(1.0 / n_live, index=live_assets)
            weights_hist.loc[date] = 0.0
            weights_hist.loc[date, live_assets] = fallback.values
            rc_hist.loc[date] = 0.0

    weights_hist = weights_hist.fillna(0.0)
    rc_hist = rc_hist.fillna(0.0)

    # 5) 组合路径与绩效（以简单收益为回测基准）
    port_ret, equity, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=weights_hist,
        daily_returns_df=simple_df.loc[weights_hist.index[0]:],
        cost_per_side_scalar=cost_per_side,
    )
    equity.name = STRATEGY_MODE
    report = calculate_performance_metrics(port_ret, equity, rf_rate, trading_days_per_year)

    total_cost_series = (weights_hist.diff().abs().sum(axis=1).astype('float64') * cost_per_side).shift(1)
    total_cost_series = total_cost_series.reindex(port_ret.index).astype('float64').fillna(0.0)
    total_cost = float(total_cost_series.sum())

    figures = {}
    if auto_plot:
        fig1 = plot_equity_curve(equity, report, STRATEGY_MODE, auto_show=False)
        fig2 = plot_weights_history(weights_hist, STRATEGY_MODE, auto_show=False)
        plt.show()
        figures = {'equity': fig1, 'weights': fig2}

    return {
        'weights_history_df': weights_hist,
        'rc_history_df': rc_hist,
        'final_portfolio_returns_series': port_ret,
        'equity_curve_series': equity,
        'performance_report': report,
        'total_cost': total_cost,
        'figures': figures,
    }
