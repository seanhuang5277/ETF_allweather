# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_AllWeather.py
@Author : Copilot
@Date   : 2025-11-22

@Description:
全天候策略 (All Weather Strategy) - 基于宏观四象限的资产配置
参考思路：华泰证券“从资产配置走向因子配置”

核心逻辑：
1. 将资产池划分为四个宏观象限组合：
   - 增长上行 (Growth +): 股票 (Stock), 商品 (Commodity)
   - 增长下行 (Growth -): 债券 (Bond)
   - 通胀上行 (Inflation +): 商品 (Commodity), 黄金 (Gold)
   - 通胀下行 (Inflation -): 股票 (Stock), 债券 (Bond)
   
2. 组合构建三步走：
   - 第一步（大类资产内）：对归属于该资产类别的的资产进行配置。（比如股票内不同股票资产的分配）
     可选方法：等权 (EW) 或 分层风险平价 (HRP)。
   - 第二步（象限内大类资产）通过提前认为设定权重的方法在象限内部分配权重（比如增长超预期象限内股票和商品的分配）
   - 第三步（象限间）：将四个象限视为四个“宏观因子资产”，进行风险平价 (Risk Parity)。
     目标是让四个经济环境对组合的风险贡献相等。


3. 最终权重穿透：
   由于资产在不同象限间存在重叠（如股票既在增长+也在通胀-），
   最终权重为各象限分配权重的叠加。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Dict, List
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

from framework.config import (
    TRADING_DAYS_PER_YEAR_SCALAR,
    COST_PER_SIDE_SCALAR,
    RISK_FREE_RATE_SCALAR_ANNUAL,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_DATA_YEARS,
    MIN_CLEAN_DAYS,
    EPSILON_NUMERIC,
)
from framework.load_data import load_etf_index_returns_by_category, compute_log_returns
from framework.performance import (
    compute_portfolio_returns_and_equity, 
    calculate_performance_metrics,
    calculate_return_attribution
)
from framework.allocation_utils import (
    to_daily_rebalance_dates,
    solve_risk_parity_weights,
    hrp_allocate_recursive_bisect,
    map_factor_to_asset_weights,
    estimate_covariance_matrix,
    estimate_factor_betas_ols,
    solve_factor_risk_parity_weights,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import plot_equity_curve, plot_weights_history, plot_return_attribution, plot_multi_equity_curves
# from framework.black_litterman import calculate_implied_returns, black_litterman_posterior, get_bl_weights

# 初始化日志
setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_AllWeather.log")
logger = get_logger("strategy.AllWeather")

STRATEGY_MODE = 'All_Weather_Quadrants'


def _to_month_end_simple_returns(daily_simple_ret_df: pd.DataFrame) -> pd.DataFrame:
    """将日频简单收益压缩成月末简单收益（按月复利）。"""
    if daily_simple_ret_df is None or daily_simple_ret_df.empty:
        return pd.DataFrame()
    mret = (1.0 + daily_simple_ret_df).groupby(daily_simple_ret_df.index.to_period('M')).prod() - 1.0
    # PeriodIndex -> 月末 Timestamp
    mret.index = mret.index.to_timestamp('M')
    return mret


def get_factor_risk_budget_from_macro(
    macro_factor_df: pd.DataFrame,
    dt: pd.Timestamp,
    lookback_years: int,
    min_data_years: int,
    min_clean_days: int,
    method: str = "cov",
    ewm_span_days: int = 252,
) -> Optional[np.ndarray]:
    """
    基于宏观因子构造 Growth / Inflation 两个因子的协方差，并在“因子空间”内做一次 2x2 风险平价，
    返回目标因子风险预算 b = [b_Growth, b_Inflation]（已归一化）。

    注意：这一步只是在因子空间里确定“目标因子风险预算”。
    真正的资产/象限权重需要结合资产对因子的暴露(β)来解（见顶层分配处）。

    适配 Level/Change 因子体系：
    - 风险平价使用 Change 因子（边际变化）：CN_Growth_Change / CN_Inflation_Change
    - 不再对因子做 diff()（因为 Change 已是“变化/意外”口径）

    如果宏观数据不足，返回 None（回退到等权）。
    """
    if macro_factor_df is None:
        return None

    # 只用 dt 之前能观测到的宏观数据（使用上月月末，避免用到本月信息）
    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    hist = macro_factor_df.loc[:dt_prev_month_end].copy()
    if hist.empty:
        return None

    # 宏观因子通常是月频；窗口长度按月估计
    lookback_periods = int(12 * lookback_years)
    if lookback_periods > 0:
        hist = hist.iloc[-lookback_periods:]

    # 优先使用 Level/Change 体系中的 Change 因子
    if "CN_Growth_Change" in hist.columns:
        growth = hist["CN_Growth_Change"]
    else:
        return None

    if "CN_Inflation_Change" in hist.columns:
        inflation = hist["CN_Inflation_Change"]
    else:
        return None

    factor_df = pd.concat({"Growth": growth, "Inflation": inflation}, axis=1).dropna ()
    # min_clean_days 是日频口径；这里换算成月频最小样本（至少 12 期）
    min_obs = max(12, int(min_clean_days / 21))
    if len(factor_df) < min_obs:
        return None

    # Level/Change 体系：Change 已是变化口径，这里直接用其协方差
    df_used = factor_df.copy()

    use_ewm = isinstance(method, str) and (method.lower() == "ewm" or ("ewm" in method.lower()))
    if use_ewm:
        # 将日频 span 映射到月频（约 21 个交易日/月）
        span_m = max(6, int(ewm_span_days / 21))
        last_date = df_used.index[-1]
        cov_f_df = df_used.ewm(span=span_m).cov().loc[last_date]
    else:
        cov_f_df = df_used.cov()

    # 对 2x2 因子协方差做一次风险平价
    # solve_risk_parity_weights 返回 pd.Series
    b_series = solve_risk_parity_weights(cov_f_df, None)
    b = b_series.values
    
    # 归一化保证 sum(b)=1
    b = np.maximum(b, 0)
    s = b.sum()
    if s > 0:
        b = b / s
    return b


def get_position_sizing_from_monetary(macro_factor_df: pd.DataFrame,
                                      dt: pd.Timestamp,
                                      max_position: float = 1.0,
                                      min_position: float = 0.8,
                                      lookback_months: int = 3) -> float:
    """
    根据货币政策因子调整总仓位系数。
    参数：
        macro_factor_df: pd.DataFrame，宏观因子数据
        dt: pd.Timestamp，当前日期
        max_position: float，最大仓位（货币宽松时）
        min_position: float，最低仓位（货币紧缩时）
        lookback_months: int，回看月数判断货币趋势
    返回：
        position_ratio: float，当前应配置的仓位比例（0-1之间）
    """
    # 只用 dt 上一个月月末（或更早）可获得的宏观数据
    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    macro_idx = macro_factor_df.index[macro_factor_df.index <= dt_prev_month_end]
    
    if len(macro_idx) < lookback_months:
        logger.warning(f"[Position Sizing] {dt.date()} 宏观数据不足，使用满仓")
        return max_position
    
    latest_idx = macro_factor_df.index.get_indexer([dt_prev_month_end], method='pad')[0]
    if latest_idx < 2:
        return max_position
    
    # 读取货币政策因子：优先用 Level/Change 体系（正值=宽松）
    monetary_col = None
    if 'CN_Monetary_Level' in macro_factor_df.columns:
        monetary_col = 'CN_Monetary_Level'
    elif 'Monetary Policy_OECD' in macro_factor_df.columns:
        monetary_col = 'Monetary Policy_OECD'
    
    if monetary_col is None or pd.isna(macro_factor_df.iloc[latest_idx][monetary_col]):
        logger.warning(f"[Position Sizing] {dt.date()} 无货币政策数据，使用满仓")
        return max_position
    
    monetary_factor_t = macro_factor_df.iloc[latest_idx][monetary_col]
    monetary_factor_t_minus_1 = macro_factor_df.iloc[latest_idx - 1][monetary_col]
    monetary_factor_t_minus_2 = macro_factor_df.iloc[latest_idx - 2][monetary_col]

    if not pd.isna(monetary_factor_t):
        if monetary_factor_t > monetary_factor_t_minus_1 and monetary_factor_t_minus_1 > monetary_factor_t_minus_2:
            # 货币条件走向更宽松：提高总仓位
            return max_position
        elif monetary_factor_t < monetary_factor_t_minus_1 and monetary_factor_t_minus_1 < monetary_factor_t_minus_2:
            # 货币条件走向更紧缩：降低总仓位
            return min_position
        else:
            return (max_position + min_position) / 2


def get_macro_adjusted_risk_budget(
    valid_quadrants: List[str],
    macro_factor_df: pd.DataFrame,
    dt: pd.Timestamp,
    base_factor_budget: Optional[np.ndarray] = None,  # 新增：因子层预算 [b_G, b_I]
    growth_tilt_strength: float = 0.2,
    inflation_tilt_strength: float = 0.2,
) -> Optional[pd.Series]:
    """
    现在的含义变成：
    1) 先用 base_factor_budget 决定 Growth / Inflation 总预算；
    2) 再用宏观因子趋势，把每个因子预算拆成 + / - 两个象限；
    3) 返回象限层预算（Series，索引=valid_quadrants）。

    如果 base_factor_budget is None，就退化成原来的逻辑：四象限等权 + 趋势微调。
    """

    # 先做原有的索引裁剪：使用上月月末
    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    macro_idx = macro_factor_df.index[macro_factor_df.index <= dt_prev_month_end]
    if len(macro_idx) == 0:
        return None

    latest_idx = macro_factor_df.index.get_indexer([dt_prev_month_end], method='pad')[0]
    if latest_idx < 2:
        return None

    # 读取宏观因子：优先用 Change（用于 Rising/Falling 方向判断）
    g_t = macro_factor_df.iloc[latest_idx].get('CN_Growth_Change', np.nan)
    g_1 = macro_factor_df.iloc[latest_idx - 1].get('CN_Growth_Change', np.nan)
    g_2 = macro_factor_df.iloc[latest_idx - 2].get('CN_Growth_Change', np.nan)
    if pd.isna(g_t):
        g_t = macro_factor_df.iloc[latest_idx].get('growth_OECD', macro_factor_df.iloc[latest_idx].get('growth_PCA', np.nan))
        g_1 = macro_factor_df.iloc[latest_idx - 1].get('growth_OECD', macro_factor_df.iloc[latest_idx - 1].get('growth_PCA', np.nan))
        g_2 = macro_factor_df.iloc[latest_idx - 2].get('growth_OECD', macro_factor_df.iloc[latest_idx - 2].get('growth_PCA', np.nan))

    i_t = macro_factor_df.iloc[latest_idx].get('CN_Inflation_Change', np.nan)
    i_1 = macro_factor_df.iloc[latest_idx - 1].get('CN_Inflation_Change', np.nan)
    i_2 = macro_factor_df.iloc[latest_idx - 2].get('CN_Inflation_Change', np.nan)
    if pd.isna(i_t):
        i_t = macro_factor_df.iloc[latest_idx].get('inflation_OECD', macro_factor_df.iloc[latest_idx].get('inflation_PCA', np.nan))
        i_1 = macro_factor_df.iloc[latest_idx - 1].get('inflation_OECD', macro_factor_df.iloc[latest_idx - 1].get('inflation_PCA', np.nan))
        i_2 = macro_factor_df.iloc[latest_idx - 2].get('inflation_OECD', macro_factor_df.iloc[latest_idx - 2].get('inflation_PCA', np.nan))

    # 先决定 Growth / Inflation 两个“总桶”的预算
    if base_factor_budget is not None and len(base_factor_budget) == 2:
        b_G, b_I = base_factor_budget[0], base_factor_budget[1]
    else:
        # 没有因子预算时，两个因子等权
        b_G = b_I = 0.5

    # 初始化四象限预算为 0
    budget_q = pd.Series(0.0, index=valid_quadrants)

    # -------------------------
    # 1) Growth 因子拆成两个象限
    # -------------------------
    if 'Growth_Rising' in valid_quadrants or 'Growth_Falling' in valid_quadrants:
        # 先默认 50/50 拆分
        split_G_up = 0.5
        # 用趋势给一点倾斜
        if not pd.isna(g_t):
            if g_t > g_1 > g_2:      # 增长连续上行
                split_G_up = 0.5 + growth_tilt_strength
            elif g_t < g_1 < g_2:    # 增长连续下行
                split_G_up = 0.5 - growth_tilt_strength
        split_G_up = np.clip(split_G_up, 0.0, 1.0)
        split_G_dn = 1.0 - split_G_up

        if 'Growth_Rising' in valid_quadrants:
            budget_q['Growth_Rising'] = b_G * split_G_up
        if 'Growth_Falling' in valid_quadrants:
            budget_q['Growth_Falling'] = b_G * split_G_dn

    # -------------------------
    # 2) Inflation 因子拆成两个象限
    # -------------------------
    if 'Inflation_Rising' in valid_quadrants or 'Inflation_Falling' in valid_quadrants:
        split_I_up = 0.5
        if not pd.isna(i_t):
            if i_t > i_1 > i_2:
                split_I_up = 0.5 + inflation_tilt_strength
            elif i_t < i_1 < i_2:
                split_I_up = 0.5 - inflation_tilt_strength
        split_I_up = np.clip(split_I_up, 0.0, 1.0)
        split_I_dn = 1.0 - split_I_up

        if 'Inflation_Rising' in valid_quadrants:
            budget_q['Inflation_Rising'] = b_I * split_I_up
        if 'Inflation_Falling' in valid_quadrants:
            budget_q['Inflation_Falling'] = b_I * split_I_dn

    # 万一有缺失象限（比如某期没有构造出来），做一次归一化
    if budget_q.sum() > 0:
        budget_q = budget_q / budget_q.sum()
    else:
        # 退化到等权
        n = len(valid_quadrants)
        budget_q = pd.Series(1.0 / n, index=valid_quadrants)
    
    logger.debug(f"[Macro Risk Budget] {dt.date()} Adjusted Budget: {dict(zip(valid_quadrants, np.round(budget_q.values, 3)))} "
                f"(G={g_t:.2f}, I={i_t:.2f})")

    return budget_q


def run_strategy_updated(
    *,
    data_dir: str = 'data',
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    # 策略参数
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    
    # 象限内分配方法: 'EW' (等权) 或 'HRP' (分层风险平价)
    internal_method: str = 'HRP',
    
    # 协方差估计参数
    top_cov_estimate_ways: str = 'cov',     # 顶层（象限间）协方差估计方法
    bottom_cov_estimate_ways: str = 'cov',  # 底层（ETF间，用于HRP）协方差估计方法
    ewm_span_days: int = 252,  # EWM半衰期（天数）
        #估计方法：
        # - 'cov': 标准样本协方差
        #- 'ewm': 指数加权移动协方差
        #- 'downside_cov': 下半协方差（仅使用负收益）
        #- 'downside_ewm': 下半指数加权移动协方差


    # 换仓日参数
    rebalance_day: int = None,  # 默认为月末，可指定几号

    # 成本与绩效参数
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    use_etf_real_data: bool = True,

    # 宏观因子调整 参数
    macro_factor_adjustment: bool = False,
    growth_tilt_strength: float = 0.2,
    inflation_tilt_strength: float = 0.1,
    
    # 货币政策仓位调整 参数
    use_monetary_position_sizing: bool = False,
    max_position: float = 1.0,
    min_position: float = 0.8,
) -> Dict[str, object]:


    # --------------------------------------------------------------------------
    # 一、 数据准备与预处理
    # --------------------------------------------------------------------------
    start_dt_global = pd.to_datetime(start_date)
    end_dt_global = pd.to_datetime(end_date)


    # 1. 加载宏观因子数据
    # 注意：在新框架下，我们需要始终加载宏观数据来计算"因子风险预算"(Factor Risk Budgeting)，
    # 即使 macro_factor_adjustment=False (即不做战术倾斜)，我们也需要宏观数据来构建基础的因子协方差。
    macro_factor_df = None
    # 始终尝试加载，不再受 macro_factor_adjustment 开关限制
    try:
        all_path = f"{data_dir}/all_macro_factors.csv"           # Level/Change 因子体系

        if os.path.exists(all_path):
            # all_macro_factors.csv 通常是 index 写入 CSV（第一列为日期）
            macro_factor_df = pd.read_csv(all_path, index_col=0, parse_dates=True)
            macro_factor_df = macro_factor_df.sort_index()
            logger.debug(f"已加载宏观因子数据(Level/Change): {all_path}")

            # 映射列名以统一逻辑
            # synthesized: growth_PCA, inflation_PCA
            col_map = {}
            if 'CN_Growth_Change' in macro_factor_df.columns:
                col_map['CN_Growth_Change'] = 'CN_Growth_Change'
            # elif 'growth_OECD' in macro_factor_df.columns:
            #     col_map['growth_OECD'] = 'growth_OECD'

            if 'CN_Inflation_Change' in macro_factor_df.columns:
                col_map['CN_Inflation_Change'] = 'CN_Inflation_Change'
            # elif 'inflation_OECD' in macro_factor_df.columns:
            #     col_map['inflation_OECD'] = 'inflation_OECD'


            macro_factor_df = macro_factor_df.rename(columns=col_map)

        else:
            logger.warning("未找到宏观因子文件(all_macro_factors.csv 或 macro_factors_synthesized.csv)，将跳过宏观因子预算/倾斜。")
            macro_factor_df = None

        DR007_path = f"{data_dir}/macro_factors_raw.csv" 
        if os.path.exists(DR007_path):
            dr007_df = pd.read_csv(DR007_path, index_col='date', parse_dates=True)
            if 'DR007' in dr007_df.columns:
                dr007_rate = dr007_df['DR007'] / 100  # 转为小数形式
                # 转换为日收益率 (年化利率 / 365)
                dr007_daily_return = dr007_rate / 365
                logger.debug(f"已加载 DR007 数据用于现金收益计算: {DR007_path}")
            else:
                dr007_daily_return = None
                logger.warning(f"DR007 列不存在于 {DR007_path}")
        else:
            dr007_daily_return = None
            logger.warning(f"未找到 DR007 数据文件: {DR007_path}")

    except Exception as e:
        logger.error(f"加载宏观因子数据失败: {e}")
        macro_factor_adjustment = False


    # 2.加载行情数据 (按类别分组)
    etf_by_cat_dict_df, idx_by_cat_dict_df = load_etf_index_returns_by_category(data_dir=data_dir, returns_index_col='日期')
    
    # 合并所有资产的大表
    all_index_list = []
    all_etf_list = []
    # 遍历所有类别确保数据加载
    available_cats = list(idx_by_cat_dict_df.keys())
    for cat in available_cats:
        all_index_list.append(idx_by_cat_dict_df[cat])
        all_etf_list.append(etf_by_cat_dict_df[cat])
        
    all_idx_simp_df = pd.concat(all_index_list, axis=1).sort_index()
    all_etf_simp_df = pd.concat(all_etf_list, axis=1).sort_index()

    mapping = {x : y for x,y in zip(all_idx_simp_df.columns, all_etf_simp_df.columns)}
    reversed_mapping = {y : x for x,y in mapping.items()}

    # 预计算对数收益
    all_idx_log_df = compute_log_returns(all_idx_simp_df)



    # 3.定义四象限映射 {象限：{大类资产：[权重 + ETF 名称]}}
    quadrant_defs = {
        'Growth_Rising': {
            'Growth_Rising_Stock': [0.8,'沪深300ETF', '中证500ETF', '中证1000ETF','红利质量ETF'],
            'Growth_Rising_Commodity': [0.1,'有色ETF', '豆粕ETF', '能源化工ETF'],
            'Growth_Rising_Credit_Bond': [0.1, '信用债ETF', '可转债ETF'],
        },
        'Growth_Falling': {
            'Growth_Falling_Rates_Bond': [0.7,'国债ETF', '十年国债ETF','30年国债ETF'],
            'Growth_Falling_Gold': [0.3,'黄金ETF'],
        },
        'Inflation_Rising': {
            'Inflation_Rising_Commodity': [0.7,'有色ETF', '豆粕ETF', '能源化工ETF'],
            'Inflation_Rising_Gold': [0.3,'黄金ETF'],
        },
        'Inflation_Falling': {
            'Inflation_Falling_Dividend_Stock': [0.1,'红利低波50ETF', '红利质量ETF'],
            'Inflation_Falling_Rates_Bond': [0.9,'国债ETF', '十年国债ETF', '30年国债ETF','公司债ETF','可转债ETF'],
        },
    }

    quadrant_names = list(quadrant_defs.keys())
    logger.debug(f"四象限定义: {quadrant_defs}")


    # 4创建合成现金收益序列：短融ETF上市前用DR007，之后用短融ETF
    if use_monetary_position_sizing:
        if '短融ETF' in all_etf_simp_df.columns and dr007_daily_return is not None:
            cash_etf_inception = all_etf_simp_df['短融ETF'].first_valid_index()
            
            # 对齐DR007到ETF数据的日期范围
            dr007_aligned = dr007_daily_return.reindex(all_etf_simp_df.index, method='ffill')
            
            # 创建合成现金收益
            cash_return_series = all_etf_simp_df['短融ETF'].copy()
            if cash_etf_inception is not None:
                # 短融ETF上市前使用DR007收益率
                cash_return_series.loc[:cash_etf_inception] = dr007_aligned.loc[:cash_etf_inception]
            
            # 将合成现金收益替换到ETF数据中
            all_etf_simp_df['短融ETF'] = cash_return_series
            logger.info(f"已创建合成现金收益：短融ETF上市日={cash_etf_inception.date() if cash_etf_inception else 'N/A'}，之前使用DR007")
        elif dr007_daily_return is not None and '短融ETF' not in all_etf_simp_df.columns:
            # 如果没有短融ETF但有DR007，创建虚拟短融ETF列
            dr007_aligned = dr007_daily_return.reindex(all_etf_simp_df.index, method='ffill')
            all_etf_simp_df['短融ETF'] = dr007_aligned
            logger.info("未找到短融ETF，使用DR007创建虚拟现金收益序列")


    # 5.计算 ETF 上市时间
    etf_inception = {}
    for idx_name, etf_name in mapping.items():
        if use_etf_real_data:
            first_inc = all_etf_simp_df[etf_name].first_valid_index()
        else:
            first_inc = all_idx_simp_df[idx_name].first_valid_index()
        if first_inc is not None:
            etf_inception[etf_name] = first_inc

    # 6.确定再平衡日期
    min_data_days_required = TRADING_DAYS_PER_YEAR_SCALAR * min_data_years
    rebalance_dates = to_daily_rebalance_dates(all_idx_simp_df, min_data_days_required, rebalance_day=rebalance_day)
    # 处理时间段截取
    if start_dt_global is not None:
        future_rebals = rebalance_dates[rebalance_dates >= start_dt_global]
        if start_dt_global not in future_rebals and start_dt_global >= all_idx_simp_df.index[0] + pd.DateOffset(days=min_data_days_required):
             rebalance_dates = pd.DatetimeIndex([start_dt_global] + list(future_rebals)).sort_values()
        else:
             rebalance_dates = future_rebals
    if end_dt_global is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end_dt_global]
    # 确定回溯窗口天数    
    lookback_window = TRADING_DAYS_PER_YEAR_SCALAR * lookback_years

    
    # --------------------------------------------------------------------------
    # 二、 核心回测循环
    # --------------------------------------------------------------------------
    
    # 0.存储权重
    # 用于最终画图大类权重：象限权重(仅仅只是为了画图用所以在前面保存)
    top_quadrant_w_hist = pd.DataFrame(index=rebalance_dates, columns=quadrant_names, dtype=float).fillna(0.0)        
    # 底层DataFrame权重：具体ETF口径权重（真正核心用于回测的权重）
    bottom_atomic_w_hist = pd.DataFrame(index=rebalance_dates, columns=all_etf_simp_df.columns, dtype=float).fillna(0.0)
    
    # 存储每个象限的底层资产权重历史，用于计算象限净值曲线
    quadrant_atomic_weights_hist = {q: pd.DataFrame(index=rebalance_dates, columns=all_etf_simp_df.columns, dtype=float).fillna(0.0) for q in quadrant_names}

    # 1.顶层对日期循环(第一层循环：最外层)
    for dt in rebalance_dates:

        
        
        # 准备本期数据窗口
        start_dt = dt - pd.DateOffset(days=lookback_window)              
        window_log_df = all_idx_log_df.loc[start_dt:dt]
        window_simp_df = all_idx_simp_df.loc[start_dt:dt]
        window_simp_df.columns = all_etf_simp_df.columns

        # 2.对象限循环，最终计算各象限的“合成净值/收益”序列（第二层循环：中间层）
        # 临时存储各象限内部对各Class的权重分配和最终合成的 {quadrant: {Class: weight}}
        quadrant_returns_series_list = []
        valid_quadrants = []
        quadrant_internal_weights = {}         
        for q_name in quadrant_defs.keys():
            
            # 3.对象限内资产类别（Class）循环，最终计算各Class的“合成净值/收益”序列（第三层循环：最内层）
            # 临时存储各class内部对各ETF的权重分配  {class: {ETF: weight}}
            class_returns_series_list = []
            valid_class = []
            class_internal_weights = {} 
            for class_name in quadrant_defs[q_name]:
                target_etfs = quadrant_defs[q_name][class_name][1:]
            
                # 收集该象限下的所有候选资产列名 (Index Name)
                candidate_assets = [reversed_mapping.get(etf_name) for etf_name in target_etfs]
                if None in candidate_assets:  # 仅限Debug,想跳过
                    logger.warning(f"象限 {q_name}: 部分 ETF 未找到对应的 Index 映射，请检查配置文件")

                # 筛选存活资产
                live_idx_cols = [
                    asset for asset in candidate_assets
                    if asset is not None
                    and (inc := etf_inception.get(mapping.get(asset))) is not None
                    and inc <= dt
                    and window_log_df[asset].count() >= min_data_days_required
                ]
                
                if not live_idx_cols:
                    continue
                
                # --- 象限-内分配 (Internal Allocation) ---
                n_live = len(live_idx_cols)
                live_etf_cols = [mapping[c] for c in live_idx_cols]
                if n_live == 1:
                    w_series = pd.Series(1.0, index=live_etf_cols)
                elif internal_method == 'EW':
                    w_series = pd.Series(1.0 / n_live, index=live_etf_cols)
                elif internal_method == 'HRP': # HRP方法使用最原始的COV矩阵很重要，协方差之间的细微差别是构建距离的关键
                    clean_df = window_log_df[live_idx_cols].dropna()
                    if len(clean_df) < min_clean_days:
                        w_series = pd.Series(1.0 / n_live, index=live_etf_cols)
                    else:
                        cov_mat = clean_df.cov().values
                        w_series = hrp_allocate_recursive_bisect(
                            cov_mat, live_etf_cols, 
                            linkage_method='ward',
                            allocation_metric='std'
                        )               
                class_internal_weights[class_name] = w_series
                # 用于计算收益

                
                # 计算每个Class合成收益 (用于中间层)
                # R_q = sum(w_i * R_i)
                # 使用简单收益合成
                class_simp_ret = window_simp_df[live_etf_cols].fillna(0.0).mul(class_internal_weights[class_name], axis=1).sum(axis=1)

                if class_simp_ret.count() >= min_data_days_required:
                    class_simp_ret.name = class_name
                    class_returns_series_list.append(class_simp_ret)
                    valid_class.append(class_name)      



                    
                              
            # Class循环结束,进行Class层权重
            if len(valid_class) <= 0:
                continue
            # 合并Class收益
            class_simp_ret = pd.concat(class_returns_series_list, axis=1).dropna()

            # 读取预设的权重（这个内置没有设为参数）
            w_class = pd.Series([quadrant_defs[q_name].get(x)[0] for x in valid_class], index=valid_class)

            # 归1化
            w_class = w_class / w_class.sum()    


            
            # 存储用于穿透
            quadrant_internal_weights[q_name] = {
                class_name: (w_class[class_name], class_internal_weights[class_name])
                for class_name in valid_class
            }
        
            # 计算象限合成收益 (用于顶层风险平价)
            # R_quadrant = sum(w_class * R_class)
            q_simp_ret = class_simp_ret.mul(w_class, axis=1).sum(axis=1)
            
            if q_simp_ret.count() >= min_data_days_required:
                q_simp_ret.name = q_name
                quadrant_returns_series_list.append(q_simp_ret)
                valid_quadrants.append(q_name)


        # 2. 顶层分配 (Top Level Allocation) - 象限间风险平价（这里传入的就没有Cash）
        # 合并象限收益
        quadrant_ret_df = pd.concat(quadrant_returns_series_list, axis=1).dropna()
        # 通过判断跳过的方式，这样格式不影响缩进
        if len(quadrant_ret_df) < min_clean_days:
            logger.warning(f"[{dt.date()}] 有效象限收益数据不足，跳过本次调仓")
            continue

        # 2.1 象限协方差 (仅用于最终 w_top 的风险平价)
        q_log_ret_df = compute_log_returns(quadrant_ret_df)
        cov_top = estimate_covariance_matrix(
            q_log_ret_df,
            method=top_cov_estimate_ways,
            ewm_span=ewm_span_days
        )

        # 2.2 顶层分配：优先做“因子风险平价(基于β暴露)”求象限权重
        w_top = None
        if macro_factor_df is not None:
            try:
                # 只用上月月末及更早的宏观信息
                dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))

                # 1) 因子样本（月频，Change）
                factor_cols = ['CN_Growth_Change', 'CN_Inflation_Change']
                factors_m = macro_factor_df.loc[:dt_prev_month_end, factor_cols].dropna()

                # 2) 象限收益样本（月频）
                quad_m = _to_month_end_simple_returns(quadrant_ret_df.loc[:dt_prev_month_end])

                # 3) 对齐并截取窗口
                hist = quad_m.join(factors_m, how='inner').dropna()
                lookback_periods = int(12 * lookback_years)
                if lookback_periods > 0:
                    hist = hist.iloc[-lookback_periods:]

                quad_cols = [c for c in valid_quadrants if c in hist.columns]
                if len(quad_cols) >= 2 and len(hist) >= max(24, int(min_clean_days / 21)):
                    betas_df, _ = estimate_factor_betas_ols(
                        returns_df=hist[quad_cols],
                        factors_df=hist[factor_cols],
                        min_obs=max(24, int(min_clean_days / 21)),
                    )

                    # 4) 因子协方差（在因子空间）
                    
                    use_ewm = isinstance(top_cov_estimate_ways, str) and ('ewm' in top_cov_estimate_ways.lower())
                    if use_ewm:
                        span_m = max(6, int(ewm_span_days / 21))
                        last_date = hist.index[-1]
                        cov_f = hist[factor_cols].ewm(span=span_m).cov().loc[last_date]
                    else:
                        cov_f = hist[factor_cols].cov()
                    cov_f = cov_f.loc[factor_cols, factor_cols]

                    # 5) 目标因子风险预算（可用你原来的“因子空间RP”得到），缺省=等权
                    base_factor_budget = get_factor_risk_budget_from_macro(
                        macro_factor_df=macro_factor_df,
                        dt=dt,
                        lookback_years=lookback_years,
                        min_data_years=min_data_years,
                        min_clean_days=min_clean_days,
                        method=top_cov_estimate_ways,
                        ewm_span_days=ewm_span_days,
                    )
                    target_factor_budget = None
                    if base_factor_budget is not None and len(base_factor_budget) == 2:
                        target_factor_budget = pd.Series(base_factor_budget, index=factor_cols)

                    w_top = solve_factor_risk_parity_weights(
                        betas_df=betas_df,
                        factor_cov_df=cov_f,
                        target_factor_budget=target_factor_budget,
                        ridge_lambda=1e-4,
                        long_only=True,
                        weight_cap=None,
                    )
            except Exception as e:
                logger.warning(f"[{dt.date()}] 因子RP(β暴露)失败，回退象限RP: {e}")
                w_top = None

        # 2.3 回退：传统象限协方差风险平价（可选使用宏观预算作为 target）
        if w_top is None:
            target_budget = None
            if macro_factor_df is not None:
                current_g_tilt = growth_tilt_strength if macro_factor_adjustment else 0.0
                current_i_tilt = inflation_tilt_strength if macro_factor_adjustment else 0.0
                budget_series = get_macro_adjusted_risk_budget(
                    valid_quadrants=valid_quadrants,
                    macro_factor_df=macro_factor_df,
                    dt=dt,
                    base_factor_budget=None,
                    growth_tilt_strength=current_g_tilt,
                    inflation_tilt_strength=current_i_tilt,
                )
                if budget_series is not None:
                    target_budget = budget_series.reindex(valid_quadrants)

            w_top = solve_risk_parity_weights(cov_top, target_budget)

        # 清理 & 归一化
        w_top = w_top.reindex(valid_quadrants).fillna(0.0)
        w_top = w_top.clip(lower=0.0)
        s = float(w_top.sum())
        if s > 0:
            w_top = w_top / s

        
        # Log
        w_top_str = ", ".join([f"{k}:{v:.2%}" for k, v in w_top.items()])
        logger.debug(f"[{dt.date()}] Top Quadrants Weights: {w_top_str}")
        
        # 3. 穿透计算底层资产权重 (Aggregation)
        # w_asset_final = sum(w_quadrant * w_asset_in_quadrant)
        
        # 初始化本期底层权重
        current_atomic_w = pd.Series(0.0, index=all_etf_simp_df.columns)
        
        # 三层权重穿透：象限 -> 资产类别 -> ETF
        for q_name, q_weight in w_top.items():
            if q_name not in quadrant_internal_weights:
                continue
            
            # 计算该象限的原子权重（不含象限本身权重 q_weight）
            q_atomic_w = pd.Series(0.0, index=all_etf_simp_df.columns)
            
            for class_name, (class_weight, etf_weights) in quadrant_internal_weights[q_name].items():
                # 三层穿透：象限权重 × 资产类别权重 × ETF权重
                penetrated_weights = q_weight * class_weight * etf_weights 
                
                # 累加权重 (处理重叠资产，如黄金在多个象限)
                current_atomic_w = current_atomic_w.add(penetrated_weights, fill_value=0.0)
                
                # 累加象限原子权重 (仅资产类别权重 * ETF权重)
                q_atomic_w = q_atomic_w.add(class_weight * etf_weights, fill_value=0.0)
            
            # 归一化象限原子权重并存储
            if q_atomic_w.sum() > 0:
                q_atomic_w = q_atomic_w / q_atomic_w.sum()
            quadrant_atomic_weights_hist[q_name].loc[dt] = q_atomic_w
        
        # 归一化 (理论上如果各层都归一化了，总和应该是1，但为了精度再次归一化)
        if current_atomic_w.sum() > 0:
            current_atomic_w = current_atomic_w / current_atomic_w.sum()
        
        ## 货币政策仓位调整（在时间循环内）
        # ----------------------------------  使用DR007和短融ETF  ----------------------------------------#
        if use_monetary_position_sizing and macro_factor_df is not None:
            pos_ratio = get_position_sizing_from_monetary(
                macro_factor_df, dt, max_position, min_position
            )
            # 四象限权重整体缩放
            current_atomic_w *= pos_ratio
            # 现金权重 = 1 - 四象限总权重
            cash_weight = max(0.0, 1.0 - current_atomic_w.sum())
            current_atomic_w.loc['短融ETF'] = cash_weight  # 确保现金ETF权重为0
            # 再次归一化
            current_atomic_w = current_atomic_w / current_atomic_w.sum()
            # logger.debug(f"[{dt.date()}] Position Ratio={pos_ratio:.1%}, Cash={cash_weight:.1%}")
            # 记录原子层权重
            w_top *= pos_ratio
        bottom_atomic_w_hist.loc[dt] = current_atomic_w
            # 记录顶层权重
            
        # w_top_all = pd.concat([w_top,pd.Series({'Cash': cash_weight})])
        # w_top_all = w_top_all/w_top_all.sum()
        w_top_all = w_top
        top_quadrant_w_hist.loc[dt] = w_top_all 



    # --------------------------------------------------------------------------
    # 3. 结果计算与展示
    # --------------------------------------------------------------------------
    
    # 映射回index 代码
    bottom_atomic_w_index = bottom_atomic_w_hist.rename(columns=reversed_mapping)
    bottom_atomic_w_etf = bottom_atomic_w_hist
    # 选择回测用的收益率数据 (Index 或 ETF)
    if use_etf_real_data:
        plot_w = bottom_atomic_w_etf
        plot_ret = all_etf_simp_df.loc[bottom_atomic_w_etf.index[0]:]
    else:
        plot_w = bottom_atomic_w_index
        plot_ret = all_idx_simp_df.loc[bottom_atomic_w_index.index[0]:]

    # 计算组合净值
    _,port_ret, _,equity_curve_series, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=plot_w,
        daily_returns_df=plot_ret,
        cost_per_side_scalar=cost_per_side,
    )
    
    # 绩效指标
    report = calculate_performance_metrics(
        port_ret, equity_curve_series, rf_rate, TRADING_DAYS_PER_YEAR_SCALAR,
        rebalance_dates=rebalance_dates
    )
    
    # 交易成本
    total_cost_series = (bottom_atomic_w_etf.diff().abs().sum(axis=1).astype('float64') * cost_per_side).shift(1)
    total_cost = float(total_cost_series.fillna(0.0).sum())
    logger.info(f"总交易成本: {total_cost:.6f}")
    top_quadrant_for_plot = top_quadrant_w_hist.reindex(daily_w.index).ffill()
    
    # 计算调仓期收益（用于绘图）- 只取净值曲线日期范围内的调仓日期
    valid_rebalance_dates = rebalance_dates[rebalance_dates >= equity_curve_series.index[0]]
    rebalance_returns = equity_curve_series.loc[valid_rebalance_dates].pct_change().dropna()

    # 计算每个象限的净值曲线
    quadrant_equity_curves = {}
    for q_name, q_weights_hist in quadrant_atomic_weights_hist.items():
        # 映射回index代码（如果需要）
        if use_etf_real_data:
            q_plot_w = q_weights_hist
            q_plot_ret = all_etf_simp_df.loc[q_weights_hist.index[0]:]
        else:
            q_plot_w = q_weights_hist.rename(columns=reversed_mapping)
            q_plot_ret = all_idx_simp_df.loc[q_weights_hist.index[0]:]
            
        _, _, _, q_equity_curve, _ = compute_portfolio_returns_and_equity(
            weights_history_df=q_plot_w,
            daily_returns_df=q_plot_ret,
            cost_per_side_scalar=cost_per_side,
        )
        quadrant_equity_curves[q_name] = q_equity_curve

    # 计算每个资产类别(Class)的净值曲线
    class_equity_curves = {}
    
    # 1. 构建 Class -> [ETFs] 映射
    # etf_by_cat_dict_df 的 key 是 Class Name, value 是 DataFrame (columns 是 ETF)
    class_to_etfs = {}
    for cat, df in etf_by_cat_dict_df.items():
        class_to_etfs[cat] = list(df.columns)
    
    # 手动添加 Cash (短融ETF)
    if '短融ETF' in all_etf_simp_df.columns:
        # 如果 Cash 已经在 etf_by_cat_dict_df 中则跳过，否则添加
        found_cash = False
        for cat, etfs in class_to_etfs.items():
            if '短融ETF' in etfs:
                found_cash = True
                break
        if not found_cash:
            class_to_etfs['Cash'] = ['短融ETF']

    # 2. 循环计算 Class 净值
    for cls, etfs in class_to_etfs.items():
        # 过滤出存在的 ETF
        valid_etfs = [e for e in etfs if e in bottom_atomic_w_hist.columns]
        if not valid_etfs:
            continue
            
        # 提取权重
        cls_w_hist = bottom_atomic_w_hist[valid_etfs].copy()
        
        # 归一化 (按行求和，然后除以和)
        # 仅计算该 Class 总权重 > 0 的日期
        row_sums = cls_w_hist.sum(axis=1)
        valid_dates = row_sums[row_sums > 0].index
        
        if len(valid_dates) == 0:
            continue
            
        cls_w_hist = cls_w_hist.loc[valid_dates].div(row_sums.loc[valid_dates], axis=0)
        
        # 准备收益率数据
        if use_etf_real_data:
             cls_ret = all_etf_simp_df[valid_etfs].loc[valid_dates[0]:]
        else:
             # 简化处理：如果用 Index 数据，这里可能需要映射，暂略，假设 ETF 数据可用
             cls_ret = all_etf_simp_df[valid_etfs].loc[valid_dates[0]:]

        # 计算净值
        _, _, _, cls_curve, _ = compute_portfolio_returns_and_equity(
            weights_history_df=cls_w_hist,
            daily_returns_df=cls_ret,
            cost_per_side_scalar=cost_per_side,
        )
        class_equity_curves[cls] = cls_curve

    # 收益归因 (按象限)
    # 由于象限是动态构建的，且资产重叠，精确归因比较复杂。
    # 这里我们做一个近似：计算每个象限的“贡献度” = 象限权重 * 象限收益
    # 但为了准确，我们还是用底层资产归因，然后按象限定义聚合（注意重叠部分会被重复统计在不同象限的“名义贡献”中，或者我们只看底层）
    
    # 这里提供底层资产归因
    atomic_attribution = calculate_return_attribution(
        weights_df=bottom_atomic_w_etf,
        daily_returns_df=all_etf_simp_df,
        cost_per_side=cost_per_side
    )

    return {
        'performance_report': report,
        'equity_curve_series': equity_curve_series,
        'final_weights_df': bottom_atomic_w_etf,
        'quadrant_weights_df': top_quadrant_w_hist,
        'daily_weights_df': daily_w,
        'top_quadrant_for_plot': top_quadrant_for_plot,
        'atomic_attribution': atomic_attribution,
        'rebalance_returns': rebalance_returns,
        'quadrant_equity_curves': quadrant_equity_curves,
        'class_equity_curves': class_equity_curves,
    }



if __name__ == "__main__":
    # 示例运行
    res = run_strategy_updated(
        internal_method='EW', # 象限内使用 HRP 或者EW
        start_date="2018-11-30",
        end_date="2025-11-30",
        rebalance_day = None,
        macro_factor_adjustment= False, # <--- 关闭战术倾斜，测试纯因子风险平价结构
        growth_tilt_strength = 0.1,
        inflation_tilt_strength = 0.1,
        use_monetary_position_sizing=False,  # 启用货币政策仓位调整
        max_position=1.0,  # 货币宽松时满仓
        min_position=0.8,  # 货币紧缩时80%仓位
        use_etf_real_data = False,
        # 协方差估计方法
        top_cov_estimate_ways='cov',  # 顶层（象限间）
        bottom_cov_estimate_ways='cov',  # 底层（ETF间）
        ewm_span_days=252,
    )
    
    # 打印性能指标
    perf = res['performance_report']
    logger.info("全天候策略运行完成，主要指标：")
    for k, v in perf.items():
        try:
            if isinstance(v, pd.Timestamp):
                logger.info(f"  {k}: {v.strftime('%Y-%m-%d')}")
            elif any(x in k for x in ['CAGR', 'Vol', '回撤', '胜率', 'Return', '收益率']) and not isinstance(v, pd.Timestamp):
                logger.info(f"  {k}: {v:.2%}")
            else:
                logger.info(f"  {k}: {v:.4f}")
        except Exception:
            logger.info(f"  {k}: {v}")
    
    # 绘制图表（在run_strategy外部）
    fig1 = plot_equity_curve(res['equity_curve_series'], res['performance_report'], STRATEGY_MODE, 
                            rebalance_returns=res['rebalance_returns'], auto_show=False)
    fig2 = plot_weights_history(res['top_quadrant_for_plot'], "Quadrant Allocation", reverse_legend=True, auto_show=False)
    fig3 = plot_weights_history(res['daily_weights_df'], "Asset Allocation", reverse_legend=True, auto_show=False)
    fig4 = plot_return_attribution(res['atomic_attribution'], title="底层资产收益贡献", auto_show=False)
    fig5 = plot_multi_equity_curves(res['quadrant_equity_curves'], title="四象限净值曲线对比", auto_show=False)
    fig6 = plot_multi_equity_curves(res['class_equity_curves'], title="资产类别净值曲线对比", auto_show=False)
    
    # 显示所有图表
    plt.show()
