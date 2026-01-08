# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_BL_RB.py
@Author : Sean
@Date   : 2025-11-08

@Description:
This script implements and backtests a Black-Litterman Risk-Budgeting (BL-RB) 
strategy. The core idea is to use the Black-Litterman model to generate a "tilting" 
signal, which then adjusts a baseline Risk Parity (RP) portfolio's risk budgets. 
The final portfolio weights are determined by optimizing towards these new target 
risk budgets.
"""

"""
# ==============================================================================
# 阶段 0: 核心库导入
# ==============================================================================
"""

import pandas as pd
import numpy as np
from typing import Callable, Optional, Sequence, Mapping, Any, Tuple, List
from sklearn.covariance import OAS
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from framework.plotting import plot_equity_curve, plot_weights_history
from framework.performance import compute_portfolio_returns_and_equity, calculate_performance_metrics
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
    load_returns_and_aum,
    load_category_returns,
    compute_log_returns,
    load_market_aum,
)
from framework.logging_config import setup_logging, get_logger

# 初始化统一日志（INFO级别，可根据需要调整）
setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_BL_RB.log")
logger = get_logger("strategy.BL_RB")






"""
# ==============================================================================
# 阶段 1: 全局参数与策略设定
# ==============================================================================
"""
# --- 2.1 策略核心设定 ---
STRATEGY_MODE = 'BL_RB_TILT_FROM_RP' # (BL 倾斜的风险平价)

# --- 2.2 回测通用参数 (从 config 引用默认值，可被策略局部覆盖) ---
# RISK_FREE_RATE_SCALAR_ANNUAL, COST_PER_SIDE_SCALAR, TRADING_DAYS_PER_YEAR_SCALAR 已从 framework.config 导入

# --- 2.3 Black-Litterman (BL) 模型参数 ---
PARAM_LAMBDA_SCALAR = 2.5   # 风险厌恶系数 (用于计算市场均衡收益率Pi)
PARAM_TAU_SCALAR = 0.05     # 先验不确定性因子 (衡量基础模型Sigma的不确定性)

# --- 2.4 投资观点 (Views) ---
# P 矩阵 (K x N): 定义了 K 个观点，关联 N 个资产
# 每行代表一个观点，+1为多头，-1为空头
P_matrix = np.array([
    [0, 1, -1, 0, 0], # 观点1: 科创50(1) > 创业板指(2)
    [1, 0, 0, -1, 0], # 观点2: 沪深300(0) > 5年国债(3)
    [0, 0, 0, 0, 1]  # 观点3: 黄金(4)
])
# Q 向量 (K x 1): 每个观点的预期超额收益 (年化)
Q_annual_vector = np.array([0.02, 0.05, 0.06])

# 信心水平向量 (K x 1): 基于 Idzorek 方法，代表对每个观点的信心
confidences_vector = np.array([0.60, 0.80, 0.50]) 

# --- 2.5 风险预算倾斜 (Risk Budget Tilting) 参数 ---
# 机构指导法(Institutional Guidance)的参数
PARAM_BETA_VECTOR = confidences_vector  # 使用信心水平作为观点混合权重
PARAM_GAMMA_SCALAR = 0.1                # 倾斜强度总开关: 限制信号对基准预算的最大影响（此处不超过10%）
PARAM_TANH_K_SCALAR = 2.0               # Tanh压缩参数: 用于平滑和限制极端信号
MIN_BUDGET_SCALAR = 1e-10                # 目标风险预算的数值下限，防止为零或负


"""
# ==============================================================================
# 阶段 2: 核心引擎函数定义
# ==============================================================================
"""

# --- 3.1 Black-Litterman (BL) 核心函数 ---

def calculate_idzorek_omega_daily_matrix(p_matrix_live, sigma_daily_matrix, tau_scalar, confidences_vector_live):
    """
    根据 Idzorek 方法计算观点不确定性矩阵 Omega (日度)。
    Omega 对角线上的值代表每个观点Q的不确定性(方差)。
    信心越高，对应的Omega值越小。
    
    Args:
        p_matrix_live (np.array): 当前有效的观点矩阵 P
        sigma_daily_matrix (np.array): 资产的日度协方差矩阵 Sigma
        tau_scalar (float): 先验不确定性因子 Tau
        confidences_vector_live (np.array): 当前有效的信心水平向量
        
    Returns:
        np.array: 对角阵 Omega (日度)
    """
    # 观点的先验协方差: P * (tau*Sigma) * P'
    prior_view_cov_matrix = p_matrix_live @ (tau_scalar * sigma_daily_matrix) @ p_matrix_live.T
    # 假定观点独立，只取对角线元素
    prior_view_variance_diag_vector = np.diag(prior_view_cov_matrix) 

    # Idzorek 核心: alpha = (1-C)/C
    # 限制信心 C 在 (0.5, 1) 开区间内以保证 alpha > 0
    c_vector = np.clip(confidences_vector_live, 0.5, 1 - 1e-6) 
    adjustment_factor_vector = (1 - c_vector) / c_vector

    # Omega 的对角线元素 = alpha * 观点的先验方差    
    omega_diag_idzorek_vector = adjustment_factor_vector * prior_view_variance_diag_vector
    return np.diag(omega_diag_idzorek_vector)

def black_litterman_posterior_mean_daily(sigma_daily_matrix, pi_daily_vector, p_live_matrix, q_daily_vector, omega_daily_matrix, tau_scalar):
    """
    计算 Black-Litterman 模型的后验期望收益率向量 (mu_bl)。
    这是市场均衡收益率(Pi)和投资者观点(P, Q)的加权平均。
    
    Args:
        sigma_daily_matrix (np.array): 资产日度协方差矩阵 Sigma
        pi_daily_vector (np.array): 市场均衡日度收益率向量 Pi
        p_live_matrix (np.array): 有效观点矩阵 P
        q_daily_vector (np.array): 有效观点日度收益率向量 Q
        omega_daily_matrix (np.array): 观点不确定性日度矩阵 Omega
        tau_scalar (float): 先验不确定性因子 Tau
        
    Returns:
        np.array: 后验日度期望收益率向量 mu_bl
    """        
    try:
        # BL核心公式: mu_bl = ( (tau*Sigma)^-1 + P'*Omega^-1*P )^-1 * ( (tau*Sigma)^-1*Pi + P'*Omega^-1*Q )
        inv_tau_sigma_matrix = np.linalg.inv(tau_scalar * sigma_daily_matrix)
        inv_omega_matrix = np.linalg.inv(omega_daily_matrix)
        # 计算公式左半部分
        a_matrix = inv_tau_sigma_matrix + p_live_matrix.T @ inv_omega_matrix @ p_live_matrix
        # 计算公式右半部分
        b_vector = inv_tau_sigma_matrix @ pi_daily_vector + p_live_matrix.T @ inv_omega_matrix @ q_daily_vector

        # 求解线性方程组 A * x = B，得到 mu_bl
        mu_bl_daily_vector = np.linalg.solve(a_matrix, b_vector)
        return mu_bl_daily_vector
    except np.linalg.LinAlgError as e:
        logger.warning(f"BL 矩阵计算失败: {e}。退回使用先验 Pi。")
        return pi_daily_vector


# --- 3.x 视图构造辅助：按资产名构建 P/Q/C 并映射列顺序 ---
def _normalize_weights(items: Sequence[str] | Mapping[str, float]) -> Mapping[str, float]:
    """将 long/short 项转成带权重的字典并归一化到和为1。
    支持两种输入：
      - 列表 ['A','B'] -> {'A':0.5,'B':0.5}
      - 字典 {'A':2,'B':1} -> 归一化为 {'A':2/3,'B':1/3}
    """
    if isinstance(items, Mapping):
        total = float(sum(float(v) for v in items.values()))
        if total <= 0:
            return {k: 0.0 for k in items.keys()}
        return {str(k): float(v) / total for k, v in items.items()}
    else:
        n = len(items)
        if n == 0:
            return {}
        w = 1.0 / n
        return {str(k): w for k in items}


def build_view_row_by_names(all_assets: Sequence[str], long_items: Sequence[str] | Mapping[str, float], short_items: Sequence[str] | Mapping[str, float]) -> np.ndarray:
    """根据资产名构造一行 P（长度等于 all_assets）。
    - long_items：做多资产名列表或字典（权重可不等，内部会归一化为和为1）
    - short_items：做空资产名列表或字典（同上）
    结果：对多头列填 +w，对空头列填 -w；其他填 0。
    不在 all_assets 的名称将被忽略。
    """
    long_w = _normalize_weights(long_items)
    short_w = _normalize_weights(short_items)
    p = np.zeros(len(all_assets), dtype=float)
    name_to_idx = {name: i for i, name in enumerate(all_assets)}
    for name, w in long_w.items():
        idx = name_to_idx.get(name)
        if idx is not None:
            p[idx] += float(w)
    for name, w in short_w.items():
        idx = name_to_idx.get(name)
        if idx is not None:
            p[idx] -= float(w)
    return p


def assemble_views_from_named(all_assets: Sequence[str], view_specs: Sequence[Mapping[str, Any]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """将按资产名的观点规格转换为 (P, Q_annual, C)。

    view_specs 中每个元素示例：
      {
        'long': ['沪深300','科创50'] 或 {'沪深300': 2, '科创50': 1},
        'short': ['创业板指'] 或 {'创业板指': 1},
        'Q': 0.03,           # 年化超额收益观点
        'confidence': 0.7    # (0,1)
      }

    返回：
      P: (K, N), Q:(K,), C:(K,) ；N 等于 all_assets 长度。
    """
    rows: List[np.ndarray] = []
    Q_list: List[float] = []
    C_list: List[float] = []
    for spec in view_specs:
        long_items = spec.get('long', [])
        short_items = spec.get('short', [])
        q = float(spec.get('Q', 0.0))
        c = float(spec.get('confidence', 0.6))
        row = build_view_row_by_names(all_assets, long_items, short_items)
        # 跳过全零行（可能由于名称不在资产池内）
        if np.allclose(row, 0.0):
            continue
        rows.append(row)
        Q_list.append(q)
        C_list.append(c)
    if not rows:
        return np.zeros((0, len(all_assets))), np.zeros((0,)), np.zeros((0,))
    P = np.vstack(rows)
    Q = np.array(Q_list, dtype=float)
    C = np.array(C_list, dtype=float)
    return P, Q, C

# --- 3.2 风险预算 (Risk Budgeting, RB) 核心函数 ---

def get_risk_contributions_daily(weights_vector, sigma_daily_matrix):
    """
    计算给定权重下，各资产对组合总风险的贡献度 (Risk Contribution, RC)。
    RC_i = w_i * MRC_i = w_i * (d(Port_Vol) / d(w_i))
    
    Args:
        weights_vector (np.array): 资产权重向量
        sigma_daily_matrix (np.array): 资产日度协方差矩阵
        
    Returns:
        np.array: 各资产的风险贡献度向量 (以百分比表示)
    """
    # 确保权重为正且归一化，以保证数值稳定性
    weights_vector = np.maximum(weights_vector, 0)
    if weights_vector.sum() < 1e-8:
        return np.full_like(weights_vector, 1.0 / len(weights_vector))
    weights_vector = weights_vector / weights_vector.sum()

    # 组合总方差: w' * Sigma * w    
    port_variance_scalar = weights_vector.T @ sigma_daily_matrix @ weights_vector   
    if port_variance_scalar < 1e-10:
        return np.full_like(weights_vector, 1.0 / len(weights_vector))
    
    # 边际风险贡献 (Marginal Risk Contribution, MRC): Sigma * w    
    mrc_vector = sigma_daily_matrix @ weights_vector
    # 风险贡献 (绝对值): w_i * MRC_i    
    rc_vector = weights_vector * mrc_vector    

    # 风险贡献 (百分比): RC_i / Port_Variance
    rc_percent_vector = rc_vector / port_variance_scalar
    return rc_percent_vector

def risk_budget_objective_function(weights_vector, sigma_daily_matrix, target_budget_vector): 
    """
    风险预算优化器的目标函数。
    目标是最小化 "实际风险贡献" 与 "目标风险预算" 之间的平方差。
    
    Args:
        weights_vector (np.array): 当前迭代的权重 (优化变量)
        sigma_daily_matrix (np.array): 协方差矩阵
        target_budget_vector (np.array): 目标风险预算向量
        
    Returns:
        float: 实际RC与目标RC的平方差之和
    """    
    # 计算当前权重下的实际风险贡献
    rc_percent_vector = get_risk_contributions_daily(weights_vector, sigma_daily_matrix)

    # 如果未指定目标，则默认为风险平价 (Risk Parity, RP)
    if target_budget_vector is None:
        target_contribution = 1.0 / sigma_daily_matrix.shape[0]
    else:
        target_contribution = target_budget_vector
    
    # 计算平方差，这是我们要最小化的值
    sum_sq_diff_scalar = np.sum((rc_percent_vector - target_contribution)**2)
    return sum_sq_diff_scalar

def calculate_risk_budget_weights_scipy(sigma_daily_matrix, target_budget_vector=None):
    """
    使用 Scipy 的 SLSQP 求解器来计算满足目标风险预算的资产权重。
    
    Args:
        sigma_daily_matrix (np.array): 协方差矩阵
        target_budget_vector (np.array, optional): 目标风险预算. Defaults to None (即风险平价).
        
    Returns:
        np.array: 优化后的资产权重向量
    """
    n_assets = sigma_daily_matrix.shape[0]
    # 初始猜测权重：等权重
    initial_weights_vector = np.array([1/n_assets] * n_assets)
    # 约束条件:
    # 1. 权重之和为 1   
    constraints = ({'type': 'eq', 'fun': lambda weights_vector: np.sum(weights_vector) - 1})
    # 2. 每个权重在 [0, 1] 之间 (不允许做空)
    bounds = tuple((0, 1) for _ in range(n_assets)) # 严格 W_i >= 0

    # 调用 minimize 函数    
    result = minimize(
        risk_budget_objective_function, # [!] 使用平方差目标
        initial_weights_vector,         
        args=(sigma_daily_matrix, 
              target_budget_vector),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        tol=1e-10 # [!] 增加容忍度
    )
    
    if result.success:
        # 优化成功，返回结果并做最后的清理
        w_final = result.x
        w_final[w_final < 1e-10] = 0 # 清除数值噪声
        w_final = w_final / w_final.sum() # 重新归一
        return w_final
    else:
        logger.warning(f"Scipy 风险预算优化失败: {result.message}。转为等权重。")
        return initial_weights_vector



"""
# ==============================================================================
# 阶段 4: 滚动回测主循环
# ==============================================================================
"""
# ============================================================================
# 统一接口：run_strategy (Black-Litterman Risk-Budgeting)
# ----------------------------------------------------------------------------
def run_strategy(
    returns_path: str | None = None,
    aum_path: str = 'proxy_etfs_aum_monthly.csv',
    *,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    tau_scalar: float = PARAM_TAU_SCALAR,
    lambda_scalar: float = PARAM_LAMBDA_SCALAR,
    tanh_k_scalar: float = PARAM_TANH_K_SCALAR,
    gamma_scalar: float = PARAM_GAMMA_SCALAR,
    p_matrix_override: np.ndarray | None = None,
    q_annual_override: np.ndarray | None = None,
    confidences_override: np.ndarray | None = None,
    view_provider: Optional[Callable[[pd.Timestamp, Sequence[str], np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]] = None,
    data_dir: str = 'data',
    categories: Optional[Sequence[str]] = None,
    auto_plot: bool = False,
) -> dict:
    """运行 BL_RB 策略，返回标准化结果字典。

    返回 keys:
      - weights_history_df, rc_history_df
      - final_portfolio_returns_series, equity_curve_series
      - performance_report, total_cost
      - figures (可选)
    """
# ==============================================================================
# 阶段 1: 数据加载与预处理
# ==============================================================================

    # 分类模式：从 data_dir 读取各类并按所选类别合并
    cat_simple = load_category_returns(data_dir)
    cat_log = {k: (compute_log_returns(v) if not v.empty else v) for k, v in cat_simple.items()}
    chosen = list(cat_simple.keys()) if not categories else list(categories)
    simple_df = None
    log_df = None
    for c in chosen:
        df_s = cat_simple.get(c)
        df_l = cat_log.get(c)
        if df_s is None or df_s.empty:
            continue
        simple_df = df_s if simple_df is None else simple_df.join(df_s, how='outer')
        log_df = df_l if log_df is None else log_df.join(df_l, how='outer')
    if simple_df is None or log_df is None:
        raise ValueError(f"选定的分类 {chosen} 中没有任何有效资产列")
    # 加载并对齐 AUM 到 log 索引
    aum_daily_df = load_market_aum(aum_path, target_index=log_df.index)
    assets = log_df.columns.tolist()
    if start or end:
        log_df = log_df.loc[start:end]
        simple_df = simple_df.loc[log_df.index]
        aum_daily_df = aum_daily_df.reindex(log_df.index, method='ffill')

    # 视图参数来源：若提供外部覆盖则使用之；否则使用模块内默认
    P_src = P_matrix
    Q_src = Q_annual_vector
    C_src = confidences_vector
    try:
        if p_matrix_override is not None:
            if p_matrix_override.ndim != 2 or p_matrix_override.shape[1] != len(assets):
                logger.warning(
                    f"p_matrix_override 形状不匹配，期望列数={len(assets)}，实际={getattr(p_matrix_override, 'shape', None)}；将忽略覆盖。"
                )
            else:
                P_src = p_matrix_override
        if q_annual_override is not None:
            if q_annual_override.ndim != 1 or q_annual_override.shape[0] != P_src.shape[0]:
                logger.warning(
                    f"q_annual_override 长度不匹配，期望={P_src.shape[0]}，实际={getattr(q_annual_override, 'shape', None)}；将忽略覆盖。"
                )
            else:
                Q_src = q_annual_override
        if confidences_override is not None:
            if confidences_override.ndim != 1 or confidences_override.shape[0] != P_src.shape[0]:
                logger.warning(
                    f"confidences_override 长度不匹配，期望={P_src.shape[0]}，实际={getattr(confidences_override, 'shape', None)}；将忽略覆盖。"
                )
            else:
                C_src = confidences_override
    except Exception as _e:
        logger.warning(f"视图参数覆盖校验异常：{_e}；将使用默认视图参数。")


    lookback_window = TRADING_DAYS_PER_YEAR_SCALAR * lookback_years
    min_data_days_required = TRADING_DAYS_PER_YEAR_SCALAR * min_data_years
    rebalance_dates = log_df.groupby(log_df.index.to_period('M')).tail(1).index
    first_valid = log_df.index[0] + pd.DateOffset(days=min_data_days_required)
    rebalance_dates = rebalance_dates[(rebalance_dates >= first_valid) & (rebalance_dates <= log_df.index.max())]

    weights_hist = pd.DataFrame(index=rebalance_dates, columns=assets)
    rc_hist = pd.DataFrame(index=rebalance_dates, columns=assets)

    oas_loc = OAS(assume_centered=True)
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
            oas_loc.fit(clean_live_df)
            sigma = oas_loc.covariance_
            # 市场组合权重
            current_aum_vector = aum_daily_df.loc[date, live_assets].fillna(0)
            if current_aum_vector.sum() < 1e-6:
                w_mkt = np.array([1.0 / n_live] * n_live)
            else:
                w_mkt = (current_aum_vector / current_aum_vector.sum()).values
            pi_daily = lambda_scalar * sigma @ w_mkt

            # 有效观点筛选
            live_indices = [assets.index(a) for a in live_assets]
            live_idx_set = set(live_indices)

            # 动态观点：如果提供 view_provider，用它生成当期 P/Q/C（已按资产顺序构造）
            if view_provider is not None:
                try:
                    dyn_P, dyn_Q, dyn_C = view_provider(date, assets, sigma, pi_daily)
                    if dyn_P is not None and dyn_P.ndim == 2 and dyn_P.shape[1] == len(assets):
                        if dyn_Q.shape[0] == dyn_P.shape[0] and dyn_C.shape[0] == dyn_P.shape[0]:
                            P_eff = dyn_P
                            Q_eff = dyn_Q
                            C_eff = dyn_C
                        else:
                            logger.warning(f"view_provider 返回 Q/C 维度不匹配，使用静态视图。")
                            P_eff = P_src
                            Q_eff = Q_src
                            C_eff = C_src
                    else:
                        logger.warning(f"view_provider 返回 P 形状不合法，使用静态视图。")
                        P_eff = P_src
                        Q_eff = Q_src
                        C_eff = C_src
                except Exception as e_vp:
                    logger.warning(f"view_provider 异常: {e_vp}，使用静态视图。")
                    P_eff = P_src
                    Q_eff = Q_src
                    C_eff = C_src
            else:
                P_eff = P_src
                Q_eff = Q_src
                C_eff = C_src

            valid_view_idx = [j for j in range(P_eff.shape[0]) if set(np.where(P_eff[j] != 0)[0]).issubset(live_idx_set)]
            if valid_view_idx:
                p_live = P_eff[valid_view_idx][:, live_indices]
                q_annual = Q_eff[valid_view_idx]
                q_daily = q_annual / TRADING_DAYS_PER_YEAR_SCALAR
                conf = C_eff[valid_view_idx]
                omega_daily = calculate_idzorek_omega_daily_matrix(p_live, sigma, tau_scalar, conf)
                mu_bl = black_litterman_posterior_mean_daily(sigma, pi_daily, p_live, q_daily, omega_daily, tau_scalar)
            else:
                mu_bl = pi_daily

            # RP 基准权重
            w_rp = calculate_risk_budget_weights_scipy(sigma, None)
            # 主动信号
            try:
                sigma_inv = np.linalg.inv(sigma)
                return_active = mu_bl - pi_daily
                w_active = sigma_inv @ return_active
                mrc_mkt = sigma @ w_mkt
                delta_rc = w_active * mrc_mkt
            except np.linalg.LinAlgError:
                delta_rc = np.zeros_like(w_rp)

            annual_var_delta = delta_rc * TRADING_DAYS_PER_YEAR_SCALAR
            annual_vol_delta = np.sign(annual_var_delta) * np.sqrt(np.abs(annual_var_delta))
            avg_abs = np.mean(np.abs(delta_rc))
            if avg_abs < 1e-10:
                delta_rc_squashed = np.zeros(n_live)
            else:
                delta_rc_scaled = delta_rc / avg_abs
                delta_rc_squashed = np.tanh(tanh_k_scalar * delta_rc_scaled)

            # 预算倾斜
            b_rp = np.array([1.0 / n_live] * n_live)
            tilted = b_rp + gamma_scalar * delta_rc_squashed
            if valid_view_idx:
                beta_scalar = (C_eff[valid_view_idx].mean() if len(valid_view_idx) > 0 else 0.0)
            else:
                beta_scalar = 0.0
            if tilted.sum() < 1e-6:
                b_mkt_vec = get_risk_contributions_daily(w_mkt, sigma)
                b_mkt_vec = np.maximum(b_mkt_vec, 0)
                b_tilted = (b_mkt_vec / b_mkt_vec.sum()) if b_mkt_vec.sum() > 1e-6 else b_rp.copy()
            else:
                b_tilted = tilted / tilted.sum()
            b_target = (1 - beta_scalar) * b_rp + beta_scalar * b_tilted

            w_opt = calculate_risk_budget_weights_scipy(sigma, target_budget_vector=b_target)
            rc_vector = get_risk_contributions_daily(w_opt, sigma)
            weights_hist.loc[date] = 0.0
            weights_hist.loc[date, live_assets] = w_opt
            rc_hist.loc[date] = 0.0
            rc_hist.loc[date, live_assets] = rc_vector
        except Exception:
            fallback = np.array([1.0 / n_live] * n_live)
            weights_hist.loc[date] = 0.0
            weights_hist.loc[date, live_assets] = fallback
            rc_hist.loc[date] = 0.0

    weights_hist = weights_hist.fillna(0)
    rc_hist = rc_hist.fillna(0)



# ==============================================================================
# 阶段 5: 投资组合绩效回测并展示绩效指标
# ==============================================================================
    port_ret, equity, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=weights_hist,
        daily_returns_df=simple_df,
        cost_per_side_scalar=cost_per_side,
    )
    equity.name = STRATEGY_MODE
    report = calculate_performance_metrics(port_ret, equity, rf_rate, TRADING_DAYS_PER_YEAR_SCALAR)

    total_cost_series = (weights_hist.diff().abs().sum(axis=1).astype('float64') * cost_per_side).shift(1)
    total_cost_series = total_cost_series.reindex(port_ret.index).astype('float64').fillna(0.0)
    total_cost = total_cost_series.sum()
# ==============================================================================
# 阶段 6: 可视化 - 净值曲线（调用通用模块） + 仓位历史 (调用通用模块)
# ==============================================================================
    figures = {}
    if auto_plot:
        fig1 = plot_equity_curve(equity, report, STRATEGY_MODE, auto_show=False)
        fig2 = plot_weights_history(weights_hist, STRATEGY_MODE, reverse_legend=True, auto_show=False)
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
    
if __name__ == "__main__":
    # 简化入口：不使用命令行参数，直接以默认参数运行一次，便于单文件快速回测
    setup_logging(level="INFO")
    logger.info("运行 BL_RB（默认参数，分类数据加载，使用 data 目录全量类别）…")
    res = run_strategy(
        returns_path=None,                    # 从 data 目录按分类读取
        aum_path='proxy_etfs_aum_monthly.csv',
        start=None,
        end=None,
        data_dir='data',
        categories=None,                     # None 表示使用所有可用分类
        auto_plot=True,
    )
    perf = res['performance_report']
    logger.info("运行完成，主要指标：")
    for k, v in perf.items():
        if any(x in k for x in ['CAGR','Vol','回撤']):
            logger.info(f"  {k}: {v:.2%}")
        else:
            logger.info(f"  {k}: {v:.4f}")