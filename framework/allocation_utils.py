# -*- coding: utf-8 -*-
"""Common allocation utility functions: rebalance date selection, factor choice,
risk parity solver, factor-to-asset mapping, and HRP clustering allocation.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, List, Literal
import scipy.cluster.hierarchy as sch
from scipy.optimize import minimize
try:
    from sklearn.covariance import OAS, LedoitWolf
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from framework.config import EPSILON_NUMERIC, TRADING_DAYS_PER_YEAR_SCALAR
from framework.logging_config import get_logger

logger = get_logger("framework.allocation_utils")

# -----------------------------
# Covariance estimation utilities
# -----------------------------

def estimate_covariance_matrix(
    returns_df: pd.DataFrame,
    method: str = 'cov',
    ewm_span: int = 252,
    min_negative_samples: int = 20,
    shrinkage_target: str = 'diagonal',
    shrinkage_intensity: Optional[float] = 0.5
) -> pd.DataFrame:
    """
    估计协方差矩阵，支持多种方法。
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        收益率时间序列数据框，每列代表一个资产
    method : str, default 'cov'
        估计方法：
        - 'cov': 标准样本协方差
        - 'ewm': 指数加权移动协方差
        - 'downside_cov': 下半协方差（仅使用负收益）
        - 'downside_ewm': 下半指数加权移动协方差
        - 'oas': Oracle Approximating Shrinkage (sklearn)
        - 'shrunk': 手动收缩估计 (Manual Shrinkage)
    ewm_span : int, default 252
        指数加权的半衰期（仅用于 'ewm' 和 'downside_ewm'）
    min_negative_samples : int, default 20
        下半协方差所需的最小负收益样本数
    shrinkage_target : str, default 'diagonal'
        收缩目标（仅用于 method='shrunk'）：
        - 'diagonal': 收缩向对角阵（保留方差，协方差为0）
        - 'constant_correlation': 收缩向常相关系数矩阵
        - 'identity': 收缩向单位阵（方差均值 * I）
    shrinkage_intensity : float, default 0.5
        收缩强度 (0~1)，0为不收缩，1为完全收缩至目标。
        
    Returns
    -------
    pd.DataFrame
        协方差矩阵，索引和列为资产名称
        
    Raises
    ------
    ValueError
        如果 method 不是支持的方法之一
        如果下半协方差方法中负收益样本不足
    """
    if returns_df.empty:
        raise ValueError("returns_df 不能为空")
    
    method = method.lower()
    
    if method == 'cov':
        # 标准样本协方差
        cov_matrix = returns_df.cov()
        
    elif method == 'ewm':
        # 指数加权移动协方差
        # 取最后 N 行（N = 资产数量）形成协方差矩阵
        n_assets = len(returns_df.columns)
        cov_matrix = returns_df.ewm(span=ewm_span).cov().iloc[-n_assets:]
        # ewm().cov() 返回 MultiIndex (Date, Asset)，需要去掉 Date 层级
        if isinstance(cov_matrix.index, pd.MultiIndex):
            cov_matrix = cov_matrix.droplevel(0)
        
    elif method == 'downside_cov':
        # 下半协方差：仅使用负收益率计算
        # 对每个资产，将正收益替换为0
        downside_returns = returns_df.copy()
        downside_returns[downside_returns > 0] = 0
        
        # 检查每个资产的负收益样本数
        neg_counts = (downside_returns < 0).sum()
        if (neg_counts < min_negative_samples).any():
            insufficient_assets = neg_counts[neg_counts < min_negative_samples].index.tolist()
            # 仅在样本数极少时警告，避免刷屏
            if (neg_counts < 5).any():
                logger.warning(
                    f"下半协方差估计：部分资产负收益样本极少 (<5)，"
                    f"资产列表: {insufficient_assets}，样本数: {neg_counts[insufficient_assets].to_dict()}"
                )
        
        # 计算下半协方差
        cov_matrix = downside_returns.cov()
        
    elif method == 'downside_ewm':
        # 下半指数加权移动协方差
        downside_returns = returns_df.copy()
        downside_returns[downside_returns > 0] = 0
        
        # 检查负收益样本数
        neg_counts = (downside_returns < 0).sum()
        if (neg_counts < min_negative_samples).any():
            insufficient_assets = neg_counts[neg_counts < min_negative_samples].index.tolist()
            if (neg_counts < 5).any():
                logger.warning(
                    f"下半EWM协方差估计：部分资产负收益样本极少 (<5)，"
                    f"资产列表: {insufficient_assets}，样本数: {neg_counts[insufficient_assets].to_dict()}"
                )
        
        # 计算下半EWM协方差
        n_assets = len(returns_df.columns)
        cov_matrix = downside_returns.ewm(span=ewm_span).cov().iloc[-n_assets:]
        # ewm().cov() 返回 MultiIndex (Date, Asset)，需要去掉 Date 层级
        if isinstance(cov_matrix.index, pd.MultiIndex):
            cov_matrix = cov_matrix.droplevel(0)
            
    elif method == 'oas':
        if not SKLEARN_AVAILABLE:
            raise ImportError("使用 'oas' 方法需要安装 scikit-learn")
        # OAS 自动确定收缩强度，目标是 Scaled Identity
        oas = OAS()
        oas.fit(returns_df)
        cov_matrix = pd.DataFrame(oas.covariance_, index=returns_df.columns, columns=returns_df.columns)
    elif method == 'corr_oas':
        returns_df.std().values
        current_vols = returns_df.std().values
        # 标准化数据 (使其方差为1，满足OAS假设)
        X_centered = returns_df - returns_df.mean()
        X_std = X_centered / returns_df.std()
        # OAS 估计相关性矩阵
        # fit 之后，covariance_ 就是收缩后的相关系数矩阵
        oas = OAS(assume_centered=True).fit(X_std)
        shrunk_corr = oas.covariance_ 
        
        # 3. 还原协方差: Cov = Vol * Corr_OAS * Vol
        # Outer product of vols creates the volatility matrix
        vol_matrix = np.outer(current_vols, current_vols)
        cov_matrix = pd.DataFrame(shrunk_corr * vol_matrix, index=returns_df.columns, columns=returns_df.columns)
    elif method == 'shrunk':
        # 手动收缩估计
        # S_shrunk = (1 - delta) * S_sample + delta * F_target
        sample_cov = returns_df.cov()
        delta = shrinkage_intensity if shrinkage_intensity is not None else 0.5
        
        if shrinkage_target == 'diagonal':
            # 目标：对角阵（保留方差，协方差为0）
            target = np.diag(np.diag(sample_cov))
        elif shrinkage_target == 'constant_correlation':
            # 目标：常相关系数矩阵
            # 1. 计算样本相关系数矩阵
            std = np.sqrt(np.diag(sample_cov))
            corr = sample_cov.values / np.outer(std, std)
            # 2. 计算平均相关系数 (排除对角线)
            np.fill_diagonal(corr, np.nan)
            avg_corr = np.nanmean(corr)
            # 3. 构建常相关系数矩阵
            const_corr = np.full_like(corr, avg_corr)
            np.fill_diagonal(const_corr, 1.0)
            # 4. 还原为协方差矩阵
            target = const_corr * np.outer(std, std)
        elif shrinkage_target == 'identity':
            # 目标：单位阵 * 平均方差
            avg_var = np.trace(sample_cov) / len(sample_cov)
            target = np.eye(len(sample_cov)) * avg_var
        else:
            raise ValueError(f"不支持的收缩目标: {shrinkage_target}")
            
        # 执行收缩
        cov_values = (1 - delta) * sample_cov.values + delta * target
        cov_matrix = pd.DataFrame(cov_values, index=returns_df.columns, columns=returns_df.columns)
        
    else:
        raise ValueError(
            f"不支持的协方差估计方法: {method}. "
            f"支持的方法: 'cov', 'ewm', 'downside_cov', 'downside_ewm', 'oas', 'shrunk'"
        )
    
    # 确保协方差矩阵是正定的
    # 1. 基础正则化：防止因全零列（无下行风险）导致的奇异矩阵
    # 对于下半协方差，无负收益是常见情况，不应视为错误
    cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * 1e-12

    # 2. 检查特征值，如果仍非正定，则加强正则化
    min_eigenval = np.linalg.eigvalsh(cov_matrix.values).min()
    if min_eigenval < EPSILON_NUMERIC:
        logger.warning(
            f"协方差矩阵非正定 (最小特征值={min_eigenval:.2e})，"
            f"添加额外正则化项 {EPSILON_NUMERIC}"
        )
        cov_matrix = cov_matrix + np.eye(len(cov_matrix)) * EPSILON_NUMERIC
    
    return cov_matrix

# -----------------------------
# Rebalance date utilities
# -----------------------------

def to_daily_rebalance_dates(log_df: pd.DataFrame, min_data_days_required: int, rebalance_day: int = None) -> pd.DatetimeIndex:
    """
    生成每月换仓日序列。
    - rebalance_day=None: 默认每月月末最后一个交易日
    - rebalance_day=5: 每月5号或该日前最近的一个交易日
    """
    idx = log_df.index
    if rebalance_day is None:
        # 默认月末
        rebalance_dates = log_df.groupby(idx.to_period('M')).tail(1).index
    else:
        # 每月指定几号
        rebalance_dates = []
        months = pd.period_range(idx.min(), idx.max(), freq='M')
        for m in months:
            # 该月所有交易日
            month_days = idx[(idx >= m.start_time) & (idx <= m.end_time)]
            if len(month_days) == 0:
                continue
            # 目标日
            target = pd.Timestamp(year=m.year, month=m.month, day=rebalance_day)
            # 若该日大于月末，则取月末
            if target > m.end_time:
                target = m.end_time
            # 过滤小于等于目标日的所有交易日
            valid_days = month_days[month_days <= target]
            if len(valid_days) == 0:
                # 该月目标日前无交易日，跳过
                continue
            # 取目标日前最近的一个交易日
            rebalance_dates.append(valid_days[-1])
        rebalance_dates = pd.DatetimeIndex(rebalance_dates)
    first_valid_rebalance_date = idx[0] + pd.DateOffset(days=min_data_days_required)
    rebalance_dates = rebalance_dates[(rebalance_dates >= first_valid_rebalance_date) & (rebalance_dates <= idx.max())]
    return rebalance_dates

# -----------------------------
# PCA factor number selection
# -----------------------------

def choose_num_factors(evals_vector: np.ndarray, *, n_factors: Optional[int], explained_var_threshold: float, max_factors: int) -> int:
    evals_vector = np.clip(evals_vector.astype(float), 0.0, None)
    n = len(evals_vector)
    if n <= 0:
        return 0
    if n_factors is not None:
        return int(max(0, min(n_factors, n)))
    total = float(evals_vector.sum())
    if total <= 0:
        return 0
    order = np.argsort(evals_vector)[::-1]
    cum = np.cumsum(evals_vector[order]) / total
    k = int(np.searchsorted(cum, explained_var_threshold) + 1)
    k = max(1, min(k, n, max_factors))
    return k

# -----------------------------
# Risk parity (asset or factor) utilities
# -----------------------------

def get_risk_contributions_daily(weights_vector: np.ndarray, sigma_daily_matrix: np.ndarray) -> np.ndarray:
    w = np.maximum(weights_vector, 0)
    s = w.sum()
    if s <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    w = w / s
    port_var = float(w.T @ sigma_daily_matrix @ w)
    if port_var <= 1e-12:
        return np.full_like(w, 1.0 / len(w))
    mrc = sigma_daily_matrix @ w
    rc = w * mrc
    return rc / port_var


def _rb_objective(x: np.ndarray, sigma: np.ndarray, target: Optional[np.ndarray]) -> float:
    rc = get_risk_contributions_daily(x, sigma)
    tgt = (1.0 / sigma.shape[0]) if target is None else target
    return float(np.sum((rc - tgt) ** 2))


def solve_risk_parity_weights(sigma_daily_matrix_df: pd.DataFrame, target_budget_vector: Optional[pd.Series] = None) -> pd.Series:
    if target_budget_vector is not None:
        target_budget_vector = target_budget_vector.loc[sigma_daily_matrix_df.index]
        target_budget_vector = target_budget_vector.values
    sigma_daily_matrix = sigma_daily_matrix_df.values
    n = sigma_daily_matrix.shape[0]
    x0 = np.array([1.0 / n] * n)
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0},)
    bnds = tuple((0.0, 1.0) for _ in range(n))
    res = minimize(_rb_objective, x0, args=(sigma_daily_matrix, target_budget_vector), method='SLSQP', bounds=bnds, constraints=cons, tol=1e-10)
    if res.success:
        w = res.x
        w[w < 1e-10] = 0.0
        s = w.sum()
        w_final =  pd.Series(w / s, index=sigma_daily_matrix_df.columns)        
        return w_final
    logger.warning(f"风险平价优化失败: {res.message}；使用等权重。")
    return pd.Series(x0,index=sigma_daily_matrix_df.columns)

def solve_factor_risk_parity_weights(
    betas_df: pd.DataFrame,
    factor_cov_df: pd.DataFrame,
    *,
    target_factor_budget: Optional[pd.Series] = None,
    ridge_lambda: float = 1e-4,
    long_only: bool = True,
    weight_cap: Optional[float] = None,
) -> pd.Series:
    """严格“因子风险平价”：在资产权重 w 上优化，使因子风险贡献接近目标。

    定义：b_p = B^T w, 组合的因子方差为 b_p^T Σ_F b_p。
    使用 Euler 分解的（绝对值）贡献：RC_k = | b_k * (Σ_F b)_k |。

    约束：sum(w)=1；默认 long-only 且每资产 [0, cap]。
    """
    if betas_df is None or factor_cov_df is None:
        raise ValueError("betas_df / factor_cov_df 不能为空")
    if betas_df.empty or factor_cov_df.empty:
        raise ValueError("betas_df / factor_cov_df 不能为空")

    # 对齐
    factor_cols = [c for c in factor_cov_df.columns if c in betas_df.columns]
    if len(factor_cols) != factor_cov_df.shape[0]:
        factor_cov_df = factor_cov_df.loc[factor_cols, factor_cols]
    betas_df = betas_df.loc[:, factor_cols]

    asset_names = list(betas_df.index)
    n_assets = len(asset_names)
    k = len(factor_cols)
    if n_assets == 0 or k == 0:
        raise ValueError("betas_df / factor_cov_df 交集数量为0")

    # beta_matrix = betas_df.to_numpy(dtype=float)  # (N, K)
    # Sigma_f = factor_cov_df.to_numpy(dtype=float)  # (K, K)

    if target_factor_budget is None:
        tgt = np.full(k, 1.0 / k, dtype=float)
    else:
        t = target_factor_budget.reindex(factor_cols).astype(float).to_numpy()
        t = np.clip(t, 0.0, None)
        s = float(t.sum())
        tgt = (t / s) if s > 0 else np.full(k, 1.0 / k, dtype=float)

    x0 = np.full(n_assets, 1/n_assets, dtype=float)
    # x0 = np.zeros(n_assets, dtype=float)
    # x0[0] = 1.0

    # bounds
    if weight_cap is not None and weight_cap > 0:
        cap = float(weight_cap)
        if cap * n_assets < 1.0:
            cap = 1.0 / n_assets + 1e-9
    else:
        cap = 1.0

    if long_only:
        bnds = tuple((0.0, cap) for _ in range(n_assets))
    else:
        bnds = tuple((-cap, cap) for _ in range(n_assets))

    # 约束：权重和=1 + 因子暴露非负（确保同号暴露）
    cons = [
        {'type': 'eq', 'fun': lambda w: float(np.sum(w) - 1.0)},
        # 确保所有因子暴露 >= 0（避免一正一负的对冲解）
        *[{'type': 'ineq', 'fun': lambda w, i=i: float((betas_df.iloc[:, i] @ w))} 
          for i in range(k)]
    ]

    def obj(w: np.ndarray) -> float:
        w = np.asarray(w, dtype=float)
        b_p = betas_df.T @ w  # (K,)
        m = factor_cov_df @ b_p  # (K,)
        # 移除绝对值，直接使用 b_p * m（已确保 b_p >= 0）
        rc = b_p * m
        rc_sum = float(rc.sum())
        if rc_sum <= 1e-18:
            rc_norm = np.full_like(rc, 1.0 / k)
        else:
            rc_norm = rc / rc_sum

        loss = float(np.sum((rc_norm - tgt) ** 2))
        if ridge_lambda and ridge_lambda > 0:
            d = w - (1.0 / n_assets)
            loss += float(ridge_lambda * (d @ d))
        return loss

    res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=cons, tol=1e-10)
    if res.success and np.all(np.isfinite(res.x)):
        w = res.x.astype(float)
        if long_only:
            w[w < 1e-12] = 0.0
        else:
            w[np.abs(w) < 1e-12] = 0.0
        s = float(w.sum())
        if abs(s) > 1e-12:
            w = w / s
        return pd.Series(w, index=asset_names)

    logger.warning(f"因子风险平价优化失败: {res.message}；回退等权。")
    return pd.Series(x0, index=asset_names)


def estimate_factor_betas_ols(
    returns_df: pd.DataFrame,
    factors_df: pd.DataFrame,
    *,
    add_intercept: bool = True,
    min_obs: int = 36,
) -> tuple[pd.DataFrame, pd.Series]:
    """用 OLS 估计每个资产对因子的暴露(β)。

    参数
    - returns_df: (T, N) 资产收益
    - factors_df: (T, K) 因子序列（通常是 Change）
    - add_intercept: 是否包含截距
    - min_obs: 最小样本数（对齐并 dropna 后）

    返回
    - betas_df: (N, K) 行=资产, 列=因子
    - resid_var: (N,) 每个资产残差方差（用于诊断/可选特质项）
    """
    if returns_df is None or factors_df is None:
        raise ValueError("returns_df / factors_df 不能为空")
    if returns_df.empty or factors_df.empty:
        raise ValueError("returns_df / factors_df 不能为空")

    # 对齐样本
    df = returns_df.join(factors_df, how='inner')
    df = df.dropna(how='any')
    if len(df) < min_obs:
        raise ValueError(f"样本不足: {len(df)} < {min_obs}")

    try:
        import statsmodels.api as sm
    except Exception as e:
        raise ImportError("estimate_factor_betas_ols 需要 statsmodels (sm.OLS)。") from e

    asset_cols = list(returns_df.columns)
    factor_cols = list(factors_df.columns)

    betas_df = pd.DataFrame(index=asset_cols, columns=factor_cols, dtype=float)
    tvalue_df = pd.DataFrame(index=asset_cols, columns=factor_cols, dtype=float)
    resid_var = pd.Series(np.nan, index=asset_cols, name='resid_var', dtype=float)

    X = df[factor_cols]
    if add_intercept:
        X = sm.add_constant(X)

    for asset in asset_cols:
        y = df[asset]
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < min_obs:
            continue

        y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
        try:
            model = sm.OLS(y_clean, X_clean).fit()
            for f in factor_cols:
                betas_df.loc[asset, f] = model.params.get(f, np.nan)
                tvalue_df.loc[asset, f] = model.tvalues.get(f, np.nan)
            # 使用 mse_resid 作为残差方差估计
            resid_var.loc[asset] = float(getattr(model, 'mse_resid', np.nan))
        except Exception:
            continue

    return betas_df,tvalue_df, resid_var





def map_factor_to_asset_weights(
    V_k: np.ndarray,
    target_factor_exposure: np.ndarray,
    *,
    weight_cap: float | None = None,
    factor_cov_matrix: np.ndarray | None = None,
    allow_short: bool = False,
    enforce_budget_constraint: bool = True,
) -> np.ndarray:
    N, k = V_k.shape
    if N == 0:
        return np.array([])
    if k == 0:
        return np.ones(N) / max(N, 1)
    t = target_factor_exposure.astype(float)
    x0 = np.ones(N) / max(N, 1)
    cap = None
    if isinstance(weight_cap, (int, float)) and weight_cap > 0:
        cap = float(weight_cap)
        if cap * N < 1.0:
            cap = 1.0 / N + 1e-9
    
    # Pre-compute inverse or use directly? 
    # We want to minimize (V'w - t).T @ Sigma @ (V'w - t)
    # If Sigma is None, it's Identity.
    
    def obj(w: np.ndarray) -> float:
        e = V_k.T @ w - t
        if factor_cov_matrix is not None:
            # Minimize Tracking Error Variance
            base_error = e @ factor_cov_matrix @ e
        else:
            # Minimize Euclidean Distance
            base_error = e @ e
            
        return float(base_error)

    constraints = []
    if enforce_budget_constraint:
        constraints.append({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    
    # Determine bounds
    if allow_short:
        # Relax bounds significantly to allow leverage for factor matching
        # Default to [-10, 10] if no specific cap is provided
        lb = -10.0
        ub = cap if cap is not None else 10.0
    else:
        # Long-only constraints
        lb = 0.0
        ub = cap if cap is not None else 1.0
        
    bnds = tuple((lb, ub) for _ in range(N))
    
    try:
        res = minimize(obj, x0, method='SLSQP', bounds=bnds, constraints=tuple(constraints), tol=1e-10)
        if res.success and np.all(np.isfinite(res.x)):
            w = res.x
            # Clean up small weights
            if allow_short:
                w[np.abs(w) < 1e-12] = 0.0
            else:
                w[w < 1e-12] = 0.0
            
            if enforce_budget_constraint:
                s = w.sum()
                return w if abs(s) <= 1e-9 else (w / s)
            else:
                # If not enforcing budget, return raw weights (they match the factor exposure magnitude)
                return w
    except Exception as e:
        logger.info(f"_map_factor_to_asset_weights 优化异常: {e}")
    return np.ones(N) / N

# -----------------------------
# HRP clustering utilities
# -----------------------------

def _correlation_distance_matrix(cov_matrix: np.ndarray) -> np.ndarray:
    std_vector = np.sqrt(np.diag(cov_matrix))
    corr_matrix = cov_matrix / np.outer(std_vector, std_vector)
    corr_matrix = np.clip(corr_matrix, -1, 1)
    distance_matrix = np.sqrt(0.5 * (1 - corr_matrix))
    return distance_matrix


def _quasi_diagonal_order(linkage_matrix: np.ndarray, asset_count: int) -> List[int]:
    def recursive_get(idx):
        if idx < asset_count:
            return [idx]
        left = int(linkage_matrix[idx - asset_count, 0])
        right = int(linkage_matrix[idx - asset_count, 1])
        return recursive_get(left) + recursive_get(right)
    root_index = 2 * asset_count - 2
    ordered_list = recursive_get(root_index)
    return ordered_list


def _cluster_variance(cov_matrix: np.ndarray, cluster_indices: list, epsilon: float = EPSILON_NUMERIC) -> float:
    if len(cluster_indices) == 0:
        return 0.0
    sub_cov = cov_matrix[np.ix_(cluster_indices, cluster_indices)]
    if sub_cov.shape[0] == 1:
        return float(sub_cov[0, 0])
    diag = np.diag(sub_cov).astype(float)
    diag = np.clip(diag, epsilon, None)
    std_vec = np.sqrt(diag)
    inv_std_w = 1.0 / np.clip(std_vec, epsilon, None)
    inv_std_w = np.maximum(inv_std_w, 0)
    if inv_std_w.sum() <= 0:
        w = np.full(len(inv_std_w), 1.0 / len(inv_std_w))
    else:
        w = inv_std_w / inv_std_w.sum()
    variance = w @ sub_cov @ w
    return float(variance)


def hrp_allocate_recursive_bisect(
    cov_matrix: np.ndarray,
    asset_names: list,
    linkage_method: str = 'ward',
    use_optimal_ordering: bool = True,
    epsilon: float = EPSILON_NUMERIC,
    allocation_metric: str = 'std',
    diag_vol_floor: float | None = None,
) -> pd.Series:
    n = cov_matrix.shape[0]
    if diag_vol_floor is not None and diag_vol_floor > 0:
        cov_matrix = cov_matrix.copy()
        diag = np.diag(cov_matrix).astype(float)
        vol = np.sqrt(np.clip(diag, 0.0, None))
        mask = vol < diag_vol_floor
        if mask.any():
            vol_adj = vol.copy()
            vol_adj[mask] = diag_vol_floor
            new_diag = np.square(vol_adj)
            np.fill_diagonal(cov_matrix, new_diag)
    distance_matrix = _correlation_distance_matrix(cov_matrix)
    condensed = sch.distance.squareform(distance_matrix, checks=False)
    Z = sch.linkage(condensed, method=linkage_method)
    if use_optimal_ordering:
        try:
            Z = sch.optimal_leaf_ordering(Z, condensed)
        except Exception:
            pass
    A = sch.leaves_list(Z).tolist()
    B = _quasi_diagonal_order(Z, n)
    ordered_indices = A if A == B else B
    weights = pd.Series(1.0, index=ordered_indices, dtype=float)
    def recursive_bisect(indices):
        if len(indices) <= 1:
            return
        split = len(indices) // 2
        left = indices[:split]
        right = indices[split:]
        var_left = _cluster_variance(cov_matrix, left, epsilon=epsilon)
        var_right = _cluster_variance(cov_matrix, right, epsilon=epsilon)
        scale_left = float(np.sqrt(max(var_left, 0.0)))
        scale_right = float(np.sqrt(max(var_right, 0.0)))
        denom = scale_left + scale_right
        if denom <= epsilon:
            alloc_left = alloc_right = 0.5
        else:
            alloc_left = 1 - scale_left / denom
            alloc_right = 1 - scale_right / denom
        weights[left] *= alloc_left
        weights[right] *= alloc_right
        recursive_bisect(left)
        recursive_bisect(right)
    recursive_bisect(ordered_indices)
    weights = weights / weights.sum()
    final_series = pd.Series(0.0, index=range(n))
    final_series.loc[weights.index] = weights.values
    final_series.index = asset_names
    return final_series

# -----------------------------
# Factor Rotation (Varimax)
# -----------------------------

def varimax(Phi: np.ndarray, gamma: float = 1.0, q: int = 20, tol: float = 1e-6) -> np.ndarray:
    """
    Perform Varimax (orthogonal) rotation on the loadings matrix Phi.
    
    Args:
        Phi: The loadings matrix (n_features, n_components).
        gamma: The coefficient for the rotation (1.0 for Varimax).
        q: Maximum number of iterations.
        tol: Tolerance for convergence.
        
    Returns:
        R: The rotation matrix (n_components, n_components).
    """
    p, k = Phi.shape
    R = np.eye(k)
    d = 0
    for i in range(q):
        d_old = d
        Lambda = np.dot(Phi, R)
        u, s, vh = np.linalg.svd(np.dot(Phi.T, np.asarray(Lambda)**3 - (gamma/p) * np.dot(Lambda, np.diag(np.diag(np.dot(Lambda.T, Lambda))))))
        R = np.dot(u, vh)
        d = np.sum(s)
        if d_old != 0 and d/d_old < 1 + tol: break
    return R

__all__ = [
    'to_daily_rebalance_dates',
    'choose_num_factors',
    'get_risk_contributions_daily',
    'solve_risk_parity_weights',
    'estimate_factor_betas_ols',
    'solve_factor_risk_parity_weights',
    'map_factor_to_asset_weights',
    'hrp_allocate_recursive_bisect',
    'varimax',
]
