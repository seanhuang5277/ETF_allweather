# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_MultiLayer_PCA_HRP.py
@Author : Copilot
@Date   : 2025-11-19

@Description:
多层资产配置：
1) 读取配置中标记为 Y 的全部指数，按资产类别分组为 Stock/Bond/Commodity/Gold/Cash。
2) 类内分配：
   - Stock: PCA 因子风险平价（PCA-RP）
   - Bond, Commodity: HRP（分层风险平价，递归二分、波动倒数分配）
   - Gold, Cash: 单资产直接持有（如有多于1个则回退 HRP）
3) 将五大类组合成“类别日度收益”宽表后，顶层再做一次 PCA-RP 或 RP（参数可选），得到最终组合。

注意：为避免重复计成本，类内组合在合成“类别日度收益”时不计交易成本；仅在顶层组合处计一次成本。
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Sequence, Dict
from sklearn.covariance import OAS
from scipy.optimize import minimize
import matplotlib.pyplot as plt

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
from framework.performance import compute_portfolio_returns_and_equity, calculate_performance_metrics
from framework.allocation_utils import (
    to_daily_rebalance_dates,
    choose_num_factors,
    get_risk_contributions_daily,
    solve_risk_parity_weights,
    map_factor_to_asset_weights,
    hrp_allocate_recursive_bisect,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import plot_equity_curve, plot_weights_history

# HRP 分配已迁移到 framework.allocation_utils


# 初始化日志（默认 INFO；如需细粒度调试，可改为 DEBUG）
setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_factor_RP.log")
logger = get_logger("strategy.MultiLayer")

STRATEGY_MODE = 'PCA_multi_Factor_RiskParity'

# =========================
# 公共小工具
# =========================

# 迁移后: 本文件不再定义通用函数，统一由 allocation_utils 提供


# =========================
# 顶层入口
# =========================

def run_strategy(
    *,
    data_dir: str = 'data',
    # 回测时间段（包含端点）；None 表示不裁剪整段数据
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    # 类内通用参数
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    # PCA 因子选择参数
    n_factors: Optional[int] = 3,
    explained_variance_threshold: float = 0.7,
    max_factors: int = 8,
    # 分散控制
    asset_weight_cap: float | None = None,
    asset_ridge_alpha: float = 1e-3,
    # HRP 可选对角线波动地板
    diag_vol_floor: float | None = None,
    # 顶层方法: 'PCA-RP' 或 'RP'
    top_level_method: str = 'PCA-RP',
    # 仅用于顶层 PCA-RP：是否用相关矩阵做因子RP并回到原尺度
    use_correlation: bool = True,
    # 成本与绩效参数
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    auto_plot: bool = False,
) -> Dict[str, object]:
    """多层配置：类内 -> 五大类 -> 顶层。

    返回:
      - category_daily_returns_df: 五大类日度简单收益
      - category_weights_history_df: 顶层月度权重（五大类）
      - inside_weights_hist_by_cat: 各类别内部的月度权重
      - final_portfolio_returns_series, equity_curve_series, performance_report, total_cost
    """
    lookback_window = TRADING_DAYS_PER_YEAR_SCALAR * lookback_years
    min_data_days_required = TRADING_DAYS_PER_YEAR_SCALAR * min_data_years

    # 统一解析日期参数（允许 str / Timestamp / None）
    def _parse_dt(x):
        if x is None:
            return None
        return pd.to_datetime(x)
    start_dt_global = _parse_dt(start_date)
    end_dt_global = _parse_dt(end_date)


    # 1) 加载 ETF/Index（只取配置中 Y）并按资产类别分组
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir=data_dir,returns_index_col='日期',)
    
    # 目标五大类名称及类内方法（大小写与配置保持一致）
    cat_dict = {
        'Stock': 'HRP',
        'Bond': 'HRP',
        'Commodity': 'HRP',
        'Gold': 'SINGLE',
    }
    cat_names = list(cat_dict.keys())
    
    ## 先将所有数据提出来形成大表
    index_cat_dfs: list[pd.DataFrame] = []
    etf_cat_dfs: list[pd.DataFrame] = []
    for cat_name in cat_dict.keys():
        index_simple_df = idx_by_cat.get(cat_name)  # 指数简单收益
        etf_simple_df = etf_by_cat.get(cat_name)    # ETF 简单收益
        index_cat_dfs.append(index_simple_df)
        etf_cat_dfs.append(etf_simple_df)
    # 组合所有底层资产的指数与ETF简单收益宽表（按列前缀区分），便于外部进一步分析
    all_index_simple_returns_df = pd.concat(index_cat_dfs, axis=1).sort_index() if index_cat_dfs else pd.DataFrame()
    all_etf_simple_returns_df = pd.concat(etf_cat_dfs, axis=1).sort_index() if etf_cat_dfs else pd.DataFrame()


    # 一一对应
    if  len(all_index_simple_returns_df.columns) != len(all_etf_simple_returns_df.columns):
        raise ValueError(f"index和etf数量不一致: index={len(all_index_simple_returns_df.columns)}, etf={len(all_etf_simple_returns_df.columns)}")    
    # 确定月末再平衡日期（用于常规调仓）
    rebalance_dates = to_daily_rebalance_dates(all_index_simple_returns_df, min_data_days_required)


    # 计算每个ETF的存活时间
    mapping = {x : y for x,y in zip(all_index_simple_returns_df,all_etf_simple_returns_df)}
    reversed_mapping = {y : x for x,y in mapping.items()}

    etf_inception: dict[str, pd.Timestamp] = {}
    for idx_name, etf_name in mapping.items():
        first_idx = all_etf_simple_returns_df[etf_name].first_valid_index()
        if first_idx is not None:
            etf_inception[etf_name] = first_idx
    if etf_inception:
        logger.info("全部ETF上市日:\n" + ",\n ".join([f"{k}:{etf_inception[k].date()}" for k in etf_inception]))    
    


    # 2) 类内分配：生成两套类别日度收益
    #   a) index路径: 只用指数数据做协方差与顶层权重；对未上市ETF对应指数在其ETF上市前置 NaN（不可投资）
    #   b) etf路径: 用实际ETF日度收益做绩效，只在上市后参与组合；上市前权重为0，收益列可保持 NaN
    inside_w_hist: Dict[str, pd.DataFrame] = {}
    cat_index_returns: Dict[str, pd.Series] = {}
    cat_etf_returns: Dict[str, pd.Series] = {}
    # 收集所有类别原始指数与ETF简单收益，用于最终拼接成两个总宽表



    # 分类处理资产
    for cat_name, cat_method in cat_dict.items():
        cat_index_simple_df = idx_by_cat.get(cat_name)  # 指数简单收益
        cat_etf_simple_df = etf_by_cat.get(cat_name)    # ETF 简单收益
        # 提取列名
        cat_idx_cols = list(cat_index_simple_df.columns)
        cat_etf_cols = list(cat_etf_simple_df.columns)

        # 获得指数的对数收益
        cat_index_log_df = compute_log_returns(cat_index_simple_df)

        if cat_method == 'SINGLE': # 单资产直接保留
            if len(cat_etf_cols) != 1 :
                raise ValueError(f"类别 {cat_name} SINGLE 模式但资产列数不为1: index={len(cat_idx_cols)}, etf={len(cat_etf_cols)}")
            inception = etf_inception.get(cat_etf_cols[0])
            # 截取上市后的再平衡日期
            single_rebals = rebalance_dates[rebalance_dates >= inception]
            w_hist = pd.DataFrame(1.0, index=single_rebals, columns=cat_etf_cols)
            inside_w_hist[cat_name] = w_hist
            index_series = cat_index_simple_df[cat_idx_cols[0]].copy()
            cat_index_returns[cat_name] = index_series
            cat_etf_returns[cat_name] = cat_etf_simple_df[cat_etf_cols[0]]
            logger.info(f"类别 {cat_name}: 单资产 ETF列={cat_etf_cols} Index列={cat_idx_cols} 上市日={inception.date()} ")
            continue
        elif cat_method == 'HRP':
            w_hist = pd.DataFrame(index=rebalance_dates, columns=cat_etf_cols, dtype=float)
            rc_hist = pd.DataFrame(index=rebalance_dates, columns=cat_etf_cols, dtype=float)
            oas = OAS(assume_centered=True)            
            for dt in rebalance_dates:
                start_dt = dt - pd.DateOffset(days=lookback_window)
                window_df = cat_index_log_df.loc[start_dt:dt]                
                # 找到这个阶段可以选取的资产etf
                cat_live_index = []                
                for idx_name in cat_idx_cols:
                    inception = etf_inception.get(mapping.get(idx_name))
                    if inception is None or inception > dt:
                        continue
                    if window_df[idx_name].count() >= min_data_days_required:
                        cat_live_index.append(idx_name)

                cat_live_etf = [mapping[idx_name] for idx_name in cat_live_index]
    
                n_live = len(cat_live_etf)
                logger.info(f"{dt.date()} 窗口[{start_dt.date()}~{dt.date()}] 成活资产数={n_live}")
                if n_live < 2:
                    w_hist.loc[dt] = 0.0
                    if n_live == 1:
                        w_hist.loc[dt, cat_live_etf] = 1.0
                    continue
                clean = window_df[cat_live_index].dropna()
                if len(clean) < min_clean_days:
                    w_hist.loc[dt] = 0.0
                    logger.info(f"{dt.date()} 有效样本天数不足: {len(clean)} < {min_clean_days}")
                    continue

                # index_cov_matrix = oas.covariance_
                index_cov_matrix = clean.cov().values
                w_series = hrp_allocate_recursive_bisect(
                    index_cov_matrix,
                    cat_live_index,
                    linkage_method='ward',
                    use_optimal_ordering=True,
                    epsilon=EPSILON_NUMERIC,
                    debug=False,
                    allocation_metric='std',
                    diag_vol_floor=diag_vol_floor,
                )
                # 类内映射（按位置一一对应）

                w_vec = w_series.values
                mrc = index_cov_matrix @ w_vec
                trc = w_vec * mrc
                port_var = float(w_vec @ index_cov_matrix @ w_vec)
                port_std_d = float(np.sqrt(max(port_var, 0.0)))
                rc_std_d = trc / port_std_d if port_std_d > 0 else np.zeros_like(trc)
                rc_std_y = rc_std_d * np.sqrt(TRADING_DAYS_PER_YEAR_SCALAR)

                w_hist.loc[dt] = 0.0
                w_hist.loc[dt, cat_live_etf] = w_vec
                # 资产内部 HRP 分配权重打印
                rc_hist.loc[dt] = 0.0
                rc_hist.loc[dt, cat_live_etf] = rc_std_y

                w_order_idx = np.argsort(w_vec)[::-1]
                rc_order_idx = np.argsort(rc_std_y)[::-1]
                w_top_idx = w_order_idx[: min(5, len(w_order_idx))]
                rc_top_idx = rc_order_idx[: min(5, len(rc_order_idx))]
                w_top_names = [cat_live_etf[i] for i in w_top_idx]
                rc_top_names = [cat_live_etf[i] for i in rc_top_idx]
                w_top_vals = [w_vec[i] for i in w_top_idx]
                rc_top_vals = [rc_std_y[i] for i in rc_top_idx]
                w_top_str = ", ".join([f"{n}:{v:.2%}" for n, v in zip(w_top_names, w_top_vals)])
                rc_top_str = ", ".join([f"{n}:{v:.2%}" for n, v in zip(rc_top_names, rc_top_vals)])
                logger.info(f"{dt.date()} {cat_name} HRP资产权重: min={w_vec.min():.4%}, median={np.median(w_vec):.4%}, max={w_vec.max():.4%}; TOP: {w_top_str}\n"
                                                    + f"HRP风险权重: min={rc_std_y.min():.4%}, median={np.median(rc_std_y):.4%}, max={rc_std_y.max():.4%}; TOP: {rc_top_str}")  

            w_hist = w_hist.fillna(0.0)
            inside_w_hist[cat_name] = w_hist
            # a) 指数路径：重命名为ETF列名并在ETF上市前置 NaN
            index_df_adj = cat_index_simple_df.copy()
            index_df_adj.columns = cat_etf_cols  # 与权重列名对齐（ETF名）
            for etf_name in cat_etf_cols:
                inc = etf_inception.get(etf_name)
                if inc is not None:
                    index_df_adj.loc[index_df_adj.index < inc, etf_name] = np.nan
            idx_port_returns_frcitionless,_, _, _ = compute_portfolio_returns_and_equity(
                weights_history_df=w_hist,
                daily_returns_df=index_df_adj.loc[w_hist.index[0]:],
                cost_per_side_scalar=0.0,
            )
            # b) ETF 路径：直接使用ETF简单收益
            etf_df_adj = cat_etf_simple_df.copy()
            etf_port_returns_frictionless, _, _, _ = compute_portfolio_returns_and_equity(
                weights_history_df=w_hist,
                daily_returns_df=etf_df_adj.loc[w_hist.index[0]:],
                cost_per_side_scalar=0.0,
            )
            cat_index_returns[cat_name] = idx_port_returns_frcitionless
            cat_etf_returns[cat_name] = etf_port_returns_frictionless
            logger.info(f"类别 {cat_name}: 方法={cat_method} 完成，窗口数={len(w_hist)}")

    # 3) 合成五大类日度收益
    if not cat_index_returns:
        raise ValueError("未生成任何类别指数路径收益，检查数据。")
    # 从字典分类里提出来成为总收益率 DataFrame
    cat_daily_df_index = pd.concat(cat_index_returns, axis=1)
    exist_cols_idx = [c for c in cat_names if c in cat_daily_df_index.columns]
    cat_daily_df_index = cat_daily_df_index[exist_cols_idx].sort_index()

    if not cat_etf_returns:
        raise ValueError("未生成任何类别ETF路径收益，检查数据。")
    cat_daily_df_etf = pd.concat(cat_etf_returns, axis=1)
    exist_cols_etf = [c for c in cat_names if c in cat_daily_df_etf.columns]
    cat_daily_df_etf = cat_daily_df_etf[exist_cols_etf].sort_index()

    # 4) 顶层分配（PCA-RP 或 RP）
    # 4) 顶层分配使用 index 路径（不可投资时列为 NaN 自动排除）
    log_top = compute_log_returns(cat_daily_df_index)
    # 不裁剪 log_top，协方差窗口需要完整历史；权重表索引已限制在 rebalance_dates 内。
    if log_top.empty or len(log_top.columns) < 2:
        raise ValueError("顶层类别不足 2 个，无法进行组合。")

    # 顶层月度调仓

    top_w_hist = pd.DataFrame(index=rebalance_dates, columns=log_top.columns.tolist(), dtype=float)
    oas = OAS(assume_centered=True)

    method_upper = (top_level_method or 'PCA-RP').upper()

    for dt in rebalance_dates:
        start_dt = dt - pd.DateOffset(days=lookback_window)
        window_df = log_top.loc[start_dt:dt]
        live = window_df.count()[lambda s: s >= min_data_days_required].index.tolist()
        n = len(live)
        if n < 4:
            top_w_hist.loc[dt] = 0.0
            continue
        clean = window_df[live].dropna()
        if len(clean) < min_clean_days:
            top_w_hist.loc[dt] = 0.0
            continue

        # 选择协方差矩阵使用方法，这个未来变成参数
        # top_cov_matrix = clean.cov().values
        # top_cov_matrix = clean.ewm(span=252).cov(pairwise=True).iloc[-1]

                    # 方法 B (更推荐)：按最后一个日期索引取值
        last_date = clean.index[-1]
        top_cov_matrix = clean.ewm(span=252).cov(pairwise=True).loc[last_date]
        # 说明：纯RP顶层始终使用协方差矩阵；use_correlation 仅作用于 PCA-RP 因子风险平价


        if method_upper == 'RP':
            # 顶层RP：始终基于协方差矩阵做资产风险平价
            w = solve_risk_parity_weights(top_cov_matrix, None)
            try:
                mrc_top = top_cov_matrix @ w  # 用原协方差计算真实风险贡献
                rc_top = w * mrc_top
                rc_pct_top = rc_top / rc_top.sum() if rc_top.sum() > 0 else rc_top
                logger.info(
                    f"顶层RP(Cov): 权重范围[{w.min():.4%}, {w.max():.4%}] RC均衡性(min={rc_pct_top.min():.3f}, max={rc_pct_top.max():.3f}, std={rc_pct_top.std():.3f})"
                )
            except Exception:
                pass
            top_w_hist.loc[dt] = 0.0
            top_w_hist.loc[dt, live] = w
        else:  # PCA-RP 顶层
            # PCA-RP：可选改用相关矩阵进行PCA与因子风险平价
            vol_vec = np.sqrt(np.clip(np.diag(top_cov_matrix), 0.0, None))
            vol_floor = 1e-12
            safe_vol = np.where(vol_vec > vol_floor, vol_vec, vol_floor)            
            D_inv = np.diag(1.0 / safe_vol)
            corr_matrix = D_inv @ top_cov_matrix @ D_inv
            evals, evecs = np.linalg.eigh(corr_matrix)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            k = choose_num_factors(evals, n_factors=n_factors, explained_var_threshold=explained_variance_threshold, max_factors=min(max_factors, n))
            evals_k = np.clip(evals[:k], 0.0, None)
            V_k = evecs[:, :k]
            sigma_f = np.diag(evals_k)
            w_f = solve_risk_parity_weights(sigma_f, None)
            # 顶层 PCA 调试：因子数与解释率
            try:
                total_var_top = float(np.clip(evals, 0.0, None).sum())
                exp_ratio_top = float(np.clip(evals[:k], 0.0, None).sum() / total_var_top) if (k > 0 and total_var_top > 0) else 0.0
                mrc_factor_top = sigma_f @ w_f
                rc_factor_top = w_f * mrc_factor_top
                rc_f_pct_top = rc_factor_top / rc_factor_top.sum() if rc_factor_top.sum() > 0 else rc_factor_top
                logger.info(
                    f"顶层PCA({'Corr' if use_correlation else 'Cov'}): k={k} 解释率={exp_ratio_top:.2%}; 因子RP权重范围[{w_f.min():.4%}, {w_f.max():.4%}] RC均衡性(min={rc_f_pct_top.min():.3f}, max={rc_f_pct_top.max():.3f}, std={rc_f_pct_top.std():.3f})"
                )
            except Exception:
                pass
            w_asset_top = map_factor_to_asset_weights(V_k, w_f, weight_cap=None, ridge_alpha=asset_ridge_alpha)

            # 从相关空间回到原尺度：按 1/vol 缩放后再归一
            w_asset_top = w_asset_top / np.where(vol_vec > 0, vol_vec, 1e-8)
            w_asset_top = np.maximum(w_asset_top, 0)
            s_top = w_asset_top.sum()
            if s_top > 0:
                w_asset_top = w_asset_top / s_top
            top_w_hist.loc[dt] = 0.0
            top_w_hist.loc[dt, live] = w_asset_top
            # 顶层资产（五大类）权重摘要
            try:
                order_idx = np.argsort(w_asset_top)[::-1]
                top_idx = order_idx[: min(5, len(order_idx))]
                top_names = [live[i] for i in top_idx]
                top_vals = [w_asset_top[i] for i in top_idx]
                top_str = ", ".join([f"{n}:{v:.2%}" for n, v in zip(top_names, top_vals)])
                logger.info(
                    f"{dt.date()} 顶层类别权重: min={w_asset_top.min():.4%}, median={np.median(w_asset_top):.4%}, max={w_asset_top.max():.4%}; TOP: {top_str}"
                    )
                 # 1. 算出资产组合实际实现的因子暴露
                realized_factor_exposure = V_k.T @ w_asset_top 

                # 2. 算出实际的因子风险贡献 (假设因子间独立)
                # RC_i = Exposure_i^2 * Variance_Factor_i
                realized_factor_rc = (realized_factor_exposure ** 2) * evals_k
                realized_factor_rc_pct = realized_factor_rc / realized_factor_rc.sum()

                # 3. 打印这个 realized_factor_rc_pct
                logger.info(f"实际实现的因子RC分布: {realized_factor_rc_pct}")
                
                
            except Exception:
                pass

    top_w_hist = top_w_hist.fillna(0.0)

    # 若指定 start_date：保证 start_date 作为首个再平衡点（即初始建仓），允许它不是月末。
    if start_dt_global is not None:
        if start_dt_global < all_index_simple_returns_df.index.min() or start_dt_global > all_index_simple_returns_df.index.max():
            raise ValueError("start_date 不在数据索引范围内。")
        # 保留从 start_date 之后的原月末，再把 start_date 插入最前（若不在原集合里）
        top_w_hist = top_w_hist[top_w_hist.index >= start_dt_global]

    # end_date 仅截断输出权重区间；不影响历史窗口的回看
    if end_dt_global is not None:
        top_w_hist = top_w_hist[top_w_hist.index <= end_dt_global]




    # 5) 顶层绩效使用 ETF 路径（真实可交易收益）
    port_ret_frictionless,port_ret, equity, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=top_w_hist,
        daily_returns_df=cat_daily_df_etf.loc[top_w_hist.index[0]:],
        cost_per_side_scalar=cost_per_side,
    )
    equity.name = f"MultiLayer({top_level_method})"
    report = calculate_performance_metrics(port_ret, equity, rf_rate, TRADING_DAYS_PER_YEAR_SCALAR)

    total_cost_series = (top_w_hist.diff().abs().sum(axis=1).astype('float64') * cost_per_side).shift(1)
    total_cost_series = total_cost_series.reindex(port_ret.index).astype('float64').fillna(0.0)
    total_cost = float(total_cost_series.sum())
    logger.info(f"总交易成本(单位权重*单边成本累加)={total_cost:.6f}")

    figures = {}
    if auto_plot:
        fig1 = plot_equity_curve(equity, report, STRATEGY_MODE, auto_show=False)
        fig2 = plot_weights_history(top_w_hist, STRATEGY_MODE, reverse_legend=True, auto_show=False)
        plt.show()
        figures = {'equity': fig1, 'weights': fig2}
    return {
        'category_daily_returns_index_df': cat_daily_df_index,
        'category_daily_returns_etf_df': cat_daily_df_etf,
        'category_weights_history_df': top_w_hist,
        'inside_weights_hist_by_cat': inside_w_hist,
        'final_portfolio_returns_series': port_ret,
        'equity_curve_series': equity,
        'performance_report': report,
        'total_cost': total_cost,
        'figures': figures,
        'all_index_assets_daily_returns_df': all_index_simple_returns_df,
        'all_etf_assets_daily_returns_df': all_etf_simple_returns_df,
        'backtest_start': start_dt_global,
        'backtest_end': end_dt_global,
    }


if __name__ == "__main__":
    res = run_strategy(
        data_dir='data',
        top_level_method='PCA-RP',
        asset_weight_cap=None,
        asset_ridge_alpha=1e-1,
        diag_vol_floor=None,
        auto_plot=True,
        start_date="20200101",
        end_date="20251031",
    )
    perf = res['performance_report']
    logger.info("多层配置完成：主要指标")
    for k, v in perf.items():
        try:
            if any(x in k for x in ['CAGR', 'Vol', '回撤']):
                logger.info(f"  {k}: {v:.2%}")
            else:
                logger.info(f"  {k}: {v:.4f}")
        except Exception:
            logger.info(f"  {k}: {v}")
