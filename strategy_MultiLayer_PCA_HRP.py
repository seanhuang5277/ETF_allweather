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
from framework.performance import (
    compute_portfolio_returns_and_equity, 
    calculate_performance_metrics,
    calculate_return_attribution
)
from framework.allocation_utils import (
    to_daily_rebalance_dates,
    choose_num_factors,
    get_risk_contributions_daily,
    solve_risk_parity_weights,
    map_factor_to_asset_weights,
    hrp_allocate_recursive_bisect,
    varimax,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import plot_equity_curve, plot_weights_history, plot_return_attribution

# 初始化日志（默认 INFO；如需细粒度调试，可改为 DEBUG）
setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_factor_RP.log")
logger = get_logger("strategy.MultiLayer")

STRATEGY_MODE = 'PCA_multi_Factor_RiskParity'

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
    top_cov_estimate_ways: str = 'ewm',
    ewm_span_days: int = 252,
    # PCA 因子选择参数
    n_factors: Optional[int] = 3,
    explained_variance_threshold: float = 0.8,
    max_factors: int = 8,
    # 分散控制
    asset_weight_cap: float | None = None,
    asset_ridge_alpha: float = 1e-2,
    # HRP 可选对角线波动地板
    # 顶层方法: 'PCA-RP' 或 'RP'
    top_level_method: str = 'PCA-RP',
    # 成本与绩效参数
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    auto_plot: bool = False,
    # 用ETF真实测，还是回测ETF上市前的指数
    use_etf_real_data: bool = True,
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
    etf_by_cat_dict_df, idx_by_cat_dict_df = load_etf_index_returns_by_category(data_dir=data_dir, returns_index_col='日期')

    # 目标五大类名称及类内方法（大小写与配置保持一致）
    cat_dict = {
        'Stock': 'HRP',
        'Bond': 'HRP',
        'Commodity': 'HRP',
        'Gold': 'SINGLE',
    }
    cat_names = list(cat_dict.keys())
    
    ## 先将所有数据提出来形成大表
    all_index_list = []
    all_etf_list = []
    for cat in cat_dict:
        all_index_list.append(idx_by_cat_dict_df[cat])
        all_etf_list.append(etf_by_cat_dict_df[cat])
    all_idx_simp_df = pd.concat(all_index_list, axis=1).sort_index()
    all_etf_simp_df = pd.concat(all_etf_list, axis=1).sort_index()


    # 一一对应
    if len(all_idx_simp_df.columns) != len(all_etf_simp_df.columns):
        raise ValueError(f"index和etf数量不一致: index={len(all_idx_simp_df.columns)}, etf={len(all_etf_simp_df.columns)}")
    # 确定月末再平衡日期（用于常规调仓）
    rebalance_dates = to_daily_rebalance_dates(all_idx_simp_df, min_data_days_required)

    # 插入 start_date 作为首个再平衡点（初始建仓），确保不需要等到下一个月末
    if start_dt_global is not None:
        if start_dt_global < all_idx_simp_df.index.min() or start_dt_global > all_idx_simp_df.index.max():
            raise ValueError("start_date 不在数据索引范围内。")
        future_rebals = rebalance_dates[rebalance_dates >= start_dt_global]
        if start_dt_global not in future_rebals:
            rebalance_dates = pd.DatetimeIndex([start_dt_global] + list(future_rebals))
        else:
            rebalance_dates = future_rebals
    if end_dt_global is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end_dt_global]
    if len(rebalance_dates) == 0:
        raise ValueError("所选时间段内无有效再平衡日期（检查 start_date/end_date）。")

    # 创建index和ETF的
    mapping = {x : y for x,y in zip(all_idx_simp_df.columns, all_etf_simp_df.columns)}
    reversed_mapping = {y : x for x,y in mapping.items()}
    
    # 计算每个ETF的存活时间
    etf_inception: dict[str, pd.Timestamp] = {}
    for idx_name, etf_name in mapping.items():
        if use_etf_real_data:
            first_inc = all_etf_simp_df[etf_name].first_valid_index()
        else:
            first_inc = all_idx_simp_df[idx_name].first_valid_index()
        if first_inc is not None:
            etf_inception[etf_name] = first_inc
    if etf_inception:
        logger.info("全部ETF上市日:\n" + ",\n ".join([f"{k}:{etf_inception[k].date()}" for k in etf_inception]))

    # 预计算各类别指数的对数收益（用于协方差计算）
    cat_index_log_dict_df = {c: compute_log_returns(idx_by_cat_dict_df[c]) for c in cat_names}

    # 初始化权重存储
    # inside_w_hist: {cat_name: DataFrame(index=rebalance_dates, columns=etf_cols)}
    inside_w_hist_dict: Dict[str, pd.DataFrame] = {
        c: pd.DataFrame(index=rebalance_dates, columns=idx_by_cat_dict_df[c].columns, dtype=float).fillna(0.0)
        for c in cat_names
    }
    # top_w_hist: DataFrame(index=rebalance_dates, columns=cat_names)
    top_w_hist_df = pd.DataFrame(index=rebalance_dates, columns=cat_names, dtype=float).fillna(0.0)
    final_atomic_w_hist_df = pd.DataFrame(index=rebalance_dates, columns=all_idx_simp_df.columns, dtype=float).fillna(0.0)

    method_upper = (top_level_method or 'PCA-RP').upper()

    # ==========================================================================
    # 核心循环：单一时间轴，先算当期子类权重，再合成当期大类历史，再算顶层权重
    # ==========================================================================
    for dt in rebalance_dates:
        start_dt = dt - pd.DateOffset(days=lookback_window)
        
        # 收集本期各类别“合成”收益序列（用于顶层分配）
        current_window_cat_simp_list = []
        valid_cat_names_list = []
        # --- 1. 子类资产分配 ---
        for cat_name in cat_names:
            cat_method = cat_dict[cat_name]
            cat_idx_simp_df = idx_by_cat_dict_df[cat_name]
            # cat_etf_simp_df = etf_by_cat_dict_df[cat_name]
            cat_idx_log_df = cat_index_log_dict_df[cat_name]
            
            cat_idx_cols = list(cat_idx_simp_df.columns)
            # cat_etf_cols = list(cat_etf_simp_df.columns)
            
            # 确定本期存活资产 (Index Path)
            window_idx_log_df = cat_idx_log_df.loc[start_dt:dt]
            window_idx_simp_df = cat_idx_simp_df.loc[start_dt:dt]
            live_idx_cols = []
            for idx_col in cat_idx_cols:
                inc = etf_inception.get(mapping[idx_col])
                # 必须已上市且在窗口内有足够数据
                if inc is None or inc > dt:
                    continue
                if window_idx_log_df[idx_col].count() >= min_data_days_required:
                    live_idx_cols.append(idx_col)
            
            live_etf_cols = [mapping[c] for c in live_idx_cols]
            n_live = len(live_idx_cols)
            logger.info(f"Date {dt}: Live ETFs in category {cat_name}: {live_etf_cols}")
            # 计算当前类内权重
            if n_live == 0:
                continue
            if n_live == 1: # 分为HRP 和Single两种情况
                inside_w_hist_dict[cat_name].loc[dt, live_idx_cols] = 1.0
            ## HRP不用循环控制了，通过排除法下面是HRP的部分
            elif cat_method == 'HRP':
            # 如果长度不够就等分，这里应该也进不来
                clean_idx_log_df = window_idx_log_df[live_idx_cols].dropna()
                if len(clean_idx_log_df) < min_clean_days:
                    inside_w_hist_dict[cat_name].loc[dt, live_idx_cols] = 1.0 / n_live
                    continue
                cov_mat = clean_idx_log_df.cov().values
                w_series = hrp_allocate_recursive_bisect(
                    cov_mat, live_idx_cols, 
                    linkage_method='ward',
                    use_optimal_ordering=True,
                    epsilon=EPSILON_NUMERIC,
                    allocation_metric='std',
                )
                # 根据权重结果填回inside_w_hist_dict
                inside_w_hist_dict[cat_name].loc[dt, live_idx_cols] = w_series

            w_live = inside_w_hist_dict[cat_name].loc[dt, live_idx_cols]
            
            # Log Sub-asset weights
            if not w_live.empty:
                w_live_str = ", ".join([f"{k}:{v:.2%}" for k, v in w_live.items() if v > 0.0001])
                logger.info(f"[{dt.date()}] {cat_name} (Inside): {w_live_str}")

            live_win_idx_simp_df = window_idx_simp_df[live_idx_cols].fillna(0.0)
            # --- 构造本期合成收益 (Synthetic Returns) 用于顶层 ---
            # 使用 Index Simple Returns
            # R_cat = sum(w_i * R_i)
            composite_cat_simp_series = live_win_idx_simp_df.mul(w_live, axis=1).sum(axis=1)
            # 检查合成序列长度
            if composite_cat_simp_series.count() >= min_data_days_required:
                current_window_cat_simp_list.append(composite_cat_simp_series.rename(cat_name))
                valid_cat_names_list.append(cat_name)
            else:
                logger.info("不可能")

        # --- 2. 顶层资产分配 ---
        if len(current_window_cat_simp_list) >= 2:
            window_top_simp_df = pd.concat(current_window_cat_simp_list, axis=1).dropna()
            if len(window_top_simp_df) >= min_clean_days:
                # 转对数收益用于协方差
                window_top_log_df = compute_log_returns(window_top_simp_df)
                
                # 估计协方差 (优先 EWMA)
                if top_cov_estimate_ways == 'ewm':
                    last_date = window_top_log_df.index[-1]
                    cov_top = window_top_log_df.ewm(span=ewm_span_days).cov(pairwise=True).loc[last_date]
                else:
                    cov_top = window_top_log_df.cov()
                
                cov_top_vals = cov_top.values if isinstance(cov_top, pd.DataFrame) else cov_top
                
                # 计算顶层权重
                w_top_vec = np.zeros(len(valid_cat_names_list))
                
                if method_upper == 'RP':
                    w_top_vec = solve_risk_parity_weights(cov_top_vals, None)
                else: # PCA-RP
                    vol_vec = np.sqrt(np.clip(np.diag(cov_top_vals), 0.0, None))
                    vol_floor = 1e-12
                    safe_vol = np.where(vol_vec > vol_floor, vol_vec, vol_floor)
                    D_inv = np.diag(1.0 / safe_vol)
                    corr_matrix = D_inv @ cov_top_vals @ D_inv
                    # 计算特征值和特征向量   
                    evals, evecs = np.linalg.eigh(corr_matrix)
                    order = np.argsort(evals)[::-1]
                    evals = evals[order]
                    evecs = evecs[:, order]
                    # 这里有些多余，可以直接设定k值了 
                    k = choose_num_factors(evals, n_factors=n_factors, explained_var_threshold=explained_variance_threshold, max_factors=min(max_factors, len(valid_cat_names_list)))
                    evals_k = np.clip(evals[:k], 0.0, None)
                    V_k = evecs[:, :k]
                    

                    sigma_f = np.diag(evals_k)
                    w_f = solve_risk_parity_weights(sigma_f, None)
                    V_target = V_k

                    total_var_top = float(np.clip(evals, 0.0, None).sum())
                    exp_ratio_top = float(np.clip(evals[:k], 0.0, None).sum() / total_var_top) if (k > 0 and total_var_top > 0) else 0.0
                    
                    # 仅在非旋转模式下打印标准 RC 信息 (旋转后因子间存在相关性，简单 RC 计算可能不准确)

                    mrc_factor_top = sigma_f @ w_f
                    rc_factor_top = w_f * mrc_factor_top
                    rc_f_pct_top = rc_factor_top / rc_factor_top.sum() if rc_factor_top.sum() > 0 else rc_factor_top
                    logger.info(
                        f"顶层PCA({'Corr'}): k={k} 解释率={exp_ratio_top:.2%}; 因子RP权重范围[{w_f.min():.4%}, {w_f.max():.4%}] RC均衡性(min={rc_f_pct_top.min():.3f}, max={rc_f_pct_top.max():.3f}, std={rc_f_pct_top.std():.3f})"
                    )


                    # 准备因子协方差矩阵用于映射优化 (最小化跟踪误差方差)
                    target_sigma_f = sigma_f

                    w_asset_top = map_factor_to_asset_weights(
                        V_target, 
                        w_f, 
                        weight_cap=None, 
                        ridge_alpha=asset_ridge_alpha,
                        factor_cov_matrix=sigma_f,
                        allow_short=False,
                        enforce_budget_constraint=True
                    )
                    
                    # 验证因子暴露拟合情况 (在标准化空间)
                    actual_exposure = V_target.T @ w_asset_top
                    exposure_diff = np.linalg.norm(actual_exposure - w_f)
                    
                    # 计算余弦相似度 (衡量方向一致性)
                    norm_target = np.linalg.norm(w_f)
                    norm_actual = np.linalg.norm(actual_exposure)
                    if norm_target > 1e-9 and norm_actual > 1e-9:
                        cosine_sim = np.dot(w_f, actual_exposure) / (norm_target * norm_actual)
                    else:
                        cosine_sim = 0.0
                        
                    logger.info(f"因子暴露拟合: Target={np.round(w_f, 3)} vs Actual={np.round(actual_exposure, 3)}")
                    logger.info(f"  -> Diff(L2)={exposure_diff:.4f}, CosineSim={cosine_sim:.4f}")
                    if cosine_sim < 0.9:
                        logger.warning("  -> 因子暴露拟合度较低，可能是由于多头约束(Long-Only)或归一化约束导致无法完美复制目标因子暴露。")

                    # [新增] 计算实际组合在因子层面的风险贡献 (Actual Factor RC)
                    # 这里的 actual_exposure 是实际组合在因子空间的投影权重
                    # 对应的协方差矩阵是 target_sigma_f
                    mrc_factor_actual = target_sigma_f @ actual_exposure
                    rc_factor_actual = actual_exposure * mrc_factor_actual
                    rc_f_pct_actual = rc_factor_actual / rc_factor_actual.sum() if rc_factor_actual.sum() > 0 else rc_factor_actual
                    
                    logger.info(f"  -> 实际因子RC分布: {np.round(rc_f_pct_actual, 3)} (Target: Equal)")

                    # 回到原尺度
                    w_asset_top = w_asset_top / np.where(vol_vec > 0, vol_vec, 1e-8)
                    # w_asset_top = np.maximum(w_asset_top, 0) # 允许做空，移除非负约束
                    s_top = w_asset_top.sum()
                    if abs(s_top) > 1e-9:
                        w_asset_top = w_asset_top / s_top
                    w_top_vec = w_asset_top
                # 存储顶层权重
                top_w_hist_df.loc[dt, valid_cat_names_list] = w_top_vec
                
                # Log Top-level weights
                w_top_str = ", ".join([f"{n}:{v:.2%}" for n, v in zip(valid_cat_names_list, w_top_vec)])
                logger.info(f"[{dt.date()}] Top-Level: {w_top_str}")

                # 计算穿透后的底层资产权重
                for cat_name in cat_names:
                    # 获取该类别包含的资产列名
                    cat_cols = inside_w_hist_dict[cat_name].columns
                    # 顶层权重(标量) * 类内权重(Series)
                    w_top = top_w_hist_df.loc[dt, cat_name]
                    w_inside = inside_w_hist_dict[cat_name].loc[dt]
                    
                    # 计算原子权重
                    w_atomic = w_top * w_inside
                    final_atomic_w_hist_df.loc[dt, cat_cols] = w_atomic
                    
                    # Log Atomic weights (only non-zero)
                    w_atomic_nonzero = w_atomic[w_atomic > 0.0001]
                    if not w_atomic_nonzero.empty:
                        w_atomic_str = ", ".join([f"{k}:{v:.2%}" for k, v in w_atomic_nonzero.items()])
                        logger.info(f"[{dt.date()}] {cat_name} (Atomic): {w_atomic_str}")





                
                # 日志 (可选)
                # logger.info(f"{dt.date()} Top Alloc: {dict(zip(valid_cat_names, np.round(w_top_vec, 3)))}")

    # ==========================================================================
    # 循环结束：生成最终回测结果
    # ==========================================================================
    

    # 关键修正：将原子权重的列名从 Index 映射为 ETF，以便与 ETF 收益率表对齐
    final_atomic_w_hist_etf_df = final_atomic_w_hist_df.rename(columns=mapping)
    if use_etf_real_data:
        plot_w = final_atomic_w_hist_etf_df
        plot_ret = all_etf_simp_df.loc[final_atomic_w_hist_etf_df.index[0]:]
    else:
        plot_w = final_atomic_w_hist_df
        plot_ret = all_idx_simp_df.loc[final_atomic_w_hist_df.index[0]:]

    # 带入计算回测
    port_ret_frictionless, port_ret, equity_curve_series_frictionless, equity_curve_series, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=plot_w,
        daily_returns_df=plot_ret,
        cost_per_side_scalar=cost_per_side,
    )
    
    # 获取所有调仓日期
    
    report = calculate_performance_metrics(
        port_ret, 
        equity_curve_series, 
        rf_rate, 
        TRADING_DAYS_PER_YEAR_SCALAR,
        rebalance_dates=plot_w.index.to_list()
    )

    total_cost_series = (final_atomic_w_hist_etf_df.diff().abs().sum(axis=1).astype('float64') * cost_per_side).shift(1)
    total_cost_series = total_cost_series.reindex(port_ret.index).astype('float64').fillna(0.0)
    total_cost = float(total_cost_series.sum())
    logger.info(f"总交易成本(单位权重*单边成本累加)={total_cost:.6f}")
    top_w_hist_df_for_plot = top_w_hist_df.reindex(daily_w.index).ffill()

    # --- 收益归因计算 ---
    # 1. 底层资产归因 (使用 ETF 收益，并分配交易成本)
    #    将交易成本直接扣除在产生交易的资产贡献中
    atomic_attribution = calculate_return_attribution(
        weights_df=final_atomic_w_hist_etf_df,
        daily_returns_df=all_etf_simp_df,
        cost_per_side=cost_per_side
    )

    # 2. 顶层资产归因 (基于底层归因聚合)
    #    用户要求：不重新计算路径，而是将底层资产的贡献按类别加总
    top_attribution_dict = {}
    for cat_name in cat_names:
        # 获取该类别下的所有 ETF 列名
        cat_etfs = etf_by_cat_dict_df[cat_name].columns
        # 在 atomic_attribution 中找到属于该类别的资产并求和
        # 注意：atomic_attribution 的索引是 ETF 名称
        valid_etfs = [e for e in cat_etfs if e in atomic_attribution.index]
        top_attribution_dict[cat_name] = atomic_attribution[valid_etfs].sum()
    
    top_attribution = pd.Series(top_attribution_dict).sort_values(ascending=False)

    figures = {}
    if auto_plot:
        fig1 = plot_equity_curve(equity_curve_series, report, STRATEGY_MODE, auto_show=False)
        fig2 = plot_weights_history(top_w_hist_df_for_plot, STRATEGY_MODE, reverse_legend=True, auto_show=False)
        fig3 = plot_weights_history(daily_w, STRATEGY_MODE, reverse_legend=True, auto_show=False)
        
        # 绘制归因图
        fig4 = plot_return_attribution(top_attribution, title=f"{STRATEGY_MODE} - 顶层资产收益贡献", auto_show=False)
        fig5 = plot_return_attribution(atomic_attribution, title=f"{STRATEGY_MODE} - 底层资产收益贡献", auto_show=False)
        
        plt.show()
        figures = {
            'equity': fig1, 
            'weights_top': fig2, 
            'weights_atomic': fig3,
            'attrib_top': fig4,
            'attrib_atomic': fig5
        }
    return {
        'category_weights_history_df': top_w_hist_df,
        'inside_weights_hist_by_cat': inside_w_hist_dict,
        'final_atomic_weights_history_df': final_atomic_w_hist_etf_df,
        'final_portfolio_returns_series': port_ret,
        'equity_curve_series': equity_curve_series,
        'performance_report': report,
        'total_cost': total_cost,
        'top_attribution': top_attribution,
        'atomic_attribution': atomic_attribution,
        'figures': figures,
        'all_index_assets_daily_returns_df': all_idx_simp_df,
        'all_etf_assets_daily_returns_df': all_etf_simp_df,
        'backtest_start': start_dt_global,
        'backtest_end': end_dt_global,
    }


if __name__ == "__main__":
    res = run_strategy(
        data_dir='data',
        top_level_method='PCA_RP',
        n_factors = 3,
        asset_weight_cap=None,
        asset_ridge_alpha=1e-2,
        auto_plot=True,
        start_date="20200101",
        end_date="20251031",
        lookback_years = 3,
        top_cov_estimate_ways = 'ewm',
        ewm_span_days = 252,
        use_etf_real_data = True,
    )
    perf = res['performance_report']
    logger.info("多层配置完成：主要指标")
    for k, v in perf.items():
        try:
            if isinstance(v, pd.Timestamp):
                logger.info(f"  {k}: {v.strftime('%Y-%m-%d')}")
            elif any(x in k for x in ['CAGR', 'Vol', '回撤', '胜率']) and not isinstance(v, pd.Timestamp):
                logger.info(f"  {k}: {v:.2%}")
            else:
                logger.info(f"  {k}: {v:.4f}")
        except Exception:
            logger.info(f"  {k}: {v}")
