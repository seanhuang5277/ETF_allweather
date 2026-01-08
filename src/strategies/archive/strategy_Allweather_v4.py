# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_Allweather_v4.py

@Description:
All Weather (V4): 方案二 / 做法 C —— 先设定“目标因子暴露”，再映射到资产（工程折中、可控）。

核心思想：
- 先决定组合在因子空间里“想长什么样”(b_p*)
- 再解约束最小二乘：min_w || B^T w - b_p* ||^2 + λ||w - w_prev||^2

实现要点：
- 因子使用 Level/Change 体系中的 Change：CN_Growth_Change / CN_Inflation_Change
- β 使用 statsmodels OLS（与研究脚本一致）
- 目标因子暴露 b_p*：默认采用“等风险贡献(ERC) 50/50”的方向，并用当期可实现规模做缩放
- 约束：sum(w)=1，w>=0，单资产上限(默认 20%)，可选最小换手惩罚
- 为避免未来函数：所有回归、因子协方差只使用 dt 上月月末及更早数据
"""

from __future__ import annotations

import os
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from framework.config import (
    TRADING_DAYS_PER_YEAR_SCALAR,
    COST_PER_SIDE_SCALAR,
    RISK_FREE_RATE_SCALAR_ANNUAL,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_DATA_YEARS,
    MIN_CLEAN_DAYS,
)
from framework.load_data import load_etf_index_returns_by_category
from framework.performance import (
    compute_portfolio_returns_and_equity,
    calculate_performance_metrics,
    calculate_return_attribution,
)
from framework.allocation_utils import (
    to_daily_rebalance_dates,
    estimate_factor_betas_ols,
    map_factor_to_asset_weights,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import (
    plot_equity_curve,
    plot_weights_history,
    plot_return_attribution,
    plot_multi_equity_curves,
)

setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_Allweather_v4.log")
logger = get_logger("strategy.AllWeather.v4")

STRATEGY_MODE = 'All_Weather_TargetExposure_Bottom'


def _to_month_end_simple_returns(daily_simple_ret_df: pd.DataFrame) -> pd.DataFrame:
    """将日频简单收益压缩成月末简单收益（按月复利）。"""
    if daily_simple_ret_df is None or daily_simple_ret_df.empty:
        return pd.DataFrame()
    mret = (1.0 + daily_simple_ret_df).groupby(daily_simple_ret_df.index.to_period('M')).prod() - 1.0
    mret.index = mret.index.to_timestamp('M')
    return mret


def _collect_allowed_etfs(quadrant_defs: dict) -> List[str]:
    etfs: set[str] = set()
    for q in quadrant_defs.values():
        for cls, arr in q.items():
            for etf in arr[1:]:
                etfs.add(etf)
    return sorted(etfs)


def _estimate_factor_cov(
    factors_hist: pd.DataFrame,
    *,
    method: str = 'cov',
    ewm_span_days: int = 252,
) -> pd.DataFrame:
    use_ewm = isinstance(method, str) and ('ewm' in method.lower())
    if use_ewm:
        span_m = max(6, int(ewm_span_days / 21))
        last_date = factors_hist.index[-1]
        cov_f = factors_hist.ewm(span=span_m).cov().loc[last_date]
    else:
        cov_f = factors_hist.cov()
    return cov_f


def _solve_target_factor_exposure_equal_rc(
    factor_cov_df: pd.DataFrame,
    *,
    target_budget: Optional[pd.Series] = None,
    n_grid: int = 7200,
) -> pd.Series:
    """在“因子空间”里求一个目标暴露方向，使因子风险贡献比例接近目标。

    说明：
    - 这里只求方向（单位范数），最终会按可实现的规模做缩放。
    - 风险贡献定义：RC_k = | b_k * (Σ_F b)_k |
    """
    factor_cols = list(factor_cov_df.columns)
    k = len(factor_cols)
    if k == 0:
        return pd.Series(dtype=float)
    if k == 1:
        return pd.Series([1.0], index=factor_cols)

    Sigma = factor_cov_df.to_numpy(dtype=float)
    # 目标预算（默认等权）
    if target_budget is None:
        tgt = np.full(k, 1.0 / k, dtype=float)
    else:
        t = target_budget.reindex(factor_cols).astype(float).to_numpy()
        t = np.clip(t, 0.0, None)
        s = float(t.sum())
        tgt = (t / s) if s > 0 else np.full(k, 1.0 / k, dtype=float)

    # 目前策略只使用 2 因子，这里用角度网格搜索最稳健
    if k != 2:
        # 简化退化：多因子时直接等权方向
        b = np.ones(k)
        b = b / np.linalg.norm(b)
        return pd.Series(b, index=factor_cols)

    thetas = np.linspace(-np.pi, np.pi, int(max(360, n_grid)), endpoint=False)
    best_theta = 0.0
    best_loss = np.inf

    for th in thetas:
        b = np.array([np.cos(th), np.sin(th)], dtype=float)
        m = Sigma @ b
        rc = np.abs(b * m)
        s = float(rc.sum())
        if s <= 1e-18:
            rc_norm = np.array([0.5, 0.5], dtype=float)
        else:
            rc_norm = rc / s
        loss = float(np.sum((rc_norm - tgt) ** 2))
        if loss < best_loss:
            best_loss = loss
            best_theta = th

    b_best = np.array([np.cos(best_theta), np.sin(best_theta)], dtype=float)
    nrm = float(np.linalg.norm(b_best))
    if nrm > 0:
        b_best = b_best / nrm
    return pd.Series(b_best, index=factor_cols)


def _scale_target_exposure_to_feasible_magnitude(
    target_dir: pd.Series,
    betas_df: pd.DataFrame,
    *,
    ref_weights: Optional[pd.Series] = None,
) -> pd.Series:
    """把目标暴露“方向”缩放到一个更可实现的数量级。

    默认用等权(或 ref_weights)组合的当期暴露作为尺度：
    scale = ||B^T w_ref||。
    """
    if betas_df is None or betas_df.empty or target_dir is None or target_dir.empty:
        return target_dir

    assets = list(betas_df.index)
    if ref_weights is None:
        w_ref = np.full(len(assets), 1.0 / len(assets), dtype=float)
    else:
        w_ref = ref_weights.reindex(assets).fillna(0.0).to_numpy(dtype=float)
        s = float(w_ref.sum())
        if abs(s) > 1e-12:
            w_ref = w_ref / s
        else:
            w_ref = np.full(len(assets), 1.0 / len(assets), dtype=float)

    B = betas_df.reindex(columns=target_dir.index).to_numpy(dtype=float)  # (N, K)
    b_ref = B.T @ w_ref  # (K,)

    scale = float(np.linalg.norm(b_ref))
    if not np.isfinite(scale) or scale <= 1e-12:
        return target_dir

    return target_dir * scale


def run_strategy_updated(
    *,
    data_dir: str = 'data',
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    rebalance_day: int = None,
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    use_etf_real_data: bool = True,
    # 映射约束
    weight_cap: float = 0.20,
    turnover_lambda: float = 0.0,
    # 因子协方差估计
    factor_cov_method: str = 'cov',
    ewm_span_days: int = 252,
) -> Dict[str, object]:

    start_dt_global = pd.to_datetime(start_date) if start_date is not None else None
    end_dt_global = pd.to_datetime(end_date) if end_date is not None else None

    # 1) 加载宏观因子（优先 Level/Change）
    macro_factor_df = None
    try:
        all_path = f"{data_dir}/all_macro_factors.csv"
        syn_path = f"{data_dir}/macro_factors_synthesized.csv"
        if os.path.exists(all_path):
            macro_factor_df = pd.read_csv(all_path, index_col=0, parse_dates=True).sort_index()
            logger.info(f"已加载宏观因子(Level/Change): {all_path}")
        elif os.path.exists(syn_path):
            macro_factor_df = pd.read_csv(syn_path, index_col='date', parse_dates=True).sort_index()
            logger.info(f"已加载宏观因子(旧版synthesized): {syn_path}")
        else:
            logger.warning("未找到宏观因子文件，将回退等权。")
            macro_factor_df = None
    except Exception as e:
        logger.warning(f"加载宏观因子失败: {e}")
        macro_factor_df = None

    # 2) 加载行情数据
    etf_by_cat_dict_df, idx_by_cat_dict_df = load_etf_index_returns_by_category(data_dir=data_dir, returns_index_col='日期')

    all_index_list = [idx_by_cat_dict_df[c] for c in idx_by_cat_dict_df.keys()]
    all_etf_list = [etf_by_cat_dict_df[c] for c in etf_by_cat_dict_df.keys()]
    all_idx_simp_df = pd.concat(all_index_list, axis=1).sort_index()
    all_etf_simp_df = pd.concat(all_etf_list, axis=1).sort_index()

    mapping = {x: y for x, y in zip(all_idx_simp_df.columns, all_etf_simp_df.columns)}
    reversed_mapping = {y: x for x, y in mapping.items()}

    # 3) 定义四象限的 ETF 集合（仅作为“允许的底层资产池”）
    quadrant_defs = {
        'Growth_Rising': {
            'Growth_Rising_Stock': [0.8, '沪深300ETF', '中证500ETF', '中证1000ETF'],
            'Growth_Rising_Commodity': [0.1, '有色ETF', '豆粕ETF', '能源化工ETF'],
            'Growth_Rising_Credit_Bond': [0.1, '信用债ETF', '可转债ETF'],
        },
        'Growth_Falling': {
            'Growth_Falling_Rates_Bond': [0.5, '国债ETF', '十年国债ETF'],
            'Growth_Falling_Credit_Bond': [0.0, '公司债ETF', '信用债ETF', '可转债ETF'],
            'Growth_Falling_Gold': [0.5, '黄金ETF'],
        },
        'Inflation_Rising': {
            'Inflation_Rising_Commodity': [0.5, '有色ETF', '豆粕ETF', '能源化工ETF'],
            'Inflation_Rising_Gold': [0.5, '黄金ETF'],
        },
        'Inflation_Falling': {
            'Inflation_Falling_Dividend_Stock': [0.2, '红利低波50ETF', '红利质量ETF'],
            'Inflation_Falling_Rates_Bond': [0.8, '国债ETF', '十年国债ETF', '30年国债ETF', '公司债ETF', '可转债ETF'],
        },
    }
    quadrant_names = list(quadrant_defs.keys())
    allowed_etfs = _collect_allowed_etfs(quadrant_defs)

    # 4) ETF 上市时间
    etf_inception: Dict[str, pd.Timestamp] = {}
    for idx_name, etf_name in mapping.items():
        s = all_etf_simp_df[etf_name] if use_etf_real_data else all_idx_simp_df[idx_name]
        first_inc = s.first_valid_index()
        if first_inc is not None:
            etf_inception[etf_name] = first_inc

    # 5) 再平衡日期
    min_data_days_required = TRADING_DAYS_PER_YEAR_SCALAR * min_data_years
    rebalance_dates = to_daily_rebalance_dates(all_idx_simp_df, min_data_days_required, rebalance_day=rebalance_day)
    if start_dt_global is not None:
        future_rebals = rebalance_dates[rebalance_dates >= start_dt_global]
        if start_dt_global not in future_rebals and start_dt_global >= all_idx_simp_df.index[0] + pd.DateOffset(days=min_data_days_required):
            rebalance_dates = pd.DatetimeIndex([start_dt_global] + list(future_rebals)).sort_values()
        else:
            rebalance_dates = future_rebals
    if end_dt_global is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end_dt_global]

    lookback_window_days = TRADING_DAYS_PER_YEAR_SCALAR * lookback_years

    # 输出历史
    bottom_atomic_w_hist = pd.DataFrame(index=rebalance_dates, columns=all_etf_simp_df.columns, dtype=float).fillna(0.0)
    top_quadrant_w_hist = pd.DataFrame(index=rebalance_dates, columns=quadrant_names, dtype=float).fillna(0.0)
    quadrant_atomic_weights_hist = {
        q: pd.DataFrame(index=rebalance_dates, columns=all_etf_simp_df.columns, dtype=float).fillna(0.0)
        for q in quadrant_names
    }

    prev_w_assets: Optional[pd.Series] = None

    for dt in rebalance_dates:
        start_dt = dt - pd.DateOffset(days=lookback_window_days)
        dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))

        if use_etf_real_data:
            daily_ret_all = all_etf_simp_df
        else:
            daily_ret_all = all_idx_simp_df.rename(columns=mapping)

        window_daily = daily_ret_all.loc[start_dt:dt_prev_month_end]

        # 筛选存活 ETF（只在允许清单内）
        live_etfs = []
        for etf in allowed_etfs:
            inc = etf_inception.get(etf)
            if inc is None or inc > dt:
                continue
            if etf not in window_daily.columns:
                continue
            if window_daily[etf].count() < min_data_days_required:
                continue
            live_etfs.append(etf)

        if len(live_etfs) < 2:
            logger.warning(f"[{dt.date()}] 可用 ETF 数量不足({len(live_etfs)})，跳过")
            continue

        # 1) 因子样本（月频）
        factor_cols = ['CN_Growth_Change', 'CN_Inflation_Change']
        if macro_factor_df is None or any(c not in macro_factor_df.columns for c in factor_cols):
            logger.warning(f"[{dt.date()}] 宏观因子缺失，回退等权")
            w_assets = pd.Series(1.0 / len(live_etfs), index=live_etfs)
        else:
            factors_m = macro_factor_df.loc[:dt_prev_month_end, factor_cols].dropna()
            assets_m = _to_month_end_simple_returns(window_daily[live_etfs])
            hist = assets_m.join(factors_m, how='inner').dropna()

            lookback_periods = int(12 * lookback_years)
            if lookback_periods > 0:
                hist = hist.iloc[-lookback_periods:]

            min_obs = max(24, int(min_clean_days / 21))
            if len(hist) < min_obs:
                logger.warning(f"[{dt.date()}] 月频样本不足({len(hist)}<{min_obs})，回退等权")
                w_assets = pd.Series(1.0 / len(live_etfs), index=live_etfs)
            else:
                # 2) β 暴露
                betas_df, _ = estimate_factor_betas_ols(
                    returns_df=hist[live_etfs],
                    factors_df=hist[factor_cols],
                    min_obs=min_obs,
                )
                # 3) Σ_F
                cov_f = _estimate_factor_cov(hist[factor_cols], method=factor_cov_method, ewm_span_days=ewm_span_days)
                cov_f = cov_f.loc[factor_cols, factor_cols]

                # 4) 目标因子暴露：ERC 50/50（方向）
                b_dir = _solve_target_factor_exposure_equal_rc(cov_f)

                # 5) 缩放到可实现规模（参考：上一期权重或等权）
                if prev_w_assets is None:
                    b_star = _scale_target_exposure_to_feasible_magnitude(b_dir, betas_df)
                else:
                    b_star = _scale_target_exposure_to_feasible_magnitude(b_dir, betas_df, ref_weights=prev_w_assets)

                # 6) 映射到资产：min ||B^T w - b*||^2 + λ||w-w_prev||^2
                beta_matrix = betas_df.reindex(columns=factor_cols).to_numpy(dtype=float)  # (N,2)
                target_beta_exposure_arry = b_star.reindex(factor_cols).to_numpy(dtype=float)

                prior = None
                if turnover_lambda and turnover_lambda > 0 and prev_w_assets is not None:
                    prior = prev_w_assets.reindex(live_etfs).fillna(0.0).to_numpy(dtype=float)

                w_vec = map_factor_to_asset_weights(
                    V_k=beta_matrix,
                    target_factor_exposure=target_beta_exposure_arry,
                    weight_cap=weight_cap,
                    factor_cov_matrix=None,
                    allow_short=False,
                    enforce_budget_constraint=True,
                )
                w_assets = pd.Series(w_vec, index=live_etfs)
                s = float(w_assets.sum())
                if s > 0:
                    w_assets = w_assets / s

        # 扩展到全资产列
        current_atomic_w = pd.Series(0.0, index=all_etf_simp_df.columns)
        current_atomic_w.loc[w_assets.index] = w_assets.values
        if current_atomic_w.sum() > 0:
            current_atomic_w = current_atomic_w / current_atomic_w.sum()

        bottom_atomic_w_hist.loc[dt] = current_atomic_w
        prev_w_assets = w_assets.copy()

        # 象限暴露 proxy（用于画图/诊断）
        quad_w = {}
        for q_name, q_def in quadrant_defs.items():
            members = _collect_allowed_etfs({q_name: q_def})
            members = [e for e in members if e in current_atomic_w.index]
            quad_w[q_name] = float(current_atomic_w.loc[members].sum()) if members else 0.0

            q_w = current_atomic_w.loc[members].copy() if members else pd.Series(dtype=float)
            if len(q_w) > 0 and q_w.sum() > 0:
                q_w = q_w / q_w.sum()
                quadrant_atomic_weights_hist[q_name].loc[dt, q_w.index] = q_w

        quad_sum = float(sum(quad_w.values()))
        if quad_sum > 0:
            quad_w = {k: v / quad_sum for k, v in quad_w.items()}
        top_quadrant_w_hist.loc[dt] = pd.Series(quad_w)

    # -----------------------------
    # 结果计算
    # -----------------------------
    bottom_atomic_w_index = bottom_atomic_w_hist.rename(columns=reversed_mapping)
    if use_etf_real_data:
        plot_w = bottom_atomic_w_hist
        plot_ret = all_etf_simp_df.loc[bottom_atomic_w_hist.index[0]:]
    else:
        plot_w = bottom_atomic_w_index
        plot_ret = all_idx_simp_df.loc[bottom_atomic_w_index.index[0]:]

    _, port_ret, _, equity_curve_series, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=plot_w,
        daily_returns_df=plot_ret,
        cost_per_side_scalar=cost_per_side,
    )

    report = calculate_performance_metrics(
        port_ret,
        equity_curve_series,
        rf_rate,
        TRADING_DAYS_PER_YEAR_SCALAR,
        rebalance_dates=rebalance_dates,
    )

    atomic_attribution = calculate_return_attribution(
        weights_df=bottom_atomic_w_hist,
        daily_returns_df=all_etf_simp_df,
        cost_per_side=cost_per_side,
    )

    quadrant_equity_curves = {}
    for q_name, q_weights_hist in quadrant_atomic_weights_hist.items():
        if use_etf_real_data:
            q_plot_w = q_weights_hist
            q_plot_ret = all_etf_simp_df.loc[q_weights_hist.index[0]:]
        else:
            q_plot_w = q_weights_hist.rename(columns=reversed_mapping)
            q_plot_ret = all_idx_simp_df.loc[q_plot_w.index[0]:]

        if q_plot_w.sum().sum() <= 0:
            continue

        _, _, _, q_curve, _ = compute_portfolio_returns_and_equity(
            weights_history_df=q_plot_w,
            daily_returns_df=q_plot_ret,
            cost_per_side_scalar=cost_per_side,
        )
        quadrant_equity_curves[q_name] = q_curve

    top_quadrant_for_plot = top_quadrant_w_hist.reindex(daily_w.index).ffill()

    valid_rebalance_dates = rebalance_dates[rebalance_dates >= equity_curve_series.index[0]]
    rebalance_returns = equity_curve_series.loc[valid_rebalance_dates].pct_change().dropna()

    return {
        'performance_report': report,
        'equity_curve_series': equity_curve_series,
        'final_weights_df': bottom_atomic_w_hist,
        'quadrant_weights_df': top_quadrant_w_hist,
        'daily_weights_df': daily_w,
        'top_quadrant_for_plot': top_quadrant_for_plot,
        'atomic_attribution': atomic_attribution,
        'rebalance_returns': rebalance_returns,
        'quadrant_equity_curves': quadrant_equity_curves,
    }


if __name__ == "__main__":
    res = run_strategy_updated(
        start_date="2015-11-30",
        end_date="2025-11-30",
        rebalance_day=None,
        use_etf_real_data=False,
        weight_cap=0.20,
        turnover_lambda=10.0,
        factor_cov_method='cov',
        ewm_span_days=252,
    )

    perf = res['performance_report']
    logger.info("All Weather v4 运行完成，主要指标：")
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

    fig1 = plot_equity_curve(
        res['equity_curve_series'],
        res['performance_report'],
        STRATEGY_MODE,
        rebalance_returns=res['rebalance_returns'],
        auto_show=False,
    )
    fig2 = plot_weights_history(res['top_quadrant_for_plot'], "Quadrant Exposure (proxy)", reverse_legend=True, auto_show=False)
    fig3 = plot_weights_history(res['daily_weights_df'], "Asset Allocation", reverse_legend=True, auto_show=False)
    fig4 = plot_return_attribution(res['atomic_attribution'], title="底层资产收益贡献", auto_show=False)
    fig5 = plot_multi_equity_curves(res['quadrant_equity_curves'], title="四象限净值曲线对比(成员ETF归一化)", auto_show=False)

    plt.show()
