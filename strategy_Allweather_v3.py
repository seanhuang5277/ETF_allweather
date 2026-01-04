# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_Allweather_v3.py

@Description:
All Weather (V3): 在“底层可交易资产(ETF)”层面做 Growth/Inflation 两因子风险平价。

核心区别（相对 v2）：
- v2：先对“四象限组合”求权重，再穿透到 ETF。
- v3：直接对“底层 ETF”回归估计 β，然后在 ETF 权重 w 上做因子风险平价。

实现流程（每次调仓日 dt）：
1) 用 dt 上月月末及以前的样本：对每个 ETF 做 OLS：r_i,t = a + β_i,g F_g,t + β_i,π F_π,t + ε_i,t
2) 用同一段样本估计 Σ_F = Cov(F)
3) 在 w 上优化，使因子风险贡献接近目标（默认等权或来自因子空间 RP）：RC_g ≈ RC_π

注意：
- 因子使用 Level/Change 体系中的 Change：CN_Growth_Change / CN_Inflation_Change
- 为避免未来函数：所有回归与协方差都只使用 dt 上月月末及更早的数据
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
    solve_factor_risk_parity_weights,
    solve_risk_parity_weights,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import (
    plot_equity_curve,
    plot_weights_history,
    plot_return_attribution,
    plot_multi_equity_curves,
)

# 初始化日志
setup_logging(level="INFO", log_to_file=True, filename="logs/strategy_Allweather_v3.log")
logger = get_logger("strategy.AllWeather.v3")

STRATEGY_MODE = 'All_Weather_FactorRP_Bottom'


def _to_month_end_simple_returns(daily_simple_ret_df: pd.DataFrame) -> pd.DataFrame:
    """将日频简单收益压缩成月末简单收益（按月复利）。"""
    if daily_simple_ret_df is None or daily_simple_ret_df.empty:
        return pd.DataFrame()
    mret = (1.0 + daily_simple_ret_df).groupby(daily_simple_ret_df.index.to_period('M')).prod() - 1.0
    mret.index = mret.index.to_timestamp('M')
    return mret


def get_factor_risk_budget_from_macro(
    macro_factor_df: pd.DataFrame,
    dt: pd.Timestamp,
    lookback_years: int,
    min_clean_days: int,
    method: str = "cov",
    ewm_span_days: int = 252,
) -> Optional[pd.Series]:
    """在因子空间内做 2x2 风险平价，返回目标因子预算（Series）。

    这一步只负责得到 target_factor_budget（例如 Growth/Inflation 50/50 或由协方差决定）。
    真正资产权重由 solve_factor_risk_parity_weights 在 w 上求解。
    """
    if macro_factor_df is None or macro_factor_df.empty:
        return None

    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    factor_cols = ['CN_Growth_Change', 'CN_Inflation_Change']
    if any(c not in macro_factor_df.columns for c in factor_cols):
        return None

    hist = macro_factor_df.loc[:dt_prev_month_end, factor_cols].dropna()
    if hist.empty:
        return None

    lookback_periods = int(12 * lookback_years)
    if lookback_periods > 0:
        hist = hist.iloc[-lookback_periods:]

    min_obs = max(24, int(min_clean_days / 21))
    if len(hist) < min_obs:
        return None

    use_ewm = isinstance(method, str) and ('ewm' in method.lower())
    if use_ewm:
        span_m = max(6, int(ewm_span_days / 21))
        last_date = hist.index[-1]
        cov_f = hist.ewm(span=span_m).cov().loc[last_date]
    else:
        cov_f = hist.cov()

    cov_f = cov_f.loc[factor_cols, factor_cols]
    w = solve_risk_parity_weights(cov_f, None)
    w = w.reindex(factor_cols).fillna(0.0)
    s = float(w.sum())
    if s > 0:
        w = w / s
    return w


def get_position_sizing_from_monetary(
    macro_factor_df: pd.DataFrame,
    dt: pd.Timestamp,
    max_position: float = 1.0,
    min_position: float = 0.8,
    lookback_months: int = 3,
) -> float:
    """根据货币政策因子调整总仓位系数（正值=宽松）。"""
    if macro_factor_df is None or macro_factor_df.empty:
        return max_position

    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    macro_idx = macro_factor_df.index[macro_factor_df.index <= dt_prev_month_end]
    if len(macro_idx) < lookback_months:
        return max_position

    latest_idx = macro_factor_df.index.get_indexer([dt_prev_month_end], method='pad')[0]
    if latest_idx < 2:
        return max_position

    monetary_col = None
    if 'CN_Monetary_Level' in macro_factor_df.columns:
        monetary_col = 'CN_Monetary_Level'
    elif 'Monetary Policy_OECD' in macro_factor_df.columns:
        monetary_col = 'Monetary Policy_OECD'

    if monetary_col is None or pd.isna(macro_factor_df.iloc[latest_idx][monetary_col]):
        return max_position

    m0 = macro_factor_df.iloc[latest_idx][monetary_col]
    m1 = macro_factor_df.iloc[latest_idx - 1][monetary_col]
    m2 = macro_factor_df.iloc[latest_idx - 2][monetary_col]

    if m0 > m1 > m2:
        return max_position
    if m0 < m1 < m2:
        return min_position
    return (max_position + min_position) / 2


def _collect_allowed_etfs(quadrant_defs: dict) -> List[str]:
    etfs: set[str] = set()
    for q in quadrant_defs.values():
        for cls, arr in q.items():
            for etf in arr[1:]:
                etfs.add(etf)
    return sorted(etfs)


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
    # 宏观数据
    macro_factor_adjustment: bool = False,  # v3 暂不做象限倾斜，仅保留参数兼容
    use_monetary_position_sizing: bool = False,
    max_position: float = 1.0,
    min_position: float = 0.8,
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
            logger.warning("未找到宏观因子文件，将跳过因子RP并回退等权。")
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

    # 回测循环
    for dt in rebalance_dates:
        start_dt = dt - pd.DateOffset(days=lookback_window_days)
        dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))

        # 选择用于估计与回测的底层收益（列名=ETF）
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
            # 2) 资产收益（月频）
            assets_m = _to_month_end_simple_returns(window_daily[live_etfs])
            # 3) 对齐并截取窗口
            hist = assets_m.join(factors_m, how='inner').dropna()
            lookback_periods = int(12 * lookback_years)
            if lookback_periods > 0:
                hist = hist.iloc[-lookback_periods:]

            min_obs = max(24, int(min_clean_days / 21))
            if len(hist) < min_obs:
                logger.warning(f"[{dt.date()}] 月频样本不足({len(hist)}<{min_obs})，回退等权")
                w_assets = pd.Series(1.0 / len(live_etfs), index=live_etfs)
            else:
                # 4) β 暴露（sm.OLS）
                betas_df, _ = estimate_factor_betas_ols(
                    returns_df=hist[live_etfs],
                    factors_df=hist[factor_cols],
                    min_obs=min_obs,
                )
                # 5) Σ_F
                use_ewm = isinstance(factor_cov_method, str) and ('ewm' in factor_cov_method.lower())
                if use_ewm:
                    span_m = max(6, int(ewm_span_days / 21))
                    last_date = hist.index[-1]
                    cov_f = hist[factor_cols].ewm(span=span_m).cov().loc[last_date]
                else:
                    cov_f = hist[factor_cols].cov()
                cov_f = cov_f.loc[factor_cols, factor_cols]

                # 6) 目标因子预算（可选：用因子空间 RP 得到；否则默认等权）
                target_factor_budget = get_factor_risk_budget_from_macro(
                    macro_factor_df=macro_factor_df,
                    dt=dt,
                    lookback_years=lookback_years,
                    min_clean_days=min_clean_days,
                    method=factor_cov_method,
                    ewm_span_days=ewm_span_days,
                )

                w_assets = solve_factor_risk_parity_weights(
                    betas_df=betas_df,
                    factor_cov_df=cov_f,
                    target_factor_budget=target_factor_budget,
                    ridge_lambda=1e-4,
                    long_only=True,
                    weight_cap=None,
                )

        # 扩展到全资产列
        current_atomic_w = pd.Series(0.0, index=all_etf_simp_df.columns)
        current_atomic_w.loc[w_assets.index] = w_assets.values
        if current_atomic_w.sum() > 0:
            current_atomic_w = current_atomic_w / current_atomic_w.sum()

        # 货币政策仓位调整（缩放 + 现金）
        if use_monetary_position_sizing and macro_factor_df is not None:
            pos_ratio = get_position_sizing_from_monetary(macro_factor_df, dt, max_position, min_position)
            current_atomic_w *= pos_ratio
            cash_weight = max(0.0, 1.0 - float(current_atomic_w.sum()))
            if '短融ETF' in current_atomic_w.index:
                current_atomic_w.loc['短融ETF'] = cash_weight
            if current_atomic_w.sum() > 0:
                current_atomic_w = current_atomic_w / current_atomic_w.sum()

        bottom_atomic_w_hist.loc[dt] = current_atomic_w

        # 用成员 ETF 权重“映射”成象限权重（仅用于画图/诊断）
        quad_w = {}
        for q_name, q_def in quadrant_defs.items():
            members = _collect_allowed_etfs({q_name: q_def})
            members = [e for e in members if e in current_atomic_w.index]
            quad_w[q_name] = float(current_atomic_w.loc[members].sum()) if members else 0.0

            # 象限内归一化权重（用于象限净值曲线）
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

    # 象限净值曲线（用于对比）
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

    # 供画图：将调仓权重对齐到日频
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
        use_monetary_position_sizing=False,
        use_etf_real_data=False,
        factor_cov_method='cov',
        ewm_span_days=252,
    )

    perf = res['performance_report']
    logger.info("All Weather v3 运行完成，主要指标：")
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

    fig1 = plot_equity_curve(res['equity_curve_series'], res['performance_report'], STRATEGY_MODE,
                            rebalance_returns=res['rebalance_returns'], auto_show=False)
    fig2 = plot_weights_history(res['top_quadrant_for_plot'], "Quadrant Exposure (proxy)", reverse_legend=True, auto_show=False)
    fig3 = plot_weights_history(res['daily_weights_df'], "Asset Allocation", reverse_legend=True, auto_show=False)
    fig4 = plot_return_attribution(res['atomic_attribution'], title="底层资产收益贡献", auto_show=False)
    fig5 = plot_multi_equity_curves(res['quadrant_equity_curves'], title="四象限净值曲线对比(成员ETF归一化)", auto_show=False)

    plt.show()
