# -*- coding: utf-8 -*-
"""performance.py

绩效与组合路径计算模块。
从单一策略脚本中抽取的通用功能：
1. 根据调仓后每日权重与每日简单收益率计算组合每日收益、净值曲线。
2. 计算标准绩效指标（与原策略保持一致）。

函数命名采用统一后缀：_series / _df / _scalar。
"""
from __future__ import annotations
import pandas as pd
import numpy as np

# -----------------------------------------------
# 核心组合路径与绩效计算
# -----------------------------------------------

def compute_portfolio_returns_and_equity(
    weights_history_df: pd.DataFrame,
    daily_returns_df: pd.DataFrame,
    cost_per_side_scalar: float,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """根据调仓日权重历史与每日收益率计算组合净值。

    参数:
        weights_history_df: 调仓日权重 (index=调仓日期, columns=资产)。
        daily_returns_df: 对齐的每日简单收益率 (index=交易日, columns=资产)。
        cost_per_side_scalar: 单边交易成本比例。
    返回:
        final_portfolio_returns_series: 含成本组合日收益
        equity_curve_series: 净值曲线 (初始=1)
        daily_weights_df: 对齐并前填的每日持仓权重 (用于外部诊断/绘图)
    """
    # 确保为数值型，避免 pandas 对 object dtype 的 downcasting 警告
    weights_history_df = weights_history_df.apply(pd.to_numeric, errors='coerce')
    daily_returns_df = daily_returns_df.apply(pd.to_numeric, errors='coerce')

    # 将调仓日权重前向填充到每日
    daily_weights_df = weights_history_df.reindex(daily_returns_df.index).ffill()
    # 避免前视：使用前一日权重
    tradeable_weights_df = daily_weights_df.shift(1)
    tradeable_weights_df = tradeable_weights_df.dropna(how='all')
    aligned_returns_df = daily_returns_df.loc[tradeable_weights_df.index]

    # 无摩擦收益
    portfolio_daily_returns_series_frictionless = (tradeable_weights_df * aligned_returns_df).sum(axis=1)

    # 换手与成本计算
    weights_diff_df = weights_history_df.diff().abs()
    turnover_series = weights_diff_df.sum(axis=1).astype('float64')
    transaction_costs_series = (turnover_series * cost_per_side_scalar).shift(1)
    # 确保交易成本序列为 float，避免未来版本的 downcasting 行为变化
    transaction_costs_series = (
        transaction_costs_series
        .reindex(portfolio_daily_returns_series_frictionless.index)
        .astype('float64')
        .fillna(0.0)
    )

    final_portfolio_returns_series = portfolio_daily_returns_series_frictionless - transaction_costs_series
    equity_curve_series_frictionless = (1 + portfolio_daily_returns_series_frictionless).cumprod()
    equity_curve_series = (1 + final_portfolio_returns_series).cumprod()
    equity_curve_series.name = 'Equity'
    return  [
        portfolio_daily_returns_series_frictionless,
        final_portfolio_returns_series, 
        equity_curve_series_frictionless,
        equity_curve_series, 
        daily_weights_df
    ]


def calculate_performance_metrics(
    returns_series: pd.Series,
    equity_curve_series: pd.Series,
    rf_rate_scalar: float,
    days_per_year_scalar: int,
    rebalance_dates: list[pd.Timestamp] | None = None,
) -> pd.Series:
    """计算标准绩效指标集合，与原策略实现保持一致。"""
    years = len(returns_series) / days_per_year_scalar
    end_value = equity_curve_series.iloc[-1]
    cagr = end_value ** (1 / years) - 1
    annual_ret = returns_series.mean() * days_per_year_scalar
    annual_vol = returns_series.std() * np.sqrt(days_per_year_scalar)
    sharpe = (annual_ret - rf_rate_scalar) / annual_vol if annual_vol != 0 else np.inf
    running_max = equity_curve_series.cummax()
    drawdown = (equity_curve_series - running_max) / running_max
    max_dd = drawdown.min()
    
    # 计算最大回撤区间
    max_dd_end_date = drawdown.idxmin()
    # 在最大回撤结束日之前，找到净值最高点作为开始日
    max_dd_start_date = equity_curve_series.loc[:max_dd_end_date].idxmax()
    
    calmar = cagr / abs(max_dd) if abs(max_dd) > 1e-6 else np.inf
    neg = returns_series[returns_series < 0]
    downside_dev = neg.std() * np.sqrt(days_per_year_scalar)
    sortino = (annual_ret - rf_rate_scalar) / downside_dev if downside_dev != 0 else np.inf
    
    # 计算年度收益率
    yearly_returns = {}
    equity_curve_with_year = equity_curve_series.to_frame('equity')
    equity_curve_with_year['year'] = equity_curve_with_year.index.year
    
    for year in equity_curve_with_year['year'].unique():
        year_data = equity_curve_with_year[equity_curve_with_year['year'] == year]['equity']
        if len(year_data) > 0:
            year_start_val = year_data.iloc[0]
            year_end_val = year_data.iloc[-1]
            if year_start_val != 0:
                year_ret = year_end_val / year_start_val - 1
                yearly_returns[f'Annual Return {year} (年度收益率)'] = year_ret
    
    metrics = {
        'CAGR (年化复合收益)': cagr,
        'Annual Volatility (年化波动率)': annual_vol,
        'Sharpe Ratio (夏普比率)': sharpe,
        'Calmar Ratio (Calmar比率)': calmar,
        'Sortino Ratio (索提诺比率)': sortino,
        'Max Drawdown (最大回撤)': max_dd,
        'Max Drawdown Start (最大回撤开始日)': max_dd_start_date,
        'Max Drawdown End (最大回撤结束日)': max_dd_end_date,
    }
    
    # 添加年度收益率
    metrics.update(yearly_returns)

    # 计算换仓周期胜率与赔率
    if rebalance_dates is not None and len(rebalance_dates) > 0:
        # 过滤出在回测区间内的调仓日
        backtest_start_dt = equity_curve_series.index[0]
        backtest_end_dt = equity_curve_series.index[-1]
        
        # 确保 rebalance_dates 是排序的
        sorted_dates = sorted(rebalance_dates)
        valid_starts = [d for d in sorted_dates if d < backtest_end_dt]
        
        period_returns = []
        for i in range(len(valid_starts)):
            start_dt = valid_starts[i]
            # 确定本周期的结束日
            if i < len(valid_starts) - 1:
                end_dt = valid_starts[i+1]
                if end_dt > backtest_end_dt:
                    end_dt = backtest_end_dt
            else:
                end_dt = backtest_end_dt
                
            # 获取净值
            # 如果 start_dt 在回测首日之前，则视为初始状态 1.0
            if start_dt < backtest_start_dt:
                val_start = 1.0
            else:
                val_start = equity_curve_series.asof(start_dt)
            
            val_end = equity_curve_series.asof(end_dt)
                
            if pd.notna(val_start) and pd.notna(val_end) and val_start != 0:
                ret = val_end / val_start - 1
                period_returns.append(ret)

        if period_returns:
            period_returns_arr = np.array(period_returns)
            wins = period_returns_arr[period_returns_arr > 0]
            losses = period_returns_arr[period_returns_arr <= 0]
            
            win_rate = len(wins) / len(period_returns_arr)
            avg_win = wins.mean() if len(wins) > 0 else 0.0
            avg_loss = abs(losses.mean()) if len(losses) > 0 else 0.0
            odds = avg_win / avg_loss if avg_loss != 0 else np.inf
            
            metrics['Rebalance Win Rate (换仓胜率)'] = win_rate
            metrics['Rebalance Odds (盈亏比)'] = odds

    return pd.Series(metrics)


def calculate_return_attribution(
    weights_df: pd.DataFrame,
    daily_returns_df: pd.DataFrame,
    cost_per_side: float = 0.0
) -> pd.Series:
    """计算各资产对组合总收益的贡献度。
    
    计算公式：Contribution_i = Sum(w_{i,t-1} * r_{i,t})
    注意：这是算术累加贡献，适用于解释“总收益中有多少来自该资产”。
    
    参数:
        weights_df: 权重历史 (index=日期, columns=资产)
        daily_returns_df: 资产日收益率 (index=日期, columns=资产)
        cost_per_side: 单边交易成本率 (默认0.0)
    返回:
        pd.Series: 各资产的累计贡献值，按降序排列
    """
    # 确保为数值型
    weights_df = weights_df.apply(pd.to_numeric, errors='coerce')
    daily_returns_df = daily_returns_df.apply(pd.to_numeric, errors='coerce')

    # 对齐并前填权重
    daily_weights = weights_df.reindex(daily_returns_df.index).ffill()
    # 使用前一日权重（T日收益由T-1日收盘确定的权重决定）
    tradeable_weights = daily_weights.shift(1).dropna(how='all')
    
    # 对齐收益率
    common_idx = tradeable_weights.index.intersection(daily_returns_df.index)
    aligned_weights = tradeable_weights.loc[common_idx]
    aligned_returns = daily_returns_df.loc[common_idx]
    
    # 仅保留共同存在的列
    common_cols = aligned_weights.columns.intersection(aligned_returns.columns)
    aligned_weights = aligned_weights[common_cols]
    aligned_returns = aligned_returns[common_cols]

    # 1. 计算每日算术贡献 (Return on current AUM)
    daily_contribution = aligned_weights * aligned_returns
    
    # 2. 扣除交易成本 (如果存在)
    if cost_per_side > 0:
        # 计算稀疏权重的换手率 (仅在调仓日)
        # 注意：这里保持与 compute_portfolio_returns_and_equity 一致的逻辑
        # 即 diff().abs() 后 shift(1)
        asset_turnover = weights_df.diff().abs()
        asset_cost_sparse = asset_turnover * cost_per_side
        asset_cost_sparse = asset_cost_sparse.shift(1)
        
        # 映射到每日时间轴 (仅在对应日期有成本，其余为0)
        daily_asset_cost = asset_cost_sparse.reindex(daily_contribution.index).fillna(0.0)
        
        # 仅处理共同列
        common_cost_cols = daily_asset_cost.columns.intersection(daily_contribution.columns)
        daily_contribution[common_cost_cols] -= daily_asset_cost[common_cost_cols]

    # 3. 计算组合每日总收益
    portfolio_daily_ret = daily_contribution.sum(axis=1)
    
    # 4. 计算复利修正因子 (Cumulative Return Factor)
    #    T日的美元贡献 = T-1日净值 * T日算术贡献
    #    对总收益率的贡献 = (T-1日净值 / 初始净值) * T日算术贡献
    #    cumulative_factor[t] = Equity[t-1]
    cumulative_factor = (1 + portfolio_daily_ret).cumprod().shift(1).fillna(1.0)
    
    # 5. 调整每日贡献 (Geometric Linking)
    #    使用 multiply 进行广播乘法 (axis=0 表示按行乘)
    adjusted_daily_contribution = daily_contribution.multiply(cumulative_factor, axis=0)
    
    # 6. 累计贡献
    total_contribution = adjusted_daily_contribution.sum().sort_values(ascending=False)
    
    return total_contribution
