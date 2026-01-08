# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : strategy_ThreeBasket.py
@Author : Copilot
@Date   : 2025-12-03

@Description:
三篮子宏观对冲策略 (Three-Basket Macro Hedge Strategy)
基于 "Risk Budgeting" (风险预算) 和 "Global Macro" (全球宏观) 理念重构。

核心逻辑：
1. 将资产划分为三个宏观风险篮子，解决"黄金看美国，股票看中国"的错配问题。
   - 篮子1 (CN_Growth): 中国增长风险 (股票 + 内盘商品)
   - 篮子2 (CN_Deflation): 中国利率/通缩风险 (债券)
   - 篮子3 (US_Hedge): 美元信用/全球通胀风险 (黄金 + 全球商品)

2. 权重分配机制：
   - 篮子内部 (Bottom Level): 固定权重 (Fixed Weights)。
     例如：CN_Growth 内部 70% 股票 + 30% 商品。
   - 篮子之间 (Top Level): 风险预算 (Risk Budgeting)。
     不追求风险平价(1:1:1)，而是主动分配风险预算 (e.g., 60% : 10% : 30%)，
     避免债券因低波动而过度配资，同时保证权益资产的进攻性。

3. 宏观动态调整 (Macro Tilt):
   - 支持根据宏观信号动态调整顶层风险预算。
"""

import os
import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

# 添加项目根目录到 path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from framework.config import (
    TRADING_DAYS_PER_YEAR_SCALAR,
    COST_PER_SIDE_SCALAR,
    RISK_FREE_RATE_SCALAR_ANNUAL,
    DEFAULT_LOOKBACK_YEARS,
    DEFAULT_MIN_DATA_YEARS,
    MIN_CLEAN_DAYS,
)
from framework.load_data import load_etf_index_returns_by_category, compute_log_returns
from framework.performance import (
    calculate_performance_metrics,
    compute_portfolio_returns_and_equity,
    calculate_return_attribution
)
from framework.allocation_utils import (
    to_daily_rebalance_dates,
    solve_risk_parity_weights,
    estimate_covariance_matrix,
)
from framework.logging_config import setup_logging, get_logger
from framework.plotting import plot_equity_curve, plot_weights_history, plot_return_attribution, plot_multi_equity_curves

# 初始化日志
logger = get_logger("strategy.ThreeBasket")
STRATEGY_MODE = "Three-Basket Risk Budgeting"

# --------------------------------------------------------------------------
# 配置区域
# --------------------------------------------------------------------------

# 资产分类定义
# 注意：这里使用 index_daily_simple_returns.csv 中的列名
ASSET_GROUPS = {
    'Stock': [
        '沪深300指数', 
        #'上证50指数', 
        '中证小盘500指数', 
        #'创业板指数', 
        '中证1000指数', 
       # '上证科创板50成份指数', 
       '标普中国A股大盘红利低波50指数', 
       '中证红利质量指数', 
        #'中证全指证券公司指数', '中证主要消费指数', '中证全指半导体产品与设备指数', 
        #'中证银行指数', '中证军工指数', '沪深300医药卫生指数', '中证全指房地产指数', 
        #'中证全指电力与电网指数', '国证半导体芯片', '中证机器人指数', '中证人工智能主题指数', 
        #'中证细分化工产业主题指数', '沪深300非银行金融指数', '国证新能源车电池指数', 
        #'中证光伏产业指数', '中证创新药产业指数', '中证全指通信设备指数', 
        #'中证动漫游戏指数', '中证金融科技主题指数'
    ],
    'Commodity_CN': [
        '中证煤炭指数', '大连商品交易所豆粕期货价格指数', '易盛能化A', '中证申万有色金属指数'
    ],
    'Bond': [
        '上证5年期国债指数', '上证10年期国债指数(净价)', '中债-30年期国债指数', 
        '上证城投债指数', '中债-中高等级公司债利差因子指数', '上证基准做市公司债指数', 
        '中证可转债及可交换债券指数'
    ],
    'Gold': ['SGE黄金999'],
    'Commodity_Global': ['上海期货交易所有色金属期货价格指数']
}

# 篮子定义与配置
BASKET_CONFIG = {
    'CN_Growth': {
        'default_risk_budget': 0.35,  # 默认风险预算 60%
        'sub_groups': {
            'Stock': 0.7,           # 内部权重 70%
            'Commodity_CN': 0.3     # 内部权重 30%
        }
    },
    'CN_Deflation': {
        'default_risk_budget': 0.35,  # 默认风险预算 10%
        'sub_groups': {
            'Bond': 1.0             # 内部权重 100%
        }
    },
    'US_Hedge': {
        'default_risk_budget': 0.3,  # 默认风险预算 30%
        'sub_groups': {
            'Gold': 0.6,            # 内部权重 60%
            'Commodity_Global': 0.4 # 内部权重 40%
        }
    }
}

def run_three_basket_strategy(
    *,
    data_dir: str = 'data',
    start_date: Optional[str | pd.Timestamp] = None,
    end_date: Optional[str | pd.Timestamp] = None,
    lookback_years: int = DEFAULT_LOOKBACK_YEARS,
    min_data_years: int = DEFAULT_MIN_DATA_YEARS,
    min_clean_days: int = MIN_CLEAN_DAYS,
    
    # 协方差估计参数
    cov_estimate_method: str = 'ewm',
    ewm_span_days: int = 252,
    
    # 换仓日参数
    rebalance_day: int = None,

    # 成本与绩效参数
    cost_per_side: float = COST_PER_SIDE_SCALAR,
    rf_rate: float = RISK_FREE_RATE_SCALAR_ANNUAL,
    use_etf_real_data: bool = True, # 本策略主要基于指数回测，ETF映射逻辑暂略
    
    # 宏观动态调整
    use_macro_tilt: bool = False,
) -> Dict[str, object]:
    
    # --------------------------------------------------------------------------
    # 1. 数据准备
    # --------------------------------------------------------------------------
    start_dt_global = pd.to_datetime(start_date)
    end_dt_global = pd.to_datetime(end_date)
    
    # 加载指数数据
    idx_path = os.path.join(data_dir, 'index_daily_simple_returns.csv')
    if not os.path.exists(idx_path):
        logger.error(f"数据文件不存在: {idx_path}")
        return {}
        
    all_idx_simp_df = pd.read_csv(idx_path, index_col=0, parse_dates=True)
    
    # 简单的ETF映射（如果需要）：这里假设 ETF 列名与指数列名一致，或者直接用指数回测
    # 为了兼容 plotting 模块，我们需要一个 ETF 收益表
    all_etf_simp_df = all_idx_simp_df.copy() 
    
    # 现金处理 (使用短融指数作为现金替代)
    cash_asset = '中证短融指数'
    if cash_asset not in all_idx_simp_df.columns:
        logger.warning(f"未找到现金资产 {cash_asset}，将使用 0 收益代替")
        cash_returns = pd.Series(0.0, index=all_idx_simp_df.index)
    else:
        cash_returns = all_idx_simp_df[cash_asset]

    # 确定再平衡日期
    min_data_days_required = TRADING_DAYS_PER_YEAR_SCALAR * min_data_years
    rebalance_dates = to_daily_rebalance_dates(all_idx_simp_df, min_data_days_required, rebalance_day=rebalance_day)
    
    # 时间过滤
    if start_dt_global is not None:
        rebalance_dates = rebalance_dates[rebalance_dates >= start_dt_global]
    if end_dt_global is not None:
        rebalance_dates = rebalance_dates[rebalance_dates <= end_dt_global]
        
    lookback_window = TRADING_DAYS_PER_YEAR_SCALAR * lookback_years
    
    # --------------------------------------------------------------------------
    # 2. 回测循环
    # --------------------------------------------------------------------------
    
    # 存储结果
    basket_weights_hist = pd.DataFrame(index=rebalance_dates, columns=BASKET_CONFIG.keys(), dtype=float).fillna(0.0)
    atomic_weights_hist = pd.DataFrame(index=rebalance_dates, columns=all_etf_simp_df.columns, dtype=float).fillna(0.0)
    
    # 存储篮子净值曲线（用于绘图）
    basket_equity_curves = {}
    basket_returns_series = {b: [] for b in BASKET_CONFIG.keys()}
    
    logger.info(f"开始回测: {start_date} -> {end_date}, 调仓次数: {len(rebalance_dates)}")
    
    for dt in rebalance_dates:
        # 数据窗口
        start_dt = dt - pd.DateOffset(days=lookback_window)
        window_simp_df = all_idx_simp_df.loc[start_dt:dt]
        
        if len(window_simp_df) < min_clean_days:
            continue
            
        # --- Step 1: 计算各篮子的合成历史收益率 (用于计算协方差) ---
        basket_history_returns = {}
        
        # 临时存储本期各篮子的原子权重构成 (Basket -> {Asset: Weight})
        basket_internal_composition = {} 
        
        for basket_name, config in BASKET_CONFIG.items():
            sub_groups = config['sub_groups']
            
            # 篮子合成收益序列
            basket_ret_series = pd.Series(0.0, index=window_simp_df.index)
            basket_composition = pd.Series(0.0, index=all_etf_simp_df.columns)
            
            valid_basket = True
            
            for group_name, group_weight in sub_groups.items():
                # 获取该组下的资产列表
                assets = [a for a in ASSET_GROUPS.get(group_name, []) if a in window_simp_df.columns]
                
                if not assets:
                    logger.warning(f"[{dt.date()}] 篮子 {basket_name} 的子组 {group_name} 没有有效资产")
                    continue
                
                # 组内资产等权分配 (Equal Weight within Sub-group)
                # 也可以改为波动率倒数等，这里按"固定权重"理念保持简单
                n_assets = len(assets)
                w_i = 1.0 / n_assets
                
                # 计算组收益: sum(w_i * R_i)
                group_ret = window_simp_df[assets].mean(axis=1) # 等权平均
                
                # 累加到篮子收益: Basket_Ret += Group_Weight * Group_Ret
                basket_ret_series = basket_ret_series.add(group_ret * group_weight, fill_value=0.0)
                
                # 记录构成权重
                for asset in assets:
                    basket_composition[asset] += group_weight * w_i
            
            # 检查数据有效性
            if basket_ret_series.count() < min_clean_days:
                valid_basket = False
            
            if valid_basket:
                basket_history_returns[basket_name] = basket_ret_series
                basket_internal_composition[basket_name] = basket_composition
        
        # --- Step 2: 顶层风险预算分配 (Risk Budgeting) ---
        valid_baskets = list(basket_history_returns.keys())
        if not valid_baskets:
            continue
            
        # 合并篮子收益
        basket_ret_df = pd.concat([basket_history_returns[b] for b in valid_baskets], axis=1)
        basket_ret_df.columns = valid_baskets
        
        # 转对数收益计算协方差
        basket_log_ret = compute_log_returns(basket_ret_df)
        cov_mat = estimate_covariance_matrix(basket_log_ret, method=cov_estimate_method, ewm_span=ewm_span_days)
        
        # 获取目标风险预算
        # TODO: 这里可以加入宏观动态调整逻辑 (Macro Tilt)
        target_budget = np.array([BASKET_CONFIG[b]['default_risk_budget'] for b in valid_baskets])
        # 归一化预算
        target_budget = target_budget / target_budget.sum()
        
        # 转换为 Series 以匹配 solve_risk_parity_weights 的要求
        target_budget_series = pd.Series(target_budget, index=valid_baskets)
        
        # 求解风险预算权重
        w_top = solve_risk_parity_weights(cov_mat, target_budget_vector=target_budget_series)
        w_top = w_top / w_top.sum()
        
        # 记录顶层权重
        basket_weights_hist.loc[dt, valid_baskets] = w_top.values
        
        # --- Step 3: 穿透计算底层权重 ---
        current_atomic_w = pd.Series(0.0, index=all_etf_simp_df.columns)
        
        for b_name in valid_baskets:
            b_weight = w_top[b_name]
            b_composition = basket_internal_composition[b_name]
            
            # 穿透权重 = 篮子权重 * 篮子内资产权重
            current_atomic_w = current_atomic_w.add(b_composition * b_weight, fill_value=0.0)
            
        # 归一化
        if current_atomic_w.sum() > 0:
            current_atomic_w = current_atomic_w / current_atomic_w.sum()
            
        atomic_weights_hist.loc[dt] = current_atomic_w
        
        # Log
        w_str = ", ".join([f"{k}:{v:.1%}" for k, v in w_top.items()])
        logger.debug(f"[{dt.date()}] Top Weights: {w_str}")

    # --------------------------------------------------------------------------
    # 3. 结果计算与展示
    # --------------------------------------------------------------------------
    
    # 计算组合净值
    _, port_ret, _, equity_curve, daily_w = compute_portfolio_returns_and_equity(
        weights_history_df=atomic_weights_hist,
        daily_returns_df=all_etf_simp_df,
        cost_per_side_scalar=cost_per_side
    )
    
    # 绩效指标
    report = calculate_performance_metrics(
        port_ret, equity_curve, rf_rate, TRADING_DAYS_PER_YEAR_SCALAR,
        rebalance_dates=rebalance_dates
    )
    
    # 计算各篮子的净值曲线 (用于展示)
    for b_name in BASKET_CONFIG.keys():
        # 提取该篮子的历史权重 (Top Weight * Internal Composition)
        # 这里为了展示"如果只持有该篮子"的表现，我们需要重新构建权重序列
        # 简化处理：我们假设篮子内部权重是固定的（虽然实际上随资产有效性变化），
        # 我们直接用回测中计算出的 basket_internal_composition (取最后一期近似，或者重新计算)
        # 更准确的方法：在回测循环中记录每个篮子的日收益
        pass 
        # 由于上面循环中没有记录日收益，这里暂略，或者后续补充
    
    # 调仓收益
    valid_rebalance_dates = rebalance_dates[rebalance_dates >= equity_curve.index[0]]
    rebalance_returns = equity_curve.loc[valid_rebalance_dates].pct_change().dropna()
    
    # 归因
    attribution = calculate_return_attribution(
        weights_df=atomic_weights_hist,
        daily_returns_df=all_etf_simp_df,
        cost_per_side=cost_per_side
    )

    return {
        'performance_report': report,
        'equity_curve_series': equity_curve,
        'basket_weights_df': basket_weights_hist,
        'daily_weights_df': daily_w,
        'atomic_attribution': attribution,
        'rebalance_returns': rebalance_returns,
    }

if __name__ == "__main__":
    # 运行策略
    res = run_three_basket_strategy(
        start_date="2018-11-30",
        end_date="2025-11-30",
        cov_estimate_method='cov',
        ewm_span_days=252, # 半年半衰期，反应稍快
    )
    
    # 打印报告
    perf = res['performance_report']
    logger.info("=== 三篮子风险预算策略回测结果 ===")
    for k, v in perf.items():
        if isinstance(v, float):
            logger.info(f"  {k}: {v:.2%}" if 'Ratio' not in k else f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
            
    # 绘图
    import matplotlib.pyplot as plt
    
    plot_equity_curve(res['equity_curve_series'], perf, STRATEGY_MODE, 
                     rebalance_returns=res['rebalance_returns'], auto_show=False)
    
    plot_weights_history(res['basket_weights_df'], "Basket Allocation (Risk Budgeting)", reverse_legend=True, auto_show=False)
    
    plot_return_attribution(res['atomic_attribution'], title="Asset Attribution", auto_show=False)
    
    plt.show()
