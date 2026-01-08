# -*- coding: utf-8 -*-
"""
run_all_weather_bl.py

Run All Weather strategy with and without Black-Litterman adjustment.
"""

import matplotlib.pyplot as plt
from strategy_AllWeather import run_strategy
from strategy_AllWeather_v2 import run_strategy_updated
from strategy_ThreeBasket import run_three_basket_strategy

# 对比的共同参数
internal_method='EW'
start_date='2018-11-30'
end_date='2025-11-30'
rebalance_day= None
use_etf_real_data= False

def main():
    print("Running All Weather Strategy (Baseline)...")
    res_base = run_strategy(
        internal_method=internal_method,
        start_date=start_date,
        end_date=end_date,
        rebalance_day = rebalance_day,
        # 下面是对比参数
        macro_factor_adjustment=False,
        use_monetary_position_sizing=False,  # 启用货币政策仓位调整
        use_etf_real_data= use_etf_real_data


    )
    
    print("\nRunning All Weather Strategy (Macro Factor )...")
    res_factor = run_strategy_updated(
        internal_method=internal_method,
        start_date=start_date,
        end_date=end_date,
        rebalance_day = rebalance_day,
        # 下面是对比参数
        macro_factor_adjustment=False, 
        growth_tilt_strength= 0.1,
        inflation_tilt_strength= 0.1,
        use_monetary_position_sizing=True,  # 启用货币政策仓位调整
        max_position=1.0,  # 货币宽松时满仓
        min_position=0.8,  # 货币紧缩时80%仓位
        use_etf_real_data= use_etf_real_data
    )

    print("\nRunning Three Basket Strategy ...")
    res_three = run_three_basket_strategy(
        start_date=start_date,
        end_date=end_date,
        rebalance_day = rebalance_day,
        # 下面是对比参数
        cov_estimate_method='cov',
    )
    
    # Compare Performance
    perf_base = res_base['performance_report']
    perf_factor = res_factor['performance_report']
    perf_three = res_three['performance_report']
    
    print("\n--- Performance Comparison ---")
    print(f"{'Metric':<35} {'Baseline':<18} {'Macro Tilt':<18} {'Three Basket':<18}")
    print("-" * 89)
    
    metrics_map = {
        'CAGR': 'CAGR (年化复合收益)',
        'Volatility': 'Annual Volatility (年化波动率)',
        'Sharpe Ratio': 'Sharpe Ratio (夏普比率)',
        'Max Drawdown': 'Max Drawdown (最大回撤)',
        'Calmar Ratio': 'Calmar Ratio (Calmar比率)',
        'Sortino Ratio': 'Sortino Ratio (索提诺比率)',
        'Win Rate': 'Rebalance Win Rate (换仓胜率)',
        'Odds': 'Rebalance Odds (盈亏比)',
    }
    
    for label, key in metrics_map.items():
        val_base = perf_base.get(key, 0.0)
        val_factor = perf_factor.get(key, 0.0)
        val_three = perf_three.get(key, 0.0)

        # Format based on metric type
        if 'Ratio' in label or 'Odds' in label:
            fmt = "{:<18.4f}"
        else:
            fmt = "{:<18.2%}"
            
        print(f"{label:<35} " + fmt.format(val_base) + " " + fmt.format(val_factor) + " " + fmt.format(val_three))
    
    # 打印年度收益率对比
    print("\n--- Annual Returns Comparison ---")
    print(f"{'Year':<15} {'Baseline':<18} {'Macro Tilt':<18} {'Three Basket':<18}")
    print("-" * 69)
    
    # 提取所有年度收益率
    years_base = {k: v for k, v in perf_base.items() if 'Annual Return' in k}
    years_factor = {k: v for k, v in perf_factor.items() if 'Annual Return' in k}
    years_three = {k: v for k, v in perf_three.items() if 'Annual Return' in k}

    # 获取所有年份
    all_years = sorted(set(
        [k.split()[2] for k in years_base.keys()] + 
        [k.split()[2] for k in years_factor.keys()] + 
        [k.split()[2] for k in years_three.keys()]
    ))

    
    for year in all_years:
        key = f'Annual Return {year} (年度收益率)'
        val_base = perf_base.get(key, 0.0)
        val_factor = perf_factor.get(key, 0.0)
        val_three = perf_three.get(key, 0.0)

        print(f"{year:<15} {val_base:<18.2%} {val_factor:<18.2%} {val_three:<18.2%}")
        
    # Plot Equity Curves
    df_base = res_base['equity_curve_series']
    df_factor = res_factor['equity_curve_series']
    df_three = res_three['equity_curve_series']
    
    plt.figure(figsize=(12, 6))
    plt.plot(df_base, label='Baseline (Risk Parity)')
    plt.plot(df_factor, label='Macro Factor Adjustment')
    plt.plot(df_three, label='Three Basket Strategy')
    plt.title('All Weather Strategy: Baseline vs Macro Factor Adjustment vs Three Basket Strategy')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
