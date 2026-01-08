# -*- coding: utf-8 -*-
"""
Monte Carlo 验证使用示例 - 简化版

这个脚本展示了如何快速使用 Monte Carlo 模块验证策略稳健性。
可以直接复制到 Jupyter Notebook 中运行。
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

# =============================================================================
# 示例1: 滚动窗口测试 - 最简单的稳健性检验
# =============================================================================
def example_rolling_window():
    """
    滚动窗口测试示例
    
    在不同的时间窗口上运行策略，检验策略是否在不同市场环境下都能表现良好。
    这是最基础也是最重要的稳健性检验方法。
    """
    from framework.monte_carlo import StrategyRobustnessValidator, plot_monte_carlo_results, generate_summary_report
    from framework.load_data import load_etf_index_returns_by_category
    from strategies.strategy_AllWeather_v5 import run_strategy_updated
    
    print("=" * 60)
    print("示例1: 滚动窗口测试")
    print("=" * 60)
    
    # 1. 加载收益率数据
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir='data')
    returns_df = pd.concat(list(idx_by_cat.values()), axis=1).dropna(how='all')
    
    # 2. 定义策略参数
    base_params = {
        'data_dir': 'data',
        'internal_method': 'EW',
        'start_date': "2018-11-30",
        'end_date': "2025-11-30",
        'use_etf_real_data': False,
        'macro_factor_adjustment': False,
        'use_monetary_position_sizing': False,
        'use_factor_balance': False,
    }
    
    # 3. 创建验证器
    validator = StrategyRobustnessValidator(
        strategy_func=run_strategy_updated,
        base_params=base_params,
        returns_df=returns_df,
        random_seed=42,
    )
    
    # 4. 运行滚动窗口测试
    # 每3年窗口，每6个月滑动一次
    result = validator.run_rolling_window_test(
        window_years=3,
        step_months=6,
        min_test_years=2,
    )
    
    # 5. 查看结果
    print(generate_summary_report(result))
    
    # 6. 绘制结果分布
    fig = plot_monte_carlo_results(result)
    plt.show()
    
    return result


# =============================================================================
# 示例2: 参数敏感性分析
# =============================================================================
def example_parameter_sensitivity():
    """
    参数敏感性分析示例
    
    测试不同参数组合下的策略表现，找出最稳健的参数区间。
    """
    from framework.monte_carlo import StrategyRobustnessValidator, plot_monte_carlo_results, generate_summary_report
    from framework.load_data import load_etf_index_returns_by_category
    from strategies.strategy_AllWeather_v5 import run_strategy_updated
    
    print("=" * 60)
    print("示例2: 参数敏感性分析")
    print("=" * 60)
    
    # 加载数据
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir='data')
    returns_df = pd.concat(list(idx_by_cat.values()), axis=1).dropna(how='all')
    
    base_params = {
        'data_dir': 'data',
        'start_date': "2018-11-30",
        'end_date': "2025-11-30",
        'use_etf_real_data': False,
        'macro_factor_adjustment': False,
        'use_monetary_position_sizing': False,
    }
    
    validator = StrategyRobustnessValidator(
        strategy_func=run_strategy_updated,
        base_params=base_params,
        returns_df=returns_df,
    )
    
    # 定义要测试的参数组合
    param_grid = {
        'internal_method': ['EW', 'HRP'],  # 象限内分配方法
        'top_cov_estimate_ways': ['cov', 'ewm'],  # 协方差估计方法
    }
    
    result = validator.run_parameter_sensitivity(param_grid)
    
    print(generate_summary_report(result))
    
    # 打印详细的参数-绩效对照表
    print("\n参数组合详细结果:")
    cols_to_show = ['internal_method', 'top_cov_estimate_ways', 
                    'CAGR (年化复合收益)', 'Sharpe Ratio (夏普比率)', 
                    'Max Drawdown (最大回撤)']
    cols_to_show = [c for c in cols_to_show if c in result.metrics_df.columns]
    print(result.metrics_df[cols_to_show].to_string())
    
    fig = plot_monte_carlo_results(result)
    plt.show()
    
    return result


# =============================================================================
# 示例3: 快速 Bootstrap 检验 (使用模拟数据)
# =============================================================================
def example_bootstrap_simulation():
    """
    Bootstrap 重采样模拟示例
    
    使用 Block Bootstrap 对历史数据进行重采样，生成多条模拟路径，
    检验策略在不同市场路径下的表现分布。
    
    注意：此示例展示 MonteCarloSimulator 的直接使用方法。
    """
    from framework.monte_carlo import MonteCarloSimulator
    from framework.load_data import load_etf_index_returns_by_category
    
    print("=" * 60)
    print("示例3: Bootstrap 数据生成")
    print("=" * 60)
    
    # 加载数据
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir='data')
    returns_df = pd.concat(list(idx_by_cat.values()), axis=1).dropna(how='all')
    
    # 创建模拟器
    simulator = MonteCarloSimulator(returns_df, random_seed=42)
    
    # 生成 Block Bootstrap 样本
    print(f"\n原始数据: {returns_df.shape[0]} 天, {returns_df.shape[1]} 资产")
    
    # 生成5个 Bootstrap 样本，展示数据分布
    n_samples = 5
    bootstrap_stats = []
    
    for i in range(n_samples):
        np.random.seed(42 + i)
        bootstrap_df = simulator.generate_block_bootstrap_sample(
            block_size=21,  # 1个月的区块
        )
        
        # 计算各资产的年化收益率
        annual_returns = bootstrap_df.mean() * 252
        stats = {
            'sample_id': i,
            'mean_annual_return': annual_returns.mean(),
            'std_annual_return': annual_returns.std(),
        }
        bootstrap_stats.append(stats)
        print(f"  样本 {i}: 平均年化收益 = {stats['mean_annual_return']:.2%}")
    
    # 对比原始数据
    original_annual = returns_df.mean() * 252
    print(f"\n原始数据: 平均年化收益 = {original_annual.mean():.2%}")
    
    # 绘制对比图
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 左图：原始数据收益分布
    ax1 = axes[0]
    ax1.hist(original_annual, bins=20, edgecolor='white', alpha=0.7)
    ax1.axvline(original_annual.mean(), color='red', linestyle='--', label=f'Mean: {original_annual.mean():.2%}')
    ax1.set_title('Original Data - Asset Annual Returns')
    ax1.set_xlabel('Annual Return')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # 右图：Bootstrap 样本对比
    ax2 = axes[1]
    stats_df = pd.DataFrame(bootstrap_stats)
    ax2.bar(stats_df['sample_id'], stats_df['mean_annual_return'])
    ax2.axhline(original_annual.mean(), color='red', linestyle='--', label='Original Mean')
    ax2.set_title('Bootstrap Samples - Mean Annual Return')
    ax2.set_xlabel('Sample ID')
    ax2.set_ylabel('Mean Annual Return')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
    
    return simulator


# =============================================================================
# 示例4: 协方差矩阵扰动测试
# =============================================================================
def example_covariance_perturbation():
    """
    协方差矩阵扰动测试示例
    
    测试策略对协方差矩阵估计误差的敏感性。
    这对于风险平价类策略尤为重要，因为权重直接依赖于协方差估计。
    """
    from framework.monte_carlo import MonteCarloSimulator
    from framework.allocation_utils import solve_risk_parity_weights
    from framework.load_data import load_etf_index_returns_by_category
    
    print("=" * 60)
    print("示例4: 协方差扰动对风险平价权重的影响")
    print("=" * 60)
    
    # 加载数据
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir='data')
    returns_df = pd.concat(list(idx_by_cat.values()), axis=1).dropna(how='all')
    
    # 只取部分资产便于展示
    selected_assets = returns_df.columns[:6]
    returns_subset = returns_df[selected_assets].dropna()
    
    # 计算原始协方差
    original_cov = returns_subset.cov()
    
    # 计算原始风险平价权重
    original_weights = solve_risk_parity_weights(original_cov)
    print(f"\n原始风险平价权重:")
    for asset, w in original_weights.items():
        print(f"  {asset}: {w:.2%}")
    
    # 创建模拟器并生成扰动的协方差矩阵
    simulator = MonteCarloSimulator(returns_subset, random_seed=42)
    
    # 不同扰动强度下的权重变化
    noise_levels = [0.05, 0.1, 0.2, 0.3]
    perturbed_weights_list = []
    
    print(f"\n不同扰动强度下的权重变化:")
    for noise in noise_levels:
        weights_at_noise = []
        for i in range(10):  # 每个扰动强度做10次
            np.random.seed(42 + i)
            perturbed_cov = simulator.perturb_covariance_matrix(
                original_cov, 
                noise_scale=noise,
                method='multiplicative'
            )
            w = solve_risk_parity_weights(perturbed_cov)
            weights_at_noise.append(w)
        
        # 计算权重的标准差
        weights_df = pd.DataFrame(weights_at_noise)
        weights_std = weights_df.std()
        print(f"\n  噪声强度 {noise:.0%}:")
        print(f"    权重标准差: {weights_std.mean():.2%} (平均)")
    
    # 绘制权重分布
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    for idx, noise in enumerate([0, 0.1, 0.2]):
        ax = axes[idx]
        if noise == 0:
            weights_data = [original_weights.values]
            title = 'Original'
        else:
            weights_data = []
            for i in range(50):
                np.random.seed(42 + i)
                perturbed_cov = simulator.perturb_covariance_matrix(
                    original_cov, noise_scale=noise, method='multiplicative'
                )
                w = solve_risk_parity_weights(perturbed_cov)
                weights_data.append(w.values)
            title = f'Noise = {noise:.0%}'
        
        weights_matrix = np.array(weights_data)
        positions = np.arange(len(selected_assets))
        
        ax.boxplot(weights_matrix, positions=positions, labels=selected_assets)
        ax.set_title(title)
        ax.set_ylabel('Weight')
        ax.tick_params(axis='x', rotation=45)
    
    # 下方三图：展示权重在不同噪声下的稳定性
    for idx, (i, asset) in enumerate(list(enumerate(selected_assets))[:3]):
        ax = axes[3 + idx]
        
        stability_data = []
        for noise in np.linspace(0, 0.3, 10):
            w_samples = []
            for j in range(30):
                np.random.seed(42 + j)
                if noise == 0:
                    w = original_weights[asset]
                else:
                    perturbed_cov = simulator.perturb_covariance_matrix(
                        original_cov, noise_scale=noise, method='multiplicative'
                    )
                    w = solve_risk_parity_weights(perturbed_cov)[asset]
                w_samples.append(w)
            stability_data.append({
                'noise': noise,
                'mean': np.mean(w_samples),
                'std': np.std(w_samples)
            })
        
        stab_df = pd.DataFrame(stability_data)
        ax.fill_between(stab_df['noise'], 
                       stab_df['mean'] - 2*stab_df['std'],
                       stab_df['mean'] + 2*stab_df['std'],
                       alpha=0.3)
        ax.plot(stab_df['noise'], stab_df['mean'], 'b-', linewidth=2)
        ax.axhline(original_weights[asset], color='red', linestyle='--', alpha=0.7)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Weight')
        ax.set_title(f'{asset} Weight Stability')
    
    plt.tight_layout()
    plt.show()
    
    return original_cov, original_weights


# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("Monte Carlo 稳健性验证 - 使用示例")
    print("=" * 80)
    print("\n选择要运行的示例:")
    print("  1. 滚动窗口测试 (推荐首先运行)")
    print("  2. 参数敏感性分析")
    print("  3. Bootstrap 数据生成演示")
    print("  4. 协方差扰动对权重的影响")
    print("  0. 运行全部示例")
    
    choice = input("\n请输入选择 (0-4): ").strip()
    
    if choice == '1':
        example_rolling_window()
    elif choice == '2':
        example_parameter_sensitivity()
    elif choice == '3':
        example_bootstrap_simulation()
    elif choice == '4':
        example_covariance_perturbation()
    elif choice == '0':
        print("\n运行全部示例...\n")
        example_rolling_window()
        example_parameter_sensitivity() 
        example_bootstrap_simulation()
        example_covariance_perturbation()
    else:
        print("无效选择，退出。")
