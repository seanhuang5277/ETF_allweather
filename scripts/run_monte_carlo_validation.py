# -*- coding: utf-8 -*-
"""
@Project: Quantitative Strategy Backtesting Framework
@File   : run_monte_carlo_validation.py
@Author : Copilot
@Date   : 2025-01-09

@Description:
Monte Carlo 稳健性验证脚本 - 用于全天候策略的稳健性测试

运行方式:
    python scripts/run_monte_carlo_validation.py

支持的测试类型:
1. 滚动窗口测试 (Rolling Window)
2. 随机起止日期测试 (Random Date)  
3. 参数敏感性分析 (Parameter Sensitivity)
4. Bootstrap 重采样测试 (Block Bootstrap)
"""

from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

# 设置根目录
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from framework.monte_carlo import (
    StrategyRobustnessValidator,
    MonteCarloSimulator,
    plot_monte_carlo_results,
    plot_equity_curve_fan,
    generate_summary_report,
)
from framework.logging_config import setup_logging, get_logger
from framework.load_data import load_etf_index_returns_by_category

# 导入策略函数
from strategies.strategy_AllWeather import run_strategy

# 设置日志
setup_logging(level="INFO", log_to_file=True, filename="logs/monte_carlo_validation.log")
logger = get_logger("monte_carlo_validation")

# 创建结果保存目录
RESULTS_DIR = ROOT / 'results' / 'monte_carlo'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_base_returns() -> pd.DataFrame:
    """加载基础收益率数据用于 Monte Carlo 模拟"""
    etf_by_cat, idx_by_cat = load_etf_index_returns_by_category(data_dir='data')
    
    # 合并所有类别的指数收益率
    all_returns = []
    for cat, df in idx_by_cat.items():
        all_returns.append(df)
    
    combined_returns = pd.concat(all_returns, axis=1).sort_index()
    combined_returns = combined_returns.dropna(how='all')
    
    logger.info(f"加载收益率数据: {combined_returns.shape[0]} 天, {combined_returns.shape[1]} 资产")
    return combined_returns


def get_base_strategy_params() -> dict:
    """获取策略基础参数"""
    return {
        'data_dir': 'data',
        'internal_method': 'EW',
        'start_date': "2018-11-30",
        'end_date': "2025-11-30",
        'rebalance_day': None,
        'macro_factor_adjustment': False,
        'growth_tilt_strength': 0.1,
        'inflation_tilt_strength': 0.1,
        'use_monetary_position_sizing': False,
        'max_position': 1.0,
        'min_position': 0.8,
        'use_etf_real_data': False,
        'top_cov_estimate_ways': 'cov',
        'bottom_cov_estimate_ways': 'cov',
        'ewm_span_days': 252,
        'use_factor_balance': False,
    }


def run_rolling_window_validation():
    """
    运行滚动窗口测试
    
    在不同的时间窗口上运行策略，检验策略在不同市场环境下的稳定性。
    """
    logger.info("=" * 60)
    logger.info("开始滚动窗口测试 (Rolling Window Test)")
    logger.info("=" * 60)
    
    returns_df = load_base_returns()
    base_params = get_base_strategy_params()
    
    validator = StrategyRobustnessValidator(
        strategy_func=run_strategy,
        base_params=base_params,
        returns_df=returns_df,
        random_seed=42,
    )
    
    result = validator.run_rolling_window_test(
        window_years=3,
        step_months=6,
        min_test_years=2,
    )
    
    # 生成报告
    report = generate_summary_report(result)
    print(report)
    
    # 保存报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"rolling_window_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    logger.info(f"报告已保存: {report_path}")
    
    # 保存详细结果
    result.metrics_df.to_csv(RESULTS_DIR / f"rolling_window_metrics_{timestamp}.csv", index=False)
    
    # 绘制结果
    fig1 = plot_monte_carlo_results(
        result,
        save_path=str(RESULTS_DIR / f"rolling_window_hist_{timestamp}.png")
    )
    
    if result.equity_curves:
        fig2 = plot_equity_curve_fan(
            result,
            save_path=str(RESULTS_DIR / f"rolling_window_fan_{timestamp}.png")
        )
    
    return result


def run_random_date_validation():
    """
    运行随机起止日期测试
    
    随机选择不同的回测起止日期，检验策略对样本选择的敏感性。
    """
    logger.info("=" * 60)
    logger.info("开始随机日期测试 (Random Date Test)")
    logger.info("=" * 60)
    
    returns_df = load_base_returns()
    base_params = get_base_strategy_params()
    
    validator = StrategyRobustnessValidator(
        strategy_func=run_strategy,
        base_params=base_params,
        returns_df=returns_df,
        random_seed=42,
    )
    
    result = validator.run_random_date_test(
        n_simulations=50,  # 可调整模拟次数
        min_years=2,
        max_years=5,
    )
    
    # 生成报告
    report = generate_summary_report(result)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"random_date_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    result.metrics_df.to_csv(RESULTS_DIR / f"random_date_metrics_{timestamp}.csv", index=False)
    
    # 绘制结果
    fig1 = plot_monte_carlo_results(
        result,
        save_path=str(RESULTS_DIR / f"random_date_hist_{timestamp}.png")
    )
    
    if result.equity_curves:
        fig2 = plot_equity_curve_fan(
            result,
            save_path=str(RESULTS_DIR / f"random_date_fan_{timestamp}.png")
        )
    
    return result


def run_parameter_sensitivity_validation():
    """
    运行参数敏感性分析
    
    测试不同参数组合下的策略表现，识别关键参数和最优参数区间。
    """
    logger.info("=" * 60)
    logger.info("开始参数敏感性分析 (Parameter Sensitivity Analysis)")
    logger.info("=" * 60)
    
    returns_df = load_base_returns()
    base_params = get_base_strategy_params()
    
    validator = StrategyRobustnessValidator(
        strategy_func=run_strategy,
        base_params=base_params,
        returns_df=returns_df,
        random_seed=42,
    )
    
    # 定义参数网格
    param_grid = {
        'internal_method': ['EW', 'HRP'],
        'top_cov_estimate_ways': ['cov', 'ewm', 'oas'],
        'lookback_years': [2, 3, 4],
    }
    
    result = validator.run_parameter_sensitivity(param_grid)
    
    # 生成报告
    report = generate_summary_report(result)
    print(report)
    
    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = RESULTS_DIR / f"param_sensitivity_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    # 保存详细的参数-绩效对照表
    result.metrics_df.to_csv(RESULTS_DIR / f"param_sensitivity_metrics_{timestamp}.csv", index=False)
    
    # 绘制结果
    fig1 = plot_monte_carlo_results(
        result,
        save_path=str(RESULTS_DIR / f"param_sensitivity_hist_{timestamp}.png")
    )
    
    # 绘制参数影响热力图
    plot_parameter_heatmap(result, timestamp)
    
    return result


def plot_parameter_heatmap(result, timestamp):
    """绘制参数敏感性热力图"""
    df = result.metrics_df.copy()
    
    # 检查是否有足够的参数列
    param_cols = ['internal_method', 'top_cov_estimate_ways', 'lookback_years']
    param_cols = [c for c in param_cols if c in df.columns]
    
    if len(param_cols) < 2:
        logger.warning("参数列不足，无法绘制热力图")
        return
    
    # 选择关键指标
    metric = 'Sharpe Ratio (夏普比率)'
    if metric not in df.columns:
        metric = df.select_dtypes(include=[np.number]).columns[0]
    
    fig, axes = plt.subplots(1, len(param_cols) - 1, figsize=(14, 5))
    if len(param_cols) == 2:
        axes = [axes]
    
    for i, p1 in enumerate(param_cols[:-1]):
        p2 = param_cols[i + 1]
        
        # 透视表
        try:
            pivot = df.pivot_table(values=metric, index=p1, columns=p2, aggfunc='mean')
            
            ax = axes[i]
            im = ax.imshow(pivot.values, cmap='RdYlGn', aspect='auto')
            
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
            ax.set_yticklabels(pivot.index)
            
            ax.set_xlabel(p2)
            ax.set_ylabel(p1)
            ax.set_title(f'{metric}\n{p1} vs {p2}')
            
            # 添加数值标签
            for ii in range(len(pivot.index)):
                for jj in range(len(pivot.columns)):
                    val = pivot.values[ii, jj]
                    if not np.isnan(val):
                        ax.text(jj, ii, f'{val:.3f}', ha='center', va='center', fontsize=9)
            
            plt.colorbar(im, ax=ax)
            
        except Exception as e:
            logger.warning(f"绘制热力图失败 ({p1} vs {p2}): {e}")
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"param_heatmap_{timestamp}.png", dpi=150, bbox_inches='tight')
    plt.close()


def run_comprehensive_validation():
    """
    运行全面的稳健性验证
    
    包含所有验证方法，生成综合报告。
    """
    logger.info("=" * 60)
    logger.info("开始全面稳健性验证 (Comprehensive Validation)")
    logger.info("=" * 60)
    
    results = {}
    
    # 1. 滚动窗口测试
    try:
        results['rolling_window'] = run_rolling_window_validation()
    except Exception as e:
        logger.error(f"滚动窗口测试失败: {e}")
    
    # 2. 随机日期测试
    try:
        results['random_date'] = run_random_date_validation()
    except Exception as e:
        logger.error(f"随机日期测试失败: {e}")
    
    # 3. 参数敏感性分析
    try:
        results['param_sensitivity'] = run_parameter_sensitivity_validation()
    except Exception as e:
        logger.error(f"参数敏感性分析失败: {e}")
    
    # 生成综合报告
    generate_comprehensive_report(results)
    
    logger.info("=" * 60)
    logger.info("全面稳健性验证完成!")
    logger.info(f"结果保存目录: {RESULTS_DIR}")
    logger.info("=" * 60)
    
    return results


def generate_comprehensive_report(results: dict):
    """生成综合验证报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    lines = [
        "=" * 80,
        "全天候策略 (All Weather Strategy) Monte Carlo 稳健性验证报告",
        "=" * 80,
        f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
    ]
    
    for test_name, result in results.items():
        if result is not None:
            lines.append("")
            lines.append(generate_summary_report(result))
            lines.append("")
    
    # 综合评估
    lines.append("=" * 80)
    lines.append("综合评估")
    lines.append("=" * 80)
    
    # 计算各测试的 Sharpe 比率分布
    sharpe_col = 'Sharpe Ratio (夏普比率)'
    for test_name, result in results.items():
        if result is not None and sharpe_col in result.metrics_df.columns:
            sharpe_data = result.metrics_df[sharpe_col].dropna()
            lines.append(f"\n{test_name} - Sharpe Ratio 分布:")
            lines.append(f"  正值比例: {(sharpe_data > 0).mean():.1%}")
            lines.append(f"  > 0.5 比例: {(sharpe_data > 0.5).mean():.1%}")
            lines.append(f"  > 1.0 比例: {(sharpe_data > 1.0).mean():.1%}")
    
    lines.append("")
    lines.append("=" * 80)
    
    report_text = "\n".join(lines)
    
    # 保存报告
    report_path = RESULTS_DIR / f"comprehensive_report_{timestamp}.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    logger.info(f"综合报告已保存: {report_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Monte Carlo 稳健性验证')
    parser.add_argument(
        '--test', '-t',
        choices=['rolling', 'random', 'param', 'all'],
        default='all',
        help='选择测试类型: rolling(滚动窗口), random(随机日期), param(参数敏感性), all(全部)'
    )
    parser.add_argument(
        '--n_sims', '-n',
        type=int,
        default=50,
        help='模拟次数 (仅适用于 random 测试)'
    )
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print("Monte Carlo 稳健性验证")
    print(f"测试类型: {args.test}")
    print(f"{'='*60}\n")
    
    if args.test == 'rolling':
        run_rolling_window_validation()
    elif args.test == 'random':
        run_random_date_validation()
    elif args.test == 'param':
        run_parameter_sensitivity_validation()
    else:  # all
        run_comprehensive_validation()
    
    plt.show()
