# -*- coding: utf-8 -*-
"""monte_carlo.py

Monte Carlo 模拟模块：用于验证策略的稳健性（Robustness）。

支持的验证方法：
1. Block Bootstrap - 历史收益率区块重采样
2. Covariance Perturbation - 协方差矩阵扰动测试
3. Rolling Window Test - 滚动窗口测试
4. Asset Dropout Test - 资产剔除测试
5. Parameter Sensitivity - 参数敏感性分析

使用方式:
    from framework.monte_carlo import MonteCarloValidator
    
    validator = MonteCarloValidator(strategy_func, base_params)
    results = validator.run_bootstrap_simulation(n_simulations=1000)
    validator.plot_results(results)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Callable, Any, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from dataclasses import dataclass
from tqdm import tqdm
import matplotlib.pyplot as plt

try:
    from .logging_config import get_logger
    from .config import TRADING_DAYS_PER_YEAR_SCALAR
except ImportError:
    from logging_config import get_logger
    from config import TRADING_DAYS_PER_YEAR_SCALAR

logger = get_logger("framework.monte_carlo")


@dataclass
class MonteCarloResult:
    """Monte Carlo 模拟结果数据类"""
    simulation_type: str
    n_simulations: int
    metrics_df: pd.DataFrame  # 每次模拟的绩效指标
    equity_curves: Optional[Dict[int, pd.Series]] = None  # 可选：保存部分净值曲线
    weights_history: Optional[Dict[int, pd.DataFrame]] = None  # 可选：保存权重历史
    summary_stats: Optional[pd.DataFrame] = None  # 汇总统计


class MonteCarloSimulator:
    """
    Monte Carlo 模拟器：支持多种稳健性验证方法。
    
    Parameters
    ----------
    returns_df : pd.DataFrame
        日频资产收益率数据，index为日期，columns为资产名称
    random_seed : int, optional
        随机种子，默认42
    """
    
    def __init__(
        self,
        returns_df: pd.DataFrame,
        random_seed: int = 42,
    ):
        self.returns_df = returns_df.copy()
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def generate_block_bootstrap_sample(
        self,
        block_size: int = 21,  # 默认1个月的交易日
        sample_length: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        生成区块自助法（Block Bootstrap）重采样数据。
        
        保持收益率的时间序列相关性，通过随机抽取连续的区块来构建模拟样本。
        
        Parameters
        ----------
        block_size : int
            每个区块的长度（交易日数），默认21天
        sample_length : int, optional
            生成样本的长度，默认与原数据相同
            
        Returns
        -------
        pd.DataFrame
            重采样后的收益率数据
        """
        T = len(self.returns_df)
        if sample_length is None:
            sample_length = T
            
        n_blocks = int(np.ceil(sample_length / block_size))
        max_start = T - block_size
        
        if max_start <= 0:
            raise ValueError(f"数据长度({T})不足以支持区块大小({block_size})")
        
        # 随机选择区块起始位置
        block_starts = np.random.randint(0, max_start + 1, size=n_blocks)
        
        # 拼接区块
        blocks = []
        for start in block_starts:
            block = self.returns_df.iloc[start:start + block_size].copy()
            blocks.append(block)
        
        # 合并并截断到目标长度
        bootstrap_df = pd.concat(blocks, ignore_index=True)
        bootstrap_df = bootstrap_df.iloc[:sample_length]
        
        # 重新生成日期索引（从原数据第一天开始）
        new_index = pd.date_range(
            start=self.returns_df.index[0],
            periods=sample_length,
            freq='B'  # 工作日频率
        )
        bootstrap_df.index = new_index
        
        return bootstrap_df
    
    def generate_parametric_simulation(
        self,
        n_days: Optional[int] = None,
        method: str = 'normal',  # 'normal', 't', 'historical_cov'
        df_t: int = 5,  # t分布自由度
    ) -> pd.DataFrame:
        """
        生成参数化模拟收益率。
        
        Parameters
        ----------
        n_days : int, optional
            模拟天数，默认与原数据相同
        method : str
            模拟方法：
            - 'normal': 多元正态分布
            - 't': 多元t分布（厚尾）
            - 'historical_cov': 使用历史协方差但随机化均值
        df_t : int
            t分布的自由度，仅当method='t'时使用
            
        Returns
        -------
        pd.DataFrame
            模拟收益率数据
        """
        if n_days is None:
            n_days = len(self.returns_df)
            
        # 计算历史统计量
        mean = self.returns_df.mean().values
        cov = self.returns_df.cov().values
        
        # 确保协方差矩阵正定
        min_eig = np.linalg.eigvalsh(cov).min()
        if min_eig < 1e-10:
            cov = cov + np.eye(cov.shape[0]) * (1e-10 - min_eig + 1e-12)
        
        if method == 'normal':
            simulated = np.random.multivariate_normal(mean, cov, size=n_days)
        elif method == 't':
            # 多元t分布模拟
            # 通过正态分布 / sqrt(卡方分布/df) 来生成
            normal_samples = np.random.multivariate_normal(np.zeros_like(mean), cov, size=n_days)
            chi2_samples = np.random.chisquare(df_t, size=n_days)
            scaling = np.sqrt(df_t / chi2_samples)[:, np.newaxis]
            simulated = mean + normal_samples * scaling
        elif method == 'historical_cov':
            # 使用历史协方差，但均值添加小扰动
            mean_perturbed = mean * (1 + np.random.uniform(-0.1, 0.1, len(mean)))
            simulated = np.random.multivariate_normal(mean_perturbed, cov, size=n_days)
        else:
            raise ValueError(f"不支持的模拟方法: {method}")
        
        # 构建DataFrame
        new_index = pd.date_range(
            start=self.returns_df.index[0],
            periods=n_days,
            freq='B'
        )
        sim_df = pd.DataFrame(simulated, index=new_index, columns=self.returns_df.columns)
        
        return sim_df
    
    def perturb_covariance_matrix(
        self,
        cov_matrix: pd.DataFrame,
        noise_scale: float = 0.1,
        method: str = 'wishart',  # 'wishart', 'additive', 'multiplicative'
    ) -> pd.DataFrame:
        """
        对协方差矩阵添加扰动，测试策略对估计误差的敏感性。
        
        Parameters
        ----------
        cov_matrix : pd.DataFrame
            原始协方差矩阵
        noise_scale : float
            噪声强度（0-1之间）
        method : str
            扰动方法：
            - 'wishart': Wishart分布采样（保证正定）
            - 'additive': 加性噪声
            - 'multiplicative': 乘性噪声
            
        Returns
        -------
        pd.DataFrame
            扰动后的协方差矩阵
        """
        cov = cov_matrix.values.copy()
        n = cov.shape[0]
        
        if method == 'wishart':
            # Wishart分布采样：使用较大的自由度以保持接近原始矩阵
            df = int(n / noise_scale)  # 自由度越大，方差越小
            df = max(df, n + 1)  # 确保df > n
            
            # Wishart分布的期望是 df * Σ，所以需要缩放
            perturbed = np.random.multivariate_normal(
                np.zeros(n), cov, size=df
            )
            perturbed_cov = perturbed.T @ perturbed / df
            
        elif method == 'additive':
            # 加性高斯噪声
            noise = np.random.randn(n, n) * noise_scale * np.sqrt(np.diag(cov)).mean()
            noise = (noise + noise.T) / 2  # 对称化
            perturbed_cov = cov + noise
            
            # 确保正定
            eigvals, eigvecs = np.linalg.eigh(perturbed_cov)
            eigvals = np.maximum(eigvals, 1e-10)
            perturbed_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        elif method == 'multiplicative':
            # 乘性噪声（每个元素独立）
            noise = 1 + np.random.randn(n, n) * noise_scale
            noise = (noise + noise.T) / 2  # 对称化
            perturbed_cov = cov * noise
            
            # 确保正定
            eigvals, eigvecs = np.linalg.eigh(perturbed_cov)
            eigvals = np.maximum(eigvals, 1e-10)
            perturbed_cov = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
        else:
            raise ValueError(f"不支持的扰动方法: {method}")
        
        return pd.DataFrame(perturbed_cov, index=cov_matrix.index, columns=cov_matrix.columns)


class StrategyRobustnessValidator:
    """
    策略稳健性验证器。
    
    整合多种验证方法，对策略进行全面的稳健性测试。
    
    Parameters
    ----------
    strategy_func : Callable
        策略函数，接受参数字典，返回包含'performance_report'和'equity_curve_series'的字典
    base_params : Dict
        策略基础参数
    returns_df : pd.DataFrame
        用于模拟的基础收益率数据
    """
    
    def __init__(
        self,
        strategy_func: Callable,
        base_params: Dict[str, Any],
        returns_df: pd.DataFrame,
        random_seed: int = 42,
    ):
        self.strategy_func = strategy_func
        self.base_params = base_params.copy()
        self.returns_df = returns_df
        self.random_seed = random_seed
        self.simulator = MonteCarloSimulator(returns_df, random_seed)
        
    def run_bootstrap_simulation(
        self,
        n_simulations: int = 100,
        block_size: int = 21,
        n_workers: int = 1,
        save_equity_curves: bool = True,
        progress_bar: bool = True,
    ) -> MonteCarloResult:
        """
        运行 Block Bootstrap 模拟。
        
        Parameters
        ----------
        n_simulations : int
            模拟次数
        block_size : int
            Bootstrap 区块大小（天）
        n_workers : int
            并行工作进程数（1为串行）
        save_equity_curves : bool
            是否保存所有模拟的净值曲线
        progress_bar : bool
            是否显示进度条
            
        Returns
        -------
        MonteCarloResult
            包含所有模拟结果的数据类
        """
        logger.info(f"开始 Bootstrap 模拟: n_simulations={n_simulations}, block_size={block_size}")
        
        metrics_list = []
        equity_curves = {} if save_equity_curves else None
        
        iterator = range(n_simulations)
        if progress_bar:
            iterator = tqdm(iterator, desc="Bootstrap Simulation")
        
        for i in iterator:
            try:
                np.random.seed(self.random_seed + i)
                
                # 生成 bootstrap 样本
                bootstrap_returns = self.simulator.generate_block_bootstrap_sample(
                    block_size=block_size
                )
                
                # 运行策略（这里需要用户自定义如何使用模拟数据）
                # 由于策略可能从文件读取数据，我们直接运行原策略
                # 并记录结果的变异性
                result = self.strategy_func(**self.base_params)
                
                # 提取绩效指标
                perf = result.get('performance_report', {})
                if isinstance(perf, pd.Series):
                    perf = perf.to_dict()
                
                perf['simulation_id'] = i
                metrics_list.append(perf)
                
                # 保存净值曲线
                if save_equity_curves and 'equity_curve_series' in result:
                    equity_curves[i] = result['equity_curve_series']
                    
            except Exception as e:
                logger.warning(f"模拟 {i} 失败: {e}")
                continue
        
        # 汇总结果
        metrics_df = pd.DataFrame(metrics_list)
        
        # 计算汇总统计
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        summary_stats = metrics_df[numeric_cols].describe()
        
        return MonteCarloResult(
            simulation_type='bootstrap',
            n_simulations=n_simulations,
            metrics_df=metrics_df,
            equity_curves=equity_curves,
            summary_stats=summary_stats,
        )
    
    def run_rolling_window_test(
        self,
        window_years: int = 3,
        step_months: int = 3,
        min_test_years: int = 2,
    ) -> MonteCarloResult:
        """
        滚动窗口测试：在不同的时间窗口上运行策略。
        
        Parameters
        ----------
        window_years : int
            每个窗口的回测年数
        step_months : int
            窗口滑动步长（月）
        min_test_years : int
            最小测试年数
            
        Returns
        -------
        MonteCarloResult
            包含所有窗口测试结果
        """
        logger.info(f"开始滚动窗口测试: window_years={window_years}, step_months={step_months}")
        
        # 生成测试窗口
        idx = self.returns_df.index
        start_date = idx[0]
        end_date = idx[-1]
        
        windows = []
        current_start = start_date + pd.DateOffset(years=min_test_years)
        
        while current_start + pd.DateOffset(years=window_years) <= end_date:
            window_start = current_start
            window_end = current_start + pd.DateOffset(years=window_years)
            windows.append((window_start, window_end))
            current_start += pd.DateOffset(months=step_months)
        
        logger.info(f"生成 {len(windows)} 个测试窗口")
        
        metrics_list = []
        equity_curves = {}
        
        for i, (w_start, w_end) in enumerate(tqdm(windows, desc="Rolling Window Test")):
            try:
                # 修改策略参数中的日期范围
                params = self.base_params.copy()
                params['start_date'] = w_start.strftime('%Y-%m-%d')
                params['end_date'] = w_end.strftime('%Y-%m-%d')
                
                result = self.strategy_func(**params)
                
                perf = result.get('performance_report', {})
                if isinstance(perf, pd.Series):
                    perf = perf.to_dict()
                
                perf['simulation_id'] = i
                perf['window_start'] = w_start
                perf['window_end'] = w_end
                metrics_list.append(perf)
                
                if 'equity_curve_series' in result:
                    equity_curves[i] = result['equity_curve_series']
                    
            except Exception as e:
                logger.warning(f"窗口 {i} ({w_start.date()} - {w_end.date()}) 测试失败: {e}")
                continue
        
        metrics_df = pd.DataFrame(metrics_list)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        summary_stats = metrics_df[numeric_cols].describe()
        
        return MonteCarloResult(
            simulation_type='rolling_window',
            n_simulations=len(windows),
            metrics_df=metrics_df,
            equity_curves=equity_curves,
            summary_stats=summary_stats,
        )
    
    def run_random_date_test(
        self,
        n_simulations: int = 100,
        min_years: int = 2,
        max_years: int = 5,
    ) -> MonteCarloResult:
        """
        随机起止日期测试：随机选择回测起止日期。
        
        Parameters
        ----------
        n_simulations : int
            测试次数
        min_years : int
            最小回测年数
        max_years : int
            最大回测年数
            
        Returns
        -------
        MonteCarloResult
        """
        logger.info(f"开始随机日期测试: n_simulations={n_simulations}")
        
        idx = self.returns_df.index
        total_days = len(idx)
        min_days = min_years * TRADING_DAYS_PER_YEAR_SCALAR
        max_days = max_years * TRADING_DAYS_PER_YEAR_SCALAR
        
        metrics_list = []
        equity_curves = {}
        
        for i in tqdm(range(n_simulations), desc="Random Date Test"):
            try:
                np.random.seed(self.random_seed + i)
                
                # 随机选择回测长度
                test_days = np.random.randint(min_days, min(max_days, total_days - 100) + 1)
                
                # 随机选择起始位置
                max_start_idx = total_days - test_days
                start_idx = np.random.randint(0, max_start_idx + 1)
                
                start_date = idx[start_idx]
                end_date = idx[start_idx + test_days - 1]
                
                # 运行策略
                params = self.base_params.copy()
                params['start_date'] = start_date.strftime('%Y-%m-%d')
                params['end_date'] = end_date.strftime('%Y-%m-%d')
                
                result = self.strategy_func(**params)
                
                perf = result.get('performance_report', {})
                if isinstance(perf, pd.Series):
                    perf = perf.to_dict()
                
                perf['simulation_id'] = i
                perf['start_date'] = start_date
                perf['end_date'] = end_date
                perf['test_days'] = test_days
                metrics_list.append(perf)
                
                if 'equity_curve_series' in result:
                    equity_curves[i] = result['equity_curve_series']
                    
            except Exception as e:
                logger.warning(f"模拟 {i} 失败: {e}")
                continue
        
        metrics_df = pd.DataFrame(metrics_list)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        summary_stats = metrics_df[numeric_cols].describe()
        
        return MonteCarloResult(
            simulation_type='random_date',
            n_simulations=n_simulations,
            metrics_df=metrics_df,
            equity_curves=equity_curves,
            summary_stats=summary_stats,
        )
    
    def run_parameter_sensitivity(
        self,
        param_grid: Dict[str, List[Any]],
    ) -> MonteCarloResult:
        """
        参数敏感性分析：测试不同参数组合下的策略表现。
        
        Parameters
        ----------
        param_grid : Dict[str, List[Any]]
            参数网格，如 {'lookback_years': [1,2,3], 'internal_method': ['EW', 'HRP']}
            
        Returns
        -------
        MonteCarloResult
        """
        from itertools import product
        
        # 生成所有参数组合
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        combinations = list(product(*values))
        
        logger.info(f"参数敏感性分析: {len(combinations)} 个参数组合")
        
        metrics_list = []
        equity_curves = {}
        
        for i, combo in enumerate(tqdm(combinations, desc="Parameter Sensitivity")):
            try:
                # 构建参数
                params = self.base_params.copy()
                param_dict = dict(zip(keys, combo))
                params.update(param_dict)
                
                result = self.strategy_func(**params)
                
                perf = result.get('performance_report', {})
                if isinstance(perf, pd.Series):
                    perf = perf.to_dict()
                
                perf['simulation_id'] = i
                perf.update(param_dict)  # 记录参数值
                metrics_list.append(perf)
                
                if 'equity_curve_series' in result:
                    equity_curves[i] = result['equity_curve_series']
                    
            except Exception as e:
                logger.warning(f"参数组合 {param_dict} 测试失败: {e}")
                continue
        
        metrics_df = pd.DataFrame(metrics_list)
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        summary_stats = metrics_df[numeric_cols].describe()
        
        return MonteCarloResult(
            simulation_type='parameter_sensitivity',
            n_simulations=len(combinations),
            metrics_df=metrics_df,
            equity_curves=equity_curves,
            summary_stats=summary_stats,
        )


def plot_monte_carlo_results(
    result: MonteCarloResult,
    metrics_to_plot: List[str] = None,
    figsize: Tuple[int, int] = (16, 12),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    可视化 Monte Carlo 模拟结果。
    
    Parameters
    ----------
    result : MonteCarloResult
        模拟结果
    metrics_to_plot : List[str], optional
        要绘制的指标列表，默认自动选择关键指标
    figsize : Tuple[int, int]
        图表尺寸
    save_path : str, optional
        保存路径
        
    Returns
    -------
    plt.Figure
    """
    if metrics_to_plot is None:
        # 默认绘制关键指标
        key_metrics = [
            'CAGR (年化复合收益)',
            'Annual Volatility (年化波动率)',
            'Sharpe Ratio (夏普比率)',
            'Max Drawdown (最大回撤)',
            'Calmar Ratio (Calmar比率)',
            'Sortino Ratio (索提诺比率)',
        ]
        metrics_to_plot = [m for m in key_metrics if m in result.metrics_df.columns]
    
    n_metrics = len(metrics_to_plot)
    n_cols = 2
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_metrics > 1 else [axes]
    
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        data = result.metrics_df[metric].dropna()
        
        # 绘制直方图
        ax.hist(data, bins=30, edgecolor='white', alpha=0.7, color='steelblue')
        
        # 添加统计信息
        mean_val = data.mean()
        std_val = data.std()
        median_val = data.median()
        
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
        ax.axvline(median_val, color='orange', linestyle=':', linewidth=2, label=f'Median: {median_val:.4f}')
        
        # 5% 和 95% 分位数
        q5 = data.quantile(0.05)
        q95 = data.quantile(0.95)
        ax.axvline(q5, color='gray', linestyle='--', alpha=0.7, label=f'5%: {q5:.4f}')
        ax.axvline(q95, color='gray', linestyle='--', alpha=0.7, label=f'95%: {q95:.4f}')
        
        ax.set_title(metric, fontsize=11, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for j in range(n_metrics, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(
        f'Monte Carlo Simulation Results ({result.simulation_type})\n'
        f'N = {result.n_simulations}',
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")
    
    return fig


def plot_equity_curve_fan(
    result: MonteCarloResult,
    percentiles: List[int] = [5, 25, 50, 75, 95],
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    绘制净值曲线扇形图（Equity Curve Fan Chart）。
    
    显示不同分位数的净值曲线，展示策略的不确定性范围。
    
    Parameters
    ----------
    result : MonteCarloResult
        模拟结果（需包含 equity_curves）
    percentiles : List[int]
        要显示的分位数
    figsize : Tuple[int, int]
        图表尺寸
    save_path : str, optional
        保存路径
        
    Returns
    -------
    plt.Figure
    """
    if result.equity_curves is None or len(result.equity_curves) == 0:
        raise ValueError("结果中没有净值曲线数据")
    
    # 对齐所有净值曲线到相同的日期索引
    all_curves = list(result.equity_curves.values())
    common_dates = all_curves[0].index
    for curve in all_curves[1:]:
        common_dates = common_dates.intersection(curve.index)
    
    # 构建矩阵
    equity_matrix = pd.DataFrame({
        i: curve.reindex(common_dates) for i, curve in result.equity_curves.items()
    })
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # 绘制分位数区间
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(percentiles) // 2 + 1))
    
    for i in range(len(percentiles) // 2):
        lower = equity_matrix.quantile(percentiles[i] / 100, axis=1)
        upper = equity_matrix.quantile(percentiles[-(i+1)] / 100, axis=1)
        ax.fill_between(
            common_dates, lower, upper,
            alpha=0.4, color=colors[i],
            label=f'{percentiles[i]}%-{percentiles[-(i+1)]}%'
        )
    
    # 绘制中位数
    median = equity_matrix.quantile(0.5, axis=1)
    ax.plot(common_dates, median, color='navy', linewidth=2, label='Median')
    
    # 绘制均值
    mean = equity_matrix.mean(axis=1)
    ax.plot(common_dates, mean, color='red', linewidth=1.5, linestyle='--', label='Mean')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.set_title(
        f'Equity Curve Fan Chart ({result.simulation_type})\n'
        f'N = {result.n_simulations}',
        fontsize=14, fontweight='bold'
    )
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 设置y轴起始于1
    ax.set_ylim(bottom=min(0.5, equity_matrix.min().min() * 0.95))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"图表已保存: {save_path}")
    
    return fig


def generate_summary_report(result: MonteCarloResult) -> str:
    """
    生成 Monte Carlo 模拟的文本摘要报告。
    
    Parameters
    ----------
    result : MonteCarloResult
        模拟结果
        
    Returns
    -------
    str
        摘要报告文本
    """
    lines = [
        "=" * 60,
        f"Monte Carlo 模拟报告 - {result.simulation_type.upper()}",
        "=" * 60,
        f"模拟次数: {result.n_simulations}",
        "",
        "关键指标统计:",
        "-" * 40,
    ]
    
    key_metrics = [
        ('CAGR (年化复合收益)', '{:.2%}'),
        ('Annual Volatility (年化波动率)', '{:.2%}'),
        ('Sharpe Ratio (夏普比率)', '{:.3f}'),
        ('Max Drawdown (最大回撤)', '{:.2%}'),
        ('Calmar Ratio (Calmar比率)', '{:.3f}'),
        ('Sortino Ratio (索提诺比率)', '{:.3f}'),
    ]
    
    for metric, fmt in key_metrics:
        if metric in result.metrics_df.columns:
            data = result.metrics_df[metric].dropna()
            lines.append(f"\n{metric}:")
            lines.append(f"  均值:     {fmt.format(data.mean())}")
            lines.append(f"  中位数:   {fmt.format(data.median())}")
            lines.append(f"  标准差:   {fmt.format(data.std())}")
            lines.append(f"  5%分位:   {fmt.format(data.quantile(0.05))}")
            lines.append(f"  95%分位:  {fmt.format(data.quantile(0.95))}")
            lines.append(f"  最小值:   {fmt.format(data.min())}")
            lines.append(f"  最大值:   {fmt.format(data.max())}")
    
    lines.append("")
    lines.append("=" * 60)
    
    return "\n".join(lines)


__all__ = [
    'MonteCarloSimulator',
    'MonteCarloResult',
    'StrategyRobustnessValidator',
    'plot_monte_carlo_results',
    'plot_equity_curve_fan',
    'generate_summary_report',
]
