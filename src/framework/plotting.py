# -*- coding: utf-8 -*-
"""plotting.py

通用绘图模块：
- 净值曲线绘制（含标题和绩效摘要）
- 仓位历史面积图
- 资产收益贡献归因图

风格说明:
- 原始函数 (plot_*): 保持原有风格
- 现代风格函数 (plot_*_modern): Monte Carlo 一致的现代简洁风格
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, Dict

# Matplotlib 全局中文设置（与策略脚本一致）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置警告: {e}")


# =============================================================================
# 现代风格配色方案 (与 Monte Carlo 模块一致)
# =============================================================================
MODERN_COLORS = {
    'primary': '#2E4057',      # 深蓝灰 - 主色调
    'secondary': '#048A81',    # 青绿色 - 次色调
    'accent': '#54C6EB',       # 亮蓝色 - 强调色
    'positive': '#8AC926',     # 绿色 - 正收益
    'negative': '#FF595E',     # 红色 - 负收益
    'neutral': '#6C757D',      # 中性灰
    'background': '#F8F9FA',   # 浅灰背景
    'grid': '#DEE2E6',         # 网格线颜色
    'text': '#212529',         # 文字颜色
}

# 现代配色板 (用于多曲线/多类别)
MODERN_PALETTE = [
    '#2E4057',  # 深蓝灰
    '#048A81',  # 青绿
    '#54C6EB',  # 亮蓝
    '#8AC926',  # 黄绿
    '#FFCA3A',  # 金黄
    '#FF595E',  # 珊瑚红
    '#9B5DE5',  # 紫色
    '#F15BB5',  # 粉红
    '#00BBF9',  # 天蓝
    '#00F5D4',  # 青色
]

def _apply_modern_style(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    """应用现代风格到坐标轴"""
    # 设置标题和标签
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold', color=MODERN_COLORS['text'], pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=MODERN_COLORS['text'])
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=MODERN_COLORS['text'])
    
    # 设置网格
    ax.grid(True, linestyle='-', alpha=0.3, color=MODERN_COLORS['grid'], linewidth=0.8)
    ax.set_axisbelow(True)  # 网格在数据下方
    
    # 设置边框
    for spine in ax.spines.values():
        spine.set_color(MODERN_COLORS['grid'])
        spine.set_linewidth(0.8)
    
    # 设置刻度颜色
    ax.tick_params(colors=MODERN_COLORS['text'], labelsize=10)


def plot_equity_curve(equity_curve_series, performance_report, strategy_mode: str, title_suffix: str = "", 
                      rebalance_returns=None, auto_show: bool = True):
    """绘制净值曲线，并可选在下方绘制调仓收益柱状图。

    参数:
        equity_curve_series: 净值序列
        performance_report: 绩效指标（可为 None）
        strategy_mode: 策略名称
        title_suffix: 标题附加行
        rebalance_returns: 调仓收益序列（可选），如果提供则绘制双子图
        auto_show: 是否在函数内部直接 plt.show()；False 时仅返回 Figure，不阻塞
    返回:
        fig: matplotlib.figure.Figure
    """
    if performance_report is not None:
        title = (
            f"策略: {strategy_mode} - 净值曲线\n"
            f"CAGR: {performance_report['CAGR (年化复合收益)']: .2%} | "
            f"Sharpe: {performance_report['Sharpe Ratio (夏普比率)']: .2f} | "
            f"MDD: {performance_report['Max Drawdown (最大回撤)']: .2%}"
        )
    else:
        title = f"策略: {strategy_mode} - 净值曲线"
    if title_suffix:
        title += f"\n{title_suffix}"

    # 如果提供了调仓收益，使用双Y轴在同一图中绘制
    if rebalance_returns is not None:
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.gca()
        
        # 左Y轴：净值曲线
        equity_curve_series.plot(ax=ax1, legend=False, linewidth=2, color='#1f77b4', label='净值曲线')
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('组合净值 (初始=1)', fontsize=12, color='#1f77b4')
        ax1.tick_params(axis='y', labelcolor='#1f77b4')
        ax1.grid(True, alpha=0.3)
        
        # 右Y轴：调仓收益柱状图
        ax2 = ax1.twinx()
        colors = ['#2ca02c' if v < 0 else '#d62728' for v in rebalance_returns.values]  # 红涨绿跌
        ax2.bar(rebalance_returns.index, rebalance_returns.values, 
                color=colors, alpha=0.5, width=20, label='调仓期收益')
        ax2.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.3)
        ax2.set_ylabel('调仓期收益', fontsize=12, color='#d62728')
        ax2.tick_params(axis='y', labelcolor='#d62728')
        
        # 格式化右Y轴为百分比
        from matplotlib.ticker import FuncFormatter
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
        
    else:
        # 单图：仅净值曲线
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.gca()
        equity_curve_series.plot(ax=ax1, legend=False, linewidth=2)
        ax1.set_title(title, fontsize=16)
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_ylabel('组合净值 (初始=1)', fontsize=12)
        ax1.grid(True)
    
    fig.tight_layout()
    if auto_show:
        plt.show()
    return fig


def plot_multi_equity_curves(equity_curves_dict, title="多曲线净值对比", auto_show=True):
    """绘制多条净值曲线对比图"""
    fig = plt.figure(figsize=(12, 7))
    ax = fig.gca()
    
    # 使用默认色板
    colors = plt.cm.tab10.colors
    
    for i, (name, curve) in enumerate(equity_curves_dict.items()):
        # 归一化起点为1
        if not curve.empty:
            normalized_curve = curve / curve.iloc[0]
            color = colors[i % len(colors)]
            normalized_curve.plot(ax=ax, label=f"{name} (最终: {normalized_curve.iloc[-1]:.2f})", linewidth=2, color=color)
        
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('净值 (初始=1)', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', fontsize=10)
    
    fig.tight_layout()
    if auto_show:
        plt.show()
    return fig


# 兼容旧名称
plot_quadrant_equity_curves = plot_multi_equity_curves


def plot_weights_history(weights_history_df, strategy_mode: str, reverse_legend: bool = False, auto_show: bool = True):
    """绘制仓位历史面积图。

    参数:
        weights_history_df: 调仓日或每日仓位 DataFrame
        strategy_mode: 策略名称
        reverse_legend: 是否反转图例顺序
        auto_show: 是否立即 show()
    返回:
        fig: matplotlib.figure.Figure
    """
    fig = plt.figure(figsize=(14, 8))
    ax = fig.gca()
    weights_history_df.plot.area(ax=ax, stacked=True, legend='reverse' if reverse_legend else True)
    ax.set_title(f'{strategy_mode} 策略仓位历史变化')
    ax.set_ylabel('权重')
    ax.set_xlabel('日期')
    ax.grid(True, linestyle='--', alpha=0.5)
    if reverse_legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    fig.tight_layout()
    if auto_show:
        plt.show()
    return fig


def plot_return_attribution(attribution_series, title="收益归因", auto_show=True):
    """绘制资产收益贡献柱状图。

    参数:
        attribution_series: 资产贡献度序列 (index=资产名, value=贡献值)
        title: 图表标题
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure对象
    """
    # 动态调整高度
    height = max(6, len(attribution_series) * 0.4)
    fig, ax = plt.subplots(figsize=(12, height))

    # 颜色区分正负（红涨绿跌）
    colors = ['#2ca02c' if v < 0 else '#d62728' for v in attribution_series.values]

    # 绘制水平柱状图 (barh)
    # 注意：pandas plot barh 默认是从下往上画，所以如果 series 是降序，画出来最大的在最下面
    # 为了让最大的在最上面，我们需要反转一下顺序
    plot_series = attribution_series.iloc[::-1]
    plot_colors = colors[::-1]

    plot_series.plot(kind='barh', ax=ax, color=plot_colors, alpha=0.8)

    # 添加数值标签
    for i, v in enumerate(plot_series.values):
        ax.text(v, i, f' {v:.2%}', va='center', fontsize=10)

    # 计算总和并添加到标题
    total_sum = attribution_series.sum()
    ax.set_title(f"{title} (合计: {total_sum:.2%})", fontsize=14)
    
    ax.set_xlabel("累计收益贡献", fontsize=12)
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)

    # 绘制 0 线
    ax.axvline(0, color='black', linewidth=0.8)

    fig.tight_layout()
    if auto_show:
        plt.show()
    return fig


def show_all_block():
    """一次性阻塞显示所有已创建但未显示的图。
    适用于先生成多个 Figure，再集中展示，避免逐个关闭。
    """
    plt.show()


# =============================================================================
# 现代风格绘图函数 (与 Monte Carlo 模块风格一致)
# =============================================================================

def plot_equity_curve_modern(
    equity_curve_series, 
    performance_report, 
    strategy_mode: str, 
    title_suffix: str = "", 
    rebalance_returns=None, 
    benchmark_series=None,
    auto_show: bool = True
):
    """
    绘制净值曲线 - 现代简洁风格
    
    参数:
        equity_curve_series: 净值序列
        performance_report: 绩效指标（可为 None）
        strategy_mode: 策略名称
        title_suffix: 标题附加行
        rebalance_returns: 调仓收益序列（可选）
        benchmark_series: 基准净值曲线（可选）
        auto_show: 是否立即显示
    返回:
        fig: matplotlib.figure.Figure
    """
    # 创建图形
    if rebalance_returns is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[3, 1], 
                                        gridspec_kw={'hspace': 0.15})
    else:
        fig, ax1 = plt.subplots(figsize=(14, 7))
    
    fig.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    # 构建标题
    if performance_report is not None:
        cagr = performance_report.get('CAGR (年化复合收益)', 0)
        sharpe = performance_report.get('Sharpe Ratio (夏普比率)', 0)
        mdd = performance_report.get('Max Drawdown (最大回撤)', 0)
        vol = performance_report.get('Annual Volatility (年化波动率)', 0)
        
        title = f"{strategy_mode}"
        subtitle = f"CAGR: {cagr:.2%}  |  Sharpe: {sharpe:.2f}  |  Vol: {vol:.2%}  |  MaxDD: {mdd:.2%}"
    else:
        title = f"{strategy_mode}"
        subtitle = ""
    
    if title_suffix:
        subtitle = f"{subtitle}\n{title_suffix}" if subtitle else title_suffix
    
    # 绘制净值曲线
    ax1.plot(equity_curve_series.index, equity_curve_series.values, 
             color=MODERN_COLORS['primary'], linewidth=2.5, label='策略净值')
    
    # 填充净值曲线下方区域
    ax1.fill_between(equity_curve_series.index, 1, equity_curve_series.values, 
                     alpha=0.15, color=MODERN_COLORS['primary'])
    
    # 绘制基准曲线（如果有）
    if benchmark_series is not None:
        ax1.plot(benchmark_series.index, benchmark_series.values, 
                 color=MODERN_COLORS['neutral'], linewidth=1.5, linestyle='--', 
                 label='基准', alpha=0.7)
    
    # 标注最大回撤区间
    if performance_report is not None:
        try:
            mdd_start = performance_report.get('Max Drawdown Start (最大回撤开始日)')
            mdd_end = performance_report.get('Max Drawdown End (最大回撤结束日)')
            if mdd_start is not None and mdd_end is not None:
                ax1.axvspan(mdd_start, mdd_end, alpha=0.1, color=MODERN_COLORS['negative'], 
                           label=f'最大回撤区间')
        except:
            pass
    
    # 应用现代风格
    _apply_modern_style(ax1, title=title, xlabel="", ylabel="净值")
    
    # 添加副标题
    if subtitle:
        ax1.text(0.5, 1.02, subtitle, transform=ax1.transAxes, ha='center', 
                fontsize=11, color=MODERN_COLORS['neutral'], style='italic')
    
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
              framealpha=0.9, edgecolor=MODERN_COLORS['grid'], fontsize=10)
    
    # 绘制调仓收益柱状图
    if rebalance_returns is not None:
        ax2.set_facecolor('white')
        
        colors = [MODERN_COLORS['positive'] if v >= 0 else MODERN_COLORS['negative'] 
                  for v in rebalance_returns.values]
        
        ax2.bar(rebalance_returns.index, rebalance_returns.values, 
                color=colors, alpha=0.7, width=15, edgecolor='white', linewidth=0.5)
        ax2.axhline(0, color=MODERN_COLORS['neutral'], linewidth=1, linestyle='-', alpha=0.5)
        
        _apply_modern_style(ax2, xlabel="日期", ylabel="调仓收益")
        
        # 格式化Y轴为百分比
        from matplotlib.ticker import FuncFormatter
        ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 同步X轴范围
        ax2.set_xlim(ax1.get_xlim())
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_multi_equity_curves_modern(
    equity_curves_dict: Dict, 
    title: str = "净值曲线对比",
    normalize: bool = True,
    show_final_value: bool = True,
    auto_show: bool = True
):
    """
    绘制多条净值曲线对比图 - 现代风格
    
    参数:
        equity_curves_dict: {名称: 净值序列} 字典
        title: 图表标题
        normalize: 是否归一化起点为1
        show_final_value: 是否在图例中显示最终值
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    n_curves = len(equity_curves_dict)
    
    for i, (name, curve) in enumerate(equity_curves_dict.items()):
        if curve.empty:
            continue
            
        # 归一化
        if normalize:
            plot_curve = curve / curve.iloc[0]
        else:
            plot_curve = curve
        
        color = MODERN_PALETTE[i % len(MODERN_PALETTE)]
        
        # 构建标签
        if show_final_value:
            label = f"{name} ({plot_curve.iloc[-1]:.2f})"
        else:
            label = name
        
        ax.plot(plot_curve.index, plot_curve.values, 
                color=color, linewidth=2, label=label, alpha=0.9)
    
    # 基准线
    ax.axhline(1, color=MODERN_COLORS['neutral'], linewidth=1, linestyle='--', alpha=0.5)
    
    _apply_modern_style(ax, title=title, xlabel="日期", ylabel="净值 (初始=1)" if normalize else "净值")
    
    # 图例放在右侧
    ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
             framealpha=0.9, edgecolor=MODERN_COLORS['grid'], fontsize=10)
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_weights_history_modern(
    weights_history_df, 
    title: str = "仓位配置历史",
    show_legend: bool = True,
    auto_show: bool = True
):
    """
    绘制仓位历史面积图 - 现代风格
    
    参数:
        weights_history_df: 权重历史 DataFrame
        title: 图表标题
        show_legend: 是否显示图例
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    n_cols = len(weights_history_df.columns)
    colors = [MODERN_PALETTE[i % len(MODERN_PALETTE)] for i in range(n_cols)]
    
    # 绘制堆叠面积图
    weights_history_df.plot.area(ax=ax, stacked=True, color=colors, alpha=0.85, linewidth=0.5)
    
    _apply_modern_style(ax, title=title, xlabel="日期", ylabel="权重占比")
    
    # 设置Y轴范围
    ax.set_ylim(0, 1.05)
    
    # 格式化Y轴为百分比
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    if show_legend:
        # 图例放在右侧外部
        ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), 
                 frameon=True, fancybox=True, shadow=False,
                 framealpha=0.9, edgecolor=MODERN_COLORS['grid'], fontsize=9)
    else:
        ax.get_legend().remove()
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_return_attribution_modern(
    attribution_series, 
    title: str = "收益归因分析",
    top_n: int = None,
    auto_show: bool = True
):
    """
    绘制资产收益贡献柱状图 - 现代风格
    
    参数:
        attribution_series: 资产贡献度序列
        title: 图表标题
        top_n: 只显示前N个（按绝对值排序），None表示全部
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    # 如果指定了 top_n，只取绝对值最大的前N个
    if top_n is not None and len(attribution_series) > top_n:
        sorted_abs = attribution_series.abs().sort_values(ascending=False)
        top_indices = sorted_abs.head(top_n).index
        attribution_series = attribution_series.loc[top_indices]
        # 重新按值排序
        attribution_series = attribution_series.sort_values(ascending=True)
    else:
        attribution_series = attribution_series.sort_values(ascending=True)
    
    # 动态调整高度
    height = max(6, len(attribution_series) * 0.45)
    fig, ax = plt.subplots(figsize=(12, height))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 颜色区分正负
    colors = [MODERN_COLORS['positive'] if v >= 0 else MODERN_COLORS['negative'] 
              for v in attribution_series.values]
    
    # 绘制水平柱状图
    bars = ax.barh(range(len(attribution_series)), attribution_series.values, 
                   color=colors, alpha=0.85, edgecolor='white', linewidth=0.5)
    
    # 设置Y轴标签
    ax.set_yticks(range(len(attribution_series)))
    ax.set_yticklabels(attribution_series.index, fontsize=10)
    
    # 添加数值标签
    for i, (bar, v) in enumerate(zip(bars, attribution_series.values)):
        # 根据正负决定标签位置
        if v >= 0:
            ax.text(v + 0.001, i, f' {v:.2%}', va='center', ha='left', 
                   fontsize=9, color=MODERN_COLORS['text'])
        else:
            ax.text(v - 0.001, i, f'{v:.2%} ', va='center', ha='right', 
                   fontsize=9, color=MODERN_COLORS['text'])
    
    # 零线
    ax.axvline(0, color=MODERN_COLORS['neutral'], linewidth=1.5)
    
    # 计算总和
    total_sum = attribution_series.sum()
    full_title = f"{title}\n合计贡献: {total_sum:.2%}"
    
    _apply_modern_style(ax, title=full_title, xlabel="累计收益贡献", ylabel="")
    
    # 格式化X轴为百分比
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_drawdown_modern(
    equity_curve_series,
    title: str = "回撤分析",
    auto_show: bool = True
):
    """
    绘制回撤曲线图 - 现代风格
    
    参数:
        equity_curve_series: 净值序列
        title: 图表标题
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), height_ratios=[2, 1],
                                    gridspec_kw={'hspace': 0.15})
    fig.patch.set_facecolor('white')
    
    # 计算回撤
    running_max = equity_curve_series.cummax()
    drawdown = (equity_curve_series - running_max) / running_max
    
    # 上图：净值曲线
    ax1.set_facecolor('white')
    ax1.plot(equity_curve_series.index, equity_curve_series.values, 
             color=MODERN_COLORS['primary'], linewidth=2, label='策略净值')
    ax1.plot(running_max.index, running_max.values, 
             color=MODERN_COLORS['secondary'], linewidth=1, linestyle='--', 
             alpha=0.7, label='历史最高')
    
    ax1.fill_between(equity_curve_series.index, equity_curve_series.values, running_max.values,
                     alpha=0.3, color=MODERN_COLORS['negative'])
    
    _apply_modern_style(ax1, title=title, ylabel="净值")
    ax1.legend(loc='upper left', frameon=True, fancybox=True, fontsize=10)
    
    # 下图：回撤曲线
    ax2.set_facecolor('white')
    ax2.fill_between(drawdown.index, 0, drawdown.values, 
                     alpha=0.6, color=MODERN_COLORS['negative'])
    ax2.plot(drawdown.index, drawdown.values, 
             color=MODERN_COLORS['negative'], linewidth=1)
    
    # 标注最大回撤
    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax2.scatter([max_dd_date], [max_dd], color=MODERN_COLORS['negative'], 
               s=80, zorder=5, edgecolors='white', linewidths=2)
    ax2.annotate(f'最大回撤: {max_dd:.2%}', 
                xy=(max_dd_date, max_dd), 
                xytext=(10, -20), textcoords='offset points',
                fontsize=10, color=MODERN_COLORS['negative'],
                arrowprops=dict(arrowstyle='->', color=MODERN_COLORS['negative'], lw=1))
    
    _apply_modern_style(ax2, xlabel="日期", ylabel="回撤")
    
    # 格式化Y轴为百分比
    from matplotlib.ticker import FuncFormatter
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0%}'))
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_performance_summary_modern(
    performance_report,
    title: str = "策略绩效摘要",
    auto_show: bool = True
):
    """
    绘制绩效指标摘要卡片 - 现代风格
    
    参数:
        performance_report: 绩效指标字典或 Series
        title: 图表标题
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    if isinstance(performance_report, pd.Series):
        performance_report = performance_report.to_dict()
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    ax.axis('off')
    
    # 定义要显示的关键指标
    key_metrics = [
        ('CAGR (年化复合收益)', '{:.2%}', 'CAGR'),
        ('Annual Volatility (年化波动率)', '{:.2%}', '年化波动'),
        ('Sharpe Ratio (夏普比率)', '{:.2f}', 'Sharpe'),
        ('Calmar Ratio (Calmar比率)', '{:.2f}', 'Calmar'),
        ('Sortino Ratio (索提诺比率)', '{:.2f}', 'Sortino'),
        ('Max Drawdown (最大回撤)', '{:.2%}', '最大回撤'),
        ('Rebalance Win Rate (换仓胜率)', '{:.1%}', '换仓胜率'),
        ('Rebalance Odds (盈亏比)', '{:.2f}', '盈亏比'),
    ]
    
    # 过滤存在的指标
    available_metrics = [(k, fmt, label) for k, fmt, label in key_metrics if k in performance_report]
    
    # 绘制标题
    ax.text(0.5, 0.95, title, transform=ax.transAxes, ha='center', va='top',
           fontsize=16, fontweight='bold', color=MODERN_COLORS['text'])
    
    # 计算网格布局
    n_metrics = len(available_metrics)
    n_cols = 4
    n_rows = int(np.ceil(n_metrics / n_cols))
    
    # 绘制指标卡片
    for i, (key, fmt, label) in enumerate(available_metrics):
        row = i // n_cols
        col = i % n_cols
        
        x = 0.125 + col * 0.25
        y = 0.7 - row * 0.35
        
        value = performance_report[key]
        
        # 根据指标类型决定颜色
        if 'Drawdown' in key or '回撤' in key:
            value_color = MODERN_COLORS['negative']
        elif any(k in key for k in ['CAGR', 'Sharpe', 'Calmar', 'Sortino', '胜率']):
            value_color = MODERN_COLORS['positive'] if value > 0 else MODERN_COLORS['negative']
        else:
            value_color = MODERN_COLORS['text']
        
        # 绘制指标框
        bbox = dict(boxstyle='round,pad=0.5', facecolor=MODERN_COLORS['background'], 
                   edgecolor=MODERN_COLORS['grid'], linewidth=1)
        
        # 数值
        try:
            value_str = fmt.format(value)
        except:
            value_str = str(value)
        
        ax.text(x, y + 0.05, value_str, transform=ax.transAxes, ha='center', va='center',
               fontsize=18, fontweight='bold', color=value_color)
        
        # 标签
        ax.text(x, y - 0.08, label, transform=ax.transAxes, ha='center', va='center',
               fontsize=11, color=MODERN_COLORS['neutral'])
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig


def plot_rolling_metrics_modern(
    returns_series,
    window: int = 252,
    metrics: list = ['sharpe', 'volatility'],
    title: str = "滚动绩效指标",
    auto_show: bool = True
):
    """
    绘制滚动绩效指标 - 现代风格
    
    参数:
        returns_series: 日收益率序列
        window: 滚动窗口（天数）
        metrics: 要计算的指标列表 ['sharpe', 'volatility', 'return']
        title: 图表标题
        auto_show: 是否立即显示
    返回:
        fig: matplotlib Figure
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 4 * n_metrics), 
                             sharex=True, gridspec_kw={'hspace': 0.1})
    fig.patch.set_facecolor('white')
    
    if n_metrics == 1:
        axes = [axes]
    
    colors = [MODERN_COLORS['primary'], MODERN_COLORS['secondary'], MODERN_COLORS['accent']]
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        ax.set_facecolor('white')
        
        if metric == 'sharpe':
            # 滚动 Sharpe
            rolling_ret = returns_series.rolling(window).mean() * 252
            rolling_vol = returns_series.rolling(window).std() * np.sqrt(252)
            rolling_sharpe = rolling_ret / rolling_vol
            rolling_sharpe.plot(ax=ax, color=colors[i % len(colors)], linewidth=1.5)
            ax.axhline(0, color=MODERN_COLORS['neutral'], linestyle='--', alpha=0.5)
            ax.axhline(1, color=MODERN_COLORS['positive'], linestyle=':', alpha=0.5)
            ylabel = f'滚动 Sharpe ({window}天)'
            
        elif metric == 'volatility':
            # 滚动波动率
            rolling_vol = returns_series.rolling(window).std() * np.sqrt(252)
            rolling_vol.plot(ax=ax, color=colors[i % len(colors)], linewidth=1.5)
            ax.fill_between(rolling_vol.index, 0, rolling_vol.values, alpha=0.2, color=colors[i % len(colors)])
            ylabel = f'滚动波动率 ({window}天)'
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
            
        elif metric == 'return':
            # 滚动收益
            rolling_ret = returns_series.rolling(window).mean() * 252
            rolling_ret.plot(ax=ax, color=colors[i % len(colors)], linewidth=1.5)
            ax.axhline(0, color=MODERN_COLORS['neutral'], linestyle='--', alpha=0.5)
            ylabel = f'滚动年化收益 ({window}天)'
            from matplotlib.ticker import FuncFormatter
            ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        _apply_modern_style(ax, ylabel=ylabel)
    
    axes[0].set_title(title, fontsize=14, fontweight='bold', color=MODERN_COLORS['text'], pad=15)
    axes[-1].set_xlabel('日期', fontsize=11, color=MODERN_COLORS['text'])
    
    plt.tight_layout()
    
    if auto_show:
        plt.show()
    return fig
