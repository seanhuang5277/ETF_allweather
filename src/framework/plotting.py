# -*- coding: utf-8 -*-
"""plotting.py

通用绘图模块：
- 净值曲线绘制（含标题和绩效摘要）
- 仓位历史面积图
- 资产收益贡献归因图
"""
import matplotlib.pyplot as plt

# Matplotlib 全局中文设置（与策略脚本一致）
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"中文字体设置警告: {e}")


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
