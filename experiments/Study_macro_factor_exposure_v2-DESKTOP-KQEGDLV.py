# -*- coding: utf-8 -*-
"""
@Project: ETF Strategy
@File   : Study_macro_factor_exposure_v2.py
@Description: 
    Macro Factor vs Asset Return Analysis - V2
    
    Framework:
    ============================================
    1. Level Factor Analysis (State-driven)
       - Regime Analysis: Quantile grouping -> Asset performance by state
       - Beta Exposure: Multi-factor regression
       - Quadrant Analysis: Growth x Inflation quadrant
       
    2. Change Factor Analysis (Momentum-driven)
       - Predictive IC: corr(Change_t, R_{t+1/t+2})
       - IR (Information Ratio): IC_mean / IC_std
       - Predictive Regression: R_{t+1} = a + b * Change_t + e
       
    3. Level + Change Joint Analysis
       - Cross Model: R_{t+1} = a + b1*Level_t + b2*Change_t + e
       - Level x Change Quadrant Heatmap
    
    4. Sub-factor Analysis
       - Analyze each sub-component of main factors
    
    Important:
    ============================================
    - Use T+1 or T+2 future returns to avoid look-ahead bias
    - Level factors: Use regime/quantile analysis (not IC/IR)
    - Change factors: Use IC/IR for predictive power
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# --- Path Config ---
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'analysis_results')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

INDEX_RETURNS_FILE = os.path.join(DATA_DIR, 'index_daily_simple_returns.csv')
MACRO_FACTORS_FILE = os.path.join(DATA_DIR, 'all_macro_factors.csv')
SUB_FACTORS_FILE = os.path.join(DATA_DIR, 'all_macro_sub_factors.csv')

# --- Plot Settings ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 100

# --- Key Assets ---
KEY_ASSETS = {
    'equity': ['沪深300指数', '中证500指数', '创业板指数', '中证1000指数'],
    'bond': ['上证10年期国债指数(净价)', '国债指数', '30年国债指数'],
    'commodity': ['SGE黄金999', '有色金属指数', '螺纹钢指数', '能源化工指数'],
    'other': ['中证全指证券公司指数', '豆粕指数']
}


# =============================================================================
# Data Loading
# =============================================================================

def load_and_process_data():
    """
    Load and preprocess data
    
    Returns:
        df_monthly_ret: Monthly asset returns
        df_factors: Main factors (Level + Change)
        df_sub_factors: Sub-factors (if available)
    """
    print("=" * 70)
    print("Data Loading")
    print("=" * 70)
    
    # 1. Load daily index returns
    if not os.path.exists(INDEX_RETURNS_FILE):
        raise FileNotFoundError(f"File not found: {INDEX_RETURNS_FILE}")
    
    df_daily_ret = pd.read_csv(INDEX_RETURNS_FILE, index_col=0, parse_dates=True)
    print(f"[Data] Daily returns: {len(df_daily_ret.columns)} assets")
    
    # 2. Load main factors
    if not os.path.exists(MACRO_FACTORS_FILE):
        raise FileNotFoundError(f"File not found: {MACRO_FACTORS_FILE}")
    
    df_factors = pd.read_csv(MACRO_FACTORS_FILE, index_col=0, parse_dates=True)
    print(f"[Data] Main factors: {list(df_factors.columns)}")
    
    # 3. Load sub-factors (if exists)
    df_sub_factors = None
    if os.path.exists(SUB_FACTORS_FILE):
        df_sub_factors = pd.read_csv(SUB_FACTORS_FILE, index_col=0, parse_dates=True)
        print(f"[Data] Sub-factors: {len(df_sub_factors.columns)} columns")
    else:
        print("[Data] Sub-factors file not found, skipping")
    
    # 4. Convert to monthly returns
    df_monthly_ret = (1 + df_daily_ret).resample('ME').prod() - 1
    
    # 5. Align time index
    common_index = df_monthly_ret.index.intersection(df_factors.index)
    df_monthly_ret = df_monthly_ret.loc[common_index]
    df_factors = df_factors.loc[common_index]
    
    if df_sub_factors is not None:
        common_index_sub = common_index.intersection(df_sub_factors.index)
        df_sub_factors = df_sub_factors.loc[common_index_sub]
    
    print(f"[Data] Aligned: {len(common_index)} months ({common_index[0].date()} ~ {common_index[-1].date()})")
    
    return df_monthly_ret, df_factors, df_sub_factors


# =============================================================================
# Part 0: Overview - Correlation Matrix
# =============================================================================

def analyze_correlation_matrix(returns, factors):
    """
    计算资产收益率与所有宏观因子的 Spearman 相关性矩阵
    
    Args:
        returns: 月度资产收益率
        factors: 宏观因子数据 (Level + Change)
    
    Returns:
        df_corr: 相关性矩阵
    """
    print("\n" + "=" * 70)
    print("资产收益率与宏观因子的相关性分析 (Spearman)")
    print("=" * 70)
    
    # 合并数据
    df_merged = pd.concat([returns, factors], axis=1).dropna()
    
    # 计算 Spearman 相关性
    corr_results = {}
    
    for factor_col in factors.columns:
        corr_results[factor_col] = {}
        for asset_col in returns.columns:
            corr = df_merged[asset_col].corr(df_merged[factor_col], method='spearman')
            corr_results[factor_col][asset_col] = corr
    
    df_corr = pd.DataFrame(corr_results)
    
    # 保存结果
    df_corr.to_csv(os.path.join(OUTPUT_DIR, 'correlation_matrix_spearman.csv'),
                   encoding='utf-8-sig')
    print(f"相关性矩阵已保存")
    
    # 绘制热力图
    plot_correlation_heatmap(df_corr)
    
    return df_corr


def plot_correlation_heatmap(df_corr):
    """绘制资产-因子相关性热力图"""
    if df_corr is None or df_corr.empty:
        return
    
    # 按因子类型排序列
    level_cols = [c for c in df_corr.columns if '_Level' in c]
    change_cols = [c for c in df_corr.columns if '_Change' in c]
    ordered_cols = level_cols + change_cols
    df_plot = df_corr[ordered_cols]
    
    # 创建热力图
    fig, ax = plt.subplots(figsize=(16, max(12, len(df_plot) * 0.35)))
    
    sns.heatmap(df_plot, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                annot_kws={'size': 7}, vmin=-0.4, vmax=0.4,
                cbar_kws={'label': 'Spearman Correlation'})
    
    ax.set_title('资产收益率与宏观因子的相关性 (Spearman)\n同期相关性', fontsize=14)
    ax.set_xlabel('宏观因子', fontsize=11)
    ax.set_ylabel('资产', fontsize=11)
    
    # 旋转 x 轴标签
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap_all.png'), dpi=150)
    plt.close()
    
    # 绘制关键资产的精简版本
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_corr.index]
    
    if key_assets:
        df_key = df_plot.loc[key_assets]
        
        fig, ax = plt.subplots(figsize=(14, max(8, len(df_key) * 0.5)))
        
        sns.heatmap(df_key, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax,
                    annot_kws={'size': 9}, vmin=-0.4, vmax=0.4)
        
        ax.set_title('主要资产与宏观因子的相关性 (Spearman)', fontsize=14)
        ax.set_xlabel('宏观因子', fontsize=11)
        ax.set_ylabel('资产', fontsize=11)
        
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap_key_assets.png'), dpi=150)
        plt.close()


# =============================================================================
# Part 1: Level Factor Analysis (State-driven)
# =============================================================================

def analyze_level_regime(returns, factors, factor_name, n_quantiles=5):
    """
    Level Factor Regime Analysis: Quantile grouping -> Asset performance
    
    Args:
        returns: Monthly asset returns
        factors: Factor data
        factor_name: Level factor name (e.g., 'CN_Growth_Level')
        n_quantiles: Number of quantiles
    
    Returns:
        df_regime: Performance stats by quantile group
    """
    if factor_name not in factors.columns:
        return None
    
    factor = factors[factor_name].dropna()
    results = []
    
    for asset in returns.columns:
        # Align: F_t -> R_{t+1} (avoid look-ahead bias)
        df = pd.DataFrame({
            'Factor': factor,
            'Return_Next': returns[asset].shift(-1)
        }).dropna()
        
        if len(df) < 36:
            continue
        
        # Quantile grouping
        try:
            df['Quantile'] = pd.qcut(df['Factor'], n_quantiles, 
                                     labels=[f'Q{i+1}' for i in range(n_quantiles)],
                                     duplicates='drop')
        except ValueError:
            continue
        
        # Stats by group
        for q in df['Quantile'].unique():
            subset = df[df['Quantile'] == q]['Return_Next']
            
            results.append({
                'Asset': asset,
                'Factor': factor_name,
                'Quantile': q,
                'Mean_Return': subset.mean(),
                'Ann_Return': subset.mean() * 12,
                'Std_Return': subset.std(),
                'Sharpe': subset.mean() / subset.std() * np.sqrt(12) if subset.std() > 0 else np.nan,
                'Win_Rate': (subset > 0).mean(),
                'Count': len(subset)
            })
    
    df_results = pd.DataFrame(results)
    
    # Plot regime heatmap for key assets
    if not df_results.empty:
        plot_regime_heatmap(df_results, factor_name)
    
    return df_results


def plot_regime_heatmap(df_regime, factor_name):
    """Plot regime/quantile analysis heatmap"""
    if df_regime is None or df_regime.empty:
        return
    
    # Select key assets
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_regime['Asset'].unique()]
    
    if not key_assets:
        return
    
    df_plot = df_regime[df_regime['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(key_assets) * 0.4)))
    
    # 1. Annualized Return heatmap
    pivot_ret = df_plot.pivot(index='Asset', columns='Quantile', values='Ann_Return') * 100
    # Sort quantiles
    quantile_order = sorted(pivot_ret.columns, key=lambda x: int(x[1:]))
    pivot_ret = pivot_ret[quantile_order]
    
    sns.heatmap(pivot_ret, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=axes[0],
                annot_kws={'size': 8})
    axes[0].set_title(f'{factor_name} 分位数分析 - 年化收益率 (%)\n(Q1=最低, Q5=最高)')
    axes[0].set_xlabel('因子分位数')
    axes[0].set_ylabel('资产')
    
    # 2. Sharpe ratio heatmap
    pivot_sharpe = df_plot.pivot(index='Asset', columns='Quantile', values='Sharpe')
    pivot_sharpe = pivot_sharpe[quantile_order]
    
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1],
                annot_kws={'size': 8})
    axes[1].set_title(f'{factor_name} 分位数分析 - 夏普比率')
    axes[1].set_xlabel('因子分位数')
    axes[1].set_ylabel('资产')
    
    plt.tight_layout()
    safe_name = factor_name.replace('_Level', '').replace('_Change', '')
    plt.savefig(os.path.join(OUTPUT_DIR, f'regime_quantile_{safe_name}.png'), dpi=150)
    plt.close()


def analyze_level_quadrant(returns, factors, growth_col='CN_Growth_Level', inflation_col='CN_Inflation_Level'):
    """
    Growth-Inflation Quadrant Analysis
    
    Quadrants:
    - I.  Overheating (G+ I+): Growth > 0, Inflation > 0
    - II. Stagflation (G- I+): Growth < 0, Inflation > 0
    - III.Recession (G- I-): Growth < 0, Inflation < 0
    - IV. Recovery (G+ I-): Growth > 0, Inflation < 0
    
    Returns:
        df_quadrant: Asset performance by quadrant
    """
    print("\n" + "=" * 70)
    print("Level Factor - Quadrant Analysis (Growth x Inflation)")
    print("=" * 70)
    
    if growth_col not in factors.columns or inflation_col not in factors.columns:
        print(f"Cannot find {growth_col} or {inflation_col}, skipping quadrant analysis")
        return None
    
    df = factors[[growth_col, inflation_col]].copy()
    
    def get_quadrant(row):
        g, i = row[growth_col], row[inflation_col]
        if pd.isna(g) or pd.isna(i):
            return np.nan
        if g > 0 and i > 0:
            return 'I.Overheating(G+I+)'
        elif g <= 0 and i > 0:
            return 'II.Stagflation(G-I+)'
        elif g <= 0 and i <= 0:
            return 'III.Recession(G-I-)'
        else:
            return 'IV.Recovery(G+I-)'
    
    df['Quadrant'] = df.apply(get_quadrant, axis=1)
    
    # Calculate asset performance (T+1 return)
    results = []
    
    for asset in returns.columns:
        ret_next = returns[asset].shift(-1)
        
        merged = pd.DataFrame({
            'Quadrant': df['Quadrant'],
            'Return': ret_next
        }).dropna()
        
        for quad in merged['Quadrant'].unique():
            if pd.isna(quad):
                continue
            subset = merged[merged['Quadrant'] == quad]['Return']
            
            results.append({
                'Asset': asset,
                'Quadrant': quad,
                'Mean_Return': subset.mean(),
                'Ann_Return': subset.mean() * 12,
                'Std_Return': subset.std(),
                'Sharpe': subset.mean() / subset.std() * np.sqrt(12) if subset.std() > 0 else np.nan,
                'Win_Rate': (subset > 0).mean(),
                'Count': len(subset)
            })
    
    df_results = pd.DataFrame(results)
    
    # Save results
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'level_quadrant_analysis.csv'), 
                      encoding='utf-8-sig', index=False)
    print(f"Quadrant analysis saved")
    
    # Print key assets
    print("\nKey Assets Quadrant Performance (Annualized Return %):")
    print("-" * 70)
    
    key_assets = ['沪深300指数', '上证10年期国债指数(净价)', 'SGE黄金999', '螺纹钢指数']
    key_assets = [a for a in key_assets if a in returns.columns]
    
    if key_assets and not df_results.empty:
        pivot = df_results[df_results['Asset'].isin(key_assets)].pivot(
            index='Asset', columns='Quadrant', values='Ann_Return'
        ) * 100
        
        quad_order = ['IV.Recovery(G+I-)', 'I.Overheating(G+I+)', 'II.Stagflation(G-I+)', 'III.Recession(G-I-)']
        pivot = pivot[[c for c in quad_order if c in pivot.columns]]
        
        print(pivot.round(2).to_string())
    
    # Plot heatmap
    plot_quadrant_heatmap(df_results, key_assets)
    
    return df_results


def plot_quadrant_heatmap(df_quadrant, assets=None):
    """Plot quadrant heatmap"""
    if df_quadrant is None or df_quadrant.empty:
        return
    
    if assets:
        df_plot = df_quadrant[df_quadrant['Asset'].isin(assets)]
    else:
        df_plot = df_quadrant
    
    if df_plot.empty:
        return
    
    # Annualized return heatmap
    pivot = df_plot.pivot(index='Asset', columns='Quadrant', values='Ann_Return') * 100
    
    quad_order = ['IV.Recovery(G+I-)', 'I.Overheating(G+I+)', 'II.Stagflation(G-I+)', 'III.Recession(G-I-)']
    pivot = pivot[[c for c in quad_order if c in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Asset Annualized Return by Macro Quadrant (%)\n(T period Level Factor -> T+1 Return)')
    ax.set_xlabel('Macro Quadrant')
    ax.set_ylabel('Asset')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'level_quadrant_heatmap.png'), dpi=150)
    plt.close()
    
    # Sharpe ratio heatmap
    pivot_sharpe = df_plot.pivot(index='Asset', columns='Quadrant', values='Sharpe')
    pivot_sharpe = pivot_sharpe[[c for c in quad_order if c in pivot_sharpe.columns]]
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(pivot_sharpe) * 0.4)))
    sns.heatmap(pivot_sharpe, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax,
                annot_kws={'size': 9})
    ax.set_title('Asset Sharpe Ratio by Macro Quadrant\n(T period Level Factor -> T+1 Return)')
    ax.set_xlabel('Macro Quadrant')
    ax.set_ylabel('Asset')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'level_quadrant_sharpe.png'), dpi=150)
    plt.close()
    
    # 柱状图 - 不同象限下的资产收益率对比
    plot_quadrant_bar_chart(df_quadrant)


def plot_quadrant_bar_chart(df_quadrant):
    """绘制不同宏观象限下的资产平均月度收益率柱状图"""
    if df_quadrant is None or df_quadrant.empty:
        return
    
    # 选择关键资产
    key_assets = ['沪深300指数', '创业板指数', '中证全指证券公司指数', 
                  '上证10年期国债指数(净价)', 'SGE黄金999']
    key_assets = [a for a in key_assets if a in df_quadrant['Asset'].unique()]
    
    if not key_assets:
        return
    
    df_plot = df_quadrant[df_quadrant['Asset'].isin(key_assets)].copy()
    
    if df_plot.empty:
        return
    
    # 重命名象限为中文
    quadrant_map = {
        'IV.Recovery(G+I-)': '1. 复苏 (G+ I-)',
        'I.Overheating(G+I+)': '2. 过热 (G+ I+)',
        'II.Stagflation(G-I+)': '3. 滞胀 (G- I+)',
        'III.Recession(G-I-)': '4. 衰退 (G- I-)'
    }
    df_plot['Regime'] = df_plot['Quadrant'].map(quadrant_map)
    
    # 创建柱状图
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 设置柱状图位置
    x = np.arange(len(key_assets))
    width = 0.2
    
    regimes = ['1. 复苏 (G+ I-)', '2. 过热 (G+ I+)', '3. 滞胀 (G- I+)', '4. 衰退 (G- I-)']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 蓝、橙、绿、红
    
    for i, (regime, color) in enumerate(zip(regimes, colors)):
        regime_data = df_plot[df_plot['Regime'] == regime]
        values = []
        for asset in key_assets:
            asset_data = regime_data[regime_data['Asset'] == asset]
            if not asset_data.empty:
                values.append(asset_data['Mean_Return'].values[0])
            else:
                values.append(0)
        
        ax.bar(x + (i - 1.5) * width, values, width, label=regime, color=color)
    
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('平均月收益率', fontsize=12)
    ax.set_title('不同宏观象限下的资产平均月度收益率', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(key_assets, rotation=15, ha='right', fontsize=10)
    ax.legend(title='Regime', loc='upper left', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'level_quadrant_bar_chart.png'), dpi=150)
    plt.close()
    
    # 绘制年化收益率版本
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for i, (regime, color) in enumerate(zip(regimes, colors)):
        regime_data = df_plot[df_plot['Regime'] == regime]
        values = []
        for asset in key_assets:
            asset_data = regime_data[regime_data['Asset'] == asset]
            if not asset_data.empty:
                values.append(asset_data['Ann_Return'].values[0] * 100)  # 转为百分比
            else:
                values.append(0)
        
        ax.bar(x + (i - 1.5) * width, values, width, label=regime, color=color)
    
    ax.set_xlabel('Asset', fontsize=12)
    ax.set_ylabel('年化收益率 (%)', fontsize=12)
    ax.set_title('不同宏观象限下的资产年化收益率', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(key_assets, rotation=15, ha='right', fontsize=10)
    ax.legend(title='Regime', loc='upper left', fontsize=9)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'level_quadrant_bar_chart_annual.png'), dpi=150)
    plt.close()
    
    # 绘制更多资产的完整版本
    all_key_assets = []
    for category, assets in KEY_ASSETS.items():
        all_key_assets.extend(assets)
    all_key_assets = [a for a in all_key_assets if a in df_quadrant['Asset'].unique()]
    
    if len(all_key_assets) > 5:
        df_full = df_quadrant[df_quadrant['Asset'].isin(all_key_assets)].copy()
        df_full['Regime'] = df_full['Quadrant'].map(quadrant_map)
        
        fig, ax = plt.subplots(figsize=(18, 9))
        
        x_full = np.arange(len(all_key_assets))
        width_full = 0.18
        
        for i, (regime, color) in enumerate(zip(regimes, colors)):
            regime_data = df_full[df_full['Regime'] == regime]
            values = []
            for asset in all_key_assets:
                asset_data = regime_data[regime_data['Asset'] == asset]
                if not asset_data.empty:
                    values.append(asset_data['Ann_Return'].values[0] * 100)
                else:
                    values.append(0)
            
            ax.bar(x_full + (i - 1.5) * width_full, values, width_full, label=regime, color=color)
        
        ax.set_xlabel('Asset', fontsize=11)
        ax.set_ylabel('年化收益率 (%)', fontsize=11)
        ax.set_title('不同宏观象限下的资产年化收益率 (完整版)', fontsize=14)
        ax.set_xticks(x_full)
        ax.set_xticklabels(all_key_assets, rotation=45, ha='right', fontsize=9)
        ax.legend(title='Regime', loc='upper right', fontsize=9)
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'level_quadrant_bar_chart_full.png'), dpi=150)
        plt.close()


def analyze_level_beta_exposure(returns, factors):
    """
    Multi-factor Beta Exposure Analysis
    
    For each asset:
    R_t = a + b1*Growth_Level_t + b2*Inflation_Level_t + ... + e
    
    Note: Contemporaneous regression to explain return drivers
    """
    print("\n" + "=" * 70)
    print("Level Factor - Beta Exposure (Contemporaneous Regression)")
    print("=" * 70)
    
    level_cols = [c for c in factors.columns if '_Level' in c]
    
    if not level_cols:
        print("No Level factors found")
        return None
    
    results = []
    
    for asset in returns.columns:
        y = returns[asset]
        X = factors[level_cols].copy()
        X = sm.add_constant(X)
        
        data = pd.concat([y, X], axis=1).dropna()
        if len(data) < 36:
            continue
        
        y_clean = data.iloc[:, 0]
        X_clean = data.iloc[:, 1:]
        
        try:
            model = sm.OLS(y_clean, X_clean).fit()
            
            for col in level_cols:
                results.append({
                    'Asset': asset,
                    'Factor': col.replace('_Level', ''),
                    'Beta': model.params.get(col, np.nan),
                    'T_stat': model.tvalues.get(col, np.nan),
                    'P_value': model.pvalues.get(col, np.nan),
                    'Significant': abs(model.tvalues.get(col, 0)) > 2
                })
            
        except Exception:
            continue
    
    df_results = pd.DataFrame(results)
    
    # Save
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'level_beta_exposure.csv'),
                      encoding='utf-8-sig', index=False)
    
    # Plot Beta heatmap
    if not df_results.empty:
        pivot_beta = df_results.pivot(index='Asset', columns='Factor', values='Beta')
        
        # Select key assets
        key_assets = []
        for category, asset_list in KEY_ASSETS.items():
            key_assets.extend([a for a in asset_list if a in pivot_beta.index])
        
        if key_assets:
            pivot_beta = pivot_beta.loc[key_assets]
        
        fig, ax = plt.subplots(figsize=(10, max(6, len(pivot_beta) * 0.4)))
        sns.heatmap(pivot_beta, annot=True, fmt='.3f', cmap='coolwarm', center=0, ax=ax,
                    annot_kws={'size': 8})
        ax.set_title('Asset Beta Exposure to Level Factors (Contemporaneous)\nSignificance: |t| > 2')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'level_beta_heatmap.png'), dpi=150)
        plt.close()
    
    return df_results


# =============================================================================
# Part 2: Change Factor Analysis (Momentum-driven / Predictive)
# =============================================================================

def analyze_change_predictive_ic(returns, factors, lead=1):
    """
    Change Factor Predictive IC Analysis
    
    IC = corr(Change_t, R_{t+lead})
    IR = mean(IC) / std(IC)
    
    Args:
        returns: Monthly asset returns
        factors: Factor data
        lead: Prediction horizon (1=next month, 2=month after next)
    
    Returns:
        df_ic: IC statistics
    """
    print(f"\n" + "=" * 70)
    print(f"Change Factor - Predictive IC Analysis (T -> T+{lead})")
    print("=" * 70)
    
    change_cols = [c for c in factors.columns if '_Change' in c]
    
    if not change_cols:
        print("No Change factors found")
        return None
    
    results = []
    
    for factor_col in change_cols:
        factor = factors[factor_col]
        
        for asset in returns.columns:
            # IC: corr(F_t, R_{t+lead})
            ret_future = returns[asset].shift(-lead)
            
            df = pd.DataFrame({
                'Factor': factor,
                'Return': ret_future
            }).dropna()
            
            if len(df) < 36:
                continue
            
            # Rank IC (Spearman)
            rank_ic = df['Factor'].corr(df['Return'], method='spearman')
            # Pearson IC
            pearson_ic = df['Factor'].corr(df['Return'], method='pearson')
            
            # Rolling IC (for IR calculation)
            rolling_ic = df['Factor'].rolling(12).corr(df['Return'])
            ic_mean = rolling_ic.mean()
            ic_std = rolling_ic.std()
            ir = ic_mean / ic_std if ic_std > 0 else np.nan
            
            # IC significance t-test
            n = len(df)
            t_stat = rank_ic * np.sqrt(n - 2) / np.sqrt(1 - rank_ic**2) if abs(rank_ic) < 1 else np.nan
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2)) if not np.isnan(t_stat) else np.nan
            
            results.append({
                'Factor': factor_col.replace('_Change', ''),
                'Asset': asset,
                'Lead': lead,
                'Rank_IC': rank_ic,
                'Pearson_IC': pearson_ic,
                'IC_Mean_Rolling': ic_mean,
                'IC_Std_Rolling': ic_std,
                'IR': ir,
                'T_stat': t_stat,
                'P_value': p_value,
                'N_obs': n,
                'Significant': p_value < 0.05 if not np.isnan(p_value) else False
            })
    
    df_results = pd.DataFrame(results)
    
    # Save
    df_results.to_csv(os.path.join(OUTPUT_DIR, f'change_predictive_ic_T+{lead}.csv'),
                      encoding='utf-8-sig', index=False)
    
    # Print summary
    print(f"\nFactor Predictive Power Ranking (by |Rank IC|, T+{lead}):")
    print("-" * 70)
    
    if not df_results.empty:
        summary = df_results.groupby('Factor').agg({
            'Rank_IC': ['mean', 'std'],
            'IR': 'mean',
            'Significant': 'sum'
        }).round(4)
        summary.columns = ['IC_Mean', 'IC_Std', 'IR_Mean', 'Significant_Count']
        summary = summary.sort_values('IC_Mean', key=abs, ascending=False)
        print(summary.to_string())
    
    # Plot IC heatmap
    plot_ic_heatmap(df_results, lead)
    
    return df_results


def plot_ic_heatmap(df_ic, lead):
    """Plot IC heatmap"""
    if df_ic is None or df_ic.empty:
        return
    
    # Select key assets
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_ic['Asset'].unique()]
    
    df_plot = df_ic[df_ic['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    pivot = df_plot.pivot(index='Asset', columns='Factor', values='Rank_IC')
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                annot_kws={'size': 9})
    ax.set_title(f'Change Factor Predictive Rank IC (T -> T+{lead})\nPositive = Factor up -> Asset up')
    ax.set_xlabel('Factor')
    ax.set_ylabel('Asset')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'change_ic_heatmap_T+{lead}.png'), dpi=150)
    plt.close()


def analyze_change_predictive_regression(returns, factors, lead=1):
    """
    Change Factor Predictive Regression
    
    R_{t+lead} = a + b * Change_t + e
    
    Returns:
        df_reg: Regression results
    """
    print(f"\n" + "=" * 70)
    print(f"Change Factor - Predictive Regression (T -> T+{lead})")
    print("=" * 70)
    
    change_cols = [c for c in factors.columns if '_Change' in c]
    
    if not change_cols:
        return None
    
    results = []
    
    for factor_col in change_cols:
        for asset in returns.columns:
            y = returns[asset].shift(-lead)
            X = factors[[factor_col]].copy()
            X = sm.add_constant(X)
            
            data = pd.concat([y, X], axis=1).dropna()
            if len(data) < 36:
                continue
            
            y_clean = data.iloc[:, 0]
            X_clean = data.iloc[:, 1:]
            
            try:
                model = sm.OLS(y_clean, X_clean).fit()
                
                beta = model.params.get(factor_col, np.nan)
                t_stat = model.tvalues.get(factor_col, np.nan)
                r2 = model.rsquared
                resid_std = model.resid.std()
                
                # IR proxy = Beta / Residual Std
                ir_proxy = beta / resid_std if resid_std > 0 else np.nan
                
                results.append({
                    'Factor': factor_col.replace('_Change', ''),
                    'Asset': asset,
                    'Lead': lead,
                    'Beta': beta,
                    'T_stat': t_stat,
                    'R2': r2,
                    'IR_proxy': ir_proxy,
                    'N_obs': int(model.nobs),
                    'Significant': abs(t_stat) > 2
                })
            except Exception:
                continue
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, f'change_predictive_regression_T+{lead}.csv'),
                      encoding='utf-8-sig', index=False)
    
    # Plot regression results
    if not df_results.empty:
        plot_regression_heatmap(df_results, lead)
    
    return df_results


def plot_regression_heatmap(df_reg, lead):
    """Plot predictive regression Beta and T-stat heatmaps"""
    if df_reg is None or df_reg.empty:
        return
    
    # Select key assets
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_reg['Asset'].unique()]
    
    if not key_assets:
        return
    
    df_plot = df_reg[df_reg['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, max(6, len(key_assets) * 0.4)))
    
    # 1. Beta heatmap
    pivot_beta = df_plot.pivot(index='Asset', columns='Factor', values='Beta')
    sns.heatmap(pivot_beta, annot=True, fmt='.4f', cmap='coolwarm', center=0, ax=axes[0],
                annot_kws={'size': 8})
    axes[0].set_title(f'Change因子预测回归 Beta (T→T+{lead})')
    axes[0].set_xlabel('因子')
    axes[0].set_ylabel('资产')
    
    # 2. T-stat heatmap
    pivot_tstat = df_plot.pivot(index='Asset', columns='Factor', values='T_stat')
    sns.heatmap(pivot_tstat, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1],
                annot_kws={'size': 8}, vmin=-3, vmax=3)
    axes[1].set_title(f'Change因子预测回归 T统计量 (T→T+{lead})\n|t|>2 显著')
    axes[1].set_xlabel('因子')
    axes[1].set_ylabel('资产')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'change_regression_heatmap_T+{lead}.png'), dpi=150)
    plt.close()


# =============================================================================
# Part 3: Level + Change Joint Analysis
# =============================================================================

def analyze_joint_model(returns, factors, lead=1):
    """
    Level + Change Joint Predictive Model
    
    R_{t+lead} = a + b1*Level_t + b2*Change_t + e
    """
    print(f"\n" + "=" * 70)
    print(f"Level + Change Joint Model (T -> T+{lead})")
    print("=" * 70)
    
    # Pair Level and Change
    factor_pairs = []
    for col in factors.columns:
        if '_Level' in col:
            change_col = col.replace('_Level', '_Change')
            if change_col in factors.columns:
                factor_pairs.append((col, change_col))
    
    if not factor_pairs:
        print("No paired Level/Change factors found")
        return None
    
    results = []
    
    for level_col, change_col in factor_pairs:
        factor_name = level_col.replace('_Level', '')
        
        for asset in returns.columns:
            y = returns[asset].shift(-lead)
            X = factors[[level_col, change_col]].copy()
            X = sm.add_constant(X)
            
            data = pd.concat([y, X], axis=1).dropna()
            if len(data) < 36:
                continue
            
            y_clean = data.iloc[:, 0]
            X_clean = data.iloc[:, 1:]
            
            try:
                model = sm.OLS(y_clean, X_clean).fit()
                
                results.append({
                    'Factor': factor_name,
                    'Asset': asset,
                    'Lead': lead,
                    'Beta_Level': model.params.get(level_col, np.nan),
                    'T_Level': model.tvalues.get(level_col, np.nan),
                    'Beta_Change': model.params.get(change_col, np.nan),
                    'T_Change': model.tvalues.get(change_col, np.nan),
                    'R2': model.rsquared,
                    'Adj_R2': model.rsquared_adj,
                    'N_obs': int(model.nobs)
                })
            except Exception:
                continue
    
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(OUTPUT_DIR, f'joint_model_T+{lead}.csv'),
                      encoding='utf-8-sig', index=False)
    
    # Print key assets results
    print(f"\nKey Assets Level+Change Joint Model R2 (%):")
    print("-" * 70)
    
    key_assets = ['沪深300指数', '上证10年期国债指数(净价)', 'SGE黄金999']
    key_assets = [a for a in key_assets if a in df_results['Asset'].unique()]
    
    if key_assets and not df_results.empty:
        pivot = df_results[df_results['Asset'].isin(key_assets)].pivot(
            index='Asset', columns='Factor', values='R2'
        ) * 100
        print(pivot.round(2).to_string())
    
    # Plot joint model heatmap
    if not df_results.empty:
        plot_joint_model_heatmap(df_results, lead)
    
    return df_results


def plot_joint_model_heatmap(df_joint, lead):
    """Plot joint model R2 and coefficient heatmaps"""
    if df_joint is None or df_joint.empty:
        return
    
    # Select key assets
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_joint['Asset'].unique()]
    
    if not key_assets:
        return
    
    df_plot = df_joint[df_joint['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(key_assets) * 0.4)))
    
    # 1. R2 heatmap
    pivot_r2 = df_plot.pivot(index='Asset', columns='Factor', values='R2') * 100
    sns.heatmap(pivot_r2, annot=True, fmt='.2f', cmap='YlOrRd', ax=axes[0],
                annot_kws={'size': 8}, vmin=0)
    axes[0].set_title(f'Level+Change联合模型 R² (%) (T→T+{lead})')
    axes[0].set_xlabel('因子')
    axes[0].set_ylabel('资产')
    
    # 2. Level T-stat heatmap
    pivot_t_level = df_plot.pivot(index='Asset', columns='Factor', values='T_Level')
    sns.heatmap(pivot_t_level, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[1],
                annot_kws={'size': 8}, vmin=-3, vmax=3)
    axes[1].set_title(f'Level因子 T统计量 (联合模型)\n|t|>2 显著')
    axes[1].set_xlabel('因子')
    axes[1].set_ylabel('资产')
    
    # 3. Change T-stat heatmap
    pivot_t_change = df_plot.pivot(index='Asset', columns='Factor', values='T_Change')
    sns.heatmap(pivot_t_change, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=axes[2],
                annot_kws={'size': 8}, vmin=-3, vmax=3)
    axes[2].set_title(f'Change因子 T统计量 (联合模型)\n|t|>2 显著')
    axes[2].set_xlabel('因子')
    axes[2].set_ylabel('资产')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'joint_model_heatmap_T+{lead}.png'), dpi=150)
    plt.close()


def analyze_level_change_quadrant(returns, factors, factor_name='CN_Growth', lead=1):
    """
    Single Factor Level x Change Quadrant Analysis
    
    Quadrants:
    - High+Improving: Level > 0, Change > 0
    - High+Deteriorating: Level > 0, Change < 0
    - Low+Improving: Level < 0, Change > 0
    - Low+Deteriorating: Level < 0, Change < 0
    """
    level_col = f'{factor_name}_Level'
    change_col = f'{factor_name}_Change'
    
    if level_col not in factors.columns or change_col not in factors.columns:
        return None
    
    df = factors[[level_col, change_col]].copy()
    
    def get_lc_quadrant(row):
        l, c = row[level_col], row[change_col]
        if pd.isna(l) or pd.isna(c):
            return np.nan
        if l > 0 and c > 0:
            return 'High+Improving'
        elif l > 0 and c <= 0:
            return 'High+Deteriorating'
        elif l <= 0 and c > 0:
            return 'Low+Improving'
        else:
            return 'Low+Deteriorating'
    
    df['LC_Quadrant'] = df.apply(get_lc_quadrant, axis=1)
    
    results = []
    
    for asset in returns.columns:
        ret_future = returns[asset].shift(-lead)
        
        merged = pd.DataFrame({
            'Quadrant': df['LC_Quadrant'],
            'Return': ret_future
        }).dropna()
        
        for quad in merged['Quadrant'].unique():
            if pd.isna(quad):
                continue
            subset = merged[merged['Quadrant'] == quad]['Return']
            
            results.append({
                'Factor': factor_name,
                'Asset': asset,
                'Quadrant': quad,
                'Mean_Return': subset.mean(),
                'Ann_Return': subset.mean() * 12,
                'Win_Rate': (subset > 0).mean(),
                'Count': len(subset)
            })
    
    return pd.DataFrame(results)


def plot_level_change_quadrant(df_lc_quadrant, factor_name, assets=None):
    """Plot Level x Change Quadrant Heatmap"""
    if df_lc_quadrant is None or df_lc_quadrant.empty:
        return
    
    if assets:
        df_plot = df_lc_quadrant[df_lc_quadrant['Asset'].isin(assets)]
    else:
        key_assets = []
        for category, asset_list in KEY_ASSETS.items():
            key_assets.extend(asset_list)
        key_assets = [a for a in key_assets if a in df_lc_quadrant['Asset'].unique()]
        df_plot = df_lc_quadrant[df_lc_quadrant['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    pivot = df_plot.pivot(index='Asset', columns='Quadrant', values='Ann_Return') * 100
    
    quad_order = ['Low+Improving', 'High+Improving', 'High+Deteriorating', 'Low+Deteriorating']
    pivot = pivot[[c for c in quad_order if c in pivot.columns]]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.4)))
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax,
                annot_kws={'size': 9})
    ax.set_title(f'{factor_name} Level x Change Quadrant Annualized Return (%)\n(T Factor -> T+1 Return)')
    ax.set_xlabel('Quadrant')
    ax.set_ylabel('Asset')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'level_change_quadrant_{factor_name}.png'), dpi=150)
    plt.close()


# =============================================================================
# Part 4: Sub-factor Analysis
# =============================================================================

def analyze_sub_factors(returns, df_sub_factors, parent_factor='CN_Growth'):
    """
    Sub-factor Analysis: Analyze predictive power of each sub-component
    
    Args:
        returns: Asset returns
        df_sub_factors: Sub-factor data
        parent_factor: Parent factor name (for filtering)
    """
    if df_sub_factors is None or df_sub_factors.empty:
        print(f"No sub-factor data available")
        return None
    
    # Find sub-factors for this parent
    sub_cols = [c for c in df_sub_factors.columns if parent_factor in c]
    
    if not sub_cols:
        print(f"No sub-factors found for {parent_factor}")
        return None
    
    print(f"\nAnalyzing {parent_factor} sub-factors: {sub_cols}")
    
    results = []
    
    for col in sub_cols:
        factor = df_sub_factors[col]
        
        # Determine if Level or Change
        is_change = '_Change' in col
        
        for asset in returns.columns:
            # T -> T+1 prediction
            ret_next = returns[asset].shift(-1)
            
            df = pd.DataFrame({
                'Factor': factor,
                'Return': ret_next
            }).dropna()
            
            if len(df) < 24:
                continue
            
            # IC
            ic = df['Factor'].corr(df['Return'], method='spearman')
            
            # Regression
            X = sm.add_constant(df['Factor'])
            y = df['Return']
            
            try:
                model = sm.OLS(y, X).fit()
                beta = model.params.iloc[1]
                t_stat = model.tvalues.iloc[1]
            except Exception:
                beta, t_stat = np.nan, np.nan
            
            results.append({
                'Parent_Factor': parent_factor,
                'Sub_Factor': col,
                'Type': 'Change' if is_change else 'Level',
                'Asset': asset,
                'IC': ic,
                'Beta': beta,
                'T_stat': t_stat,
                'Significant': abs(t_stat) > 2 if not np.isnan(t_stat) else False
            })
    
    return pd.DataFrame(results)


def analyze_all_sub_factors(returns, df_sub_factors):
    """
    Analyze all sub-factors
    """
    if df_sub_factors is None or df_sub_factors.empty:
        print("No sub-factor data, skipping sub-factor analysis")
        return None
    
    print("\n" + "=" * 70)
    print("Sub-factor Analysis")
    print("=" * 70)
    
    all_results = []
    
    # Identify parent factors
    parent_factors = set()
    for col in df_sub_factors.columns:
        for pf in ['CN_Growth', 'CN_Inflation', 'US_Growth', 'US_Inflation', 
                   'CN_RiskAppetite', 'CN_Monetary']:
            if pf in col:
                parent_factors.add(pf)
                break
    
    for parent in parent_factors:
        df_sub = analyze_sub_factors(returns, df_sub_factors, parent)
        if df_sub is not None and not df_sub.empty:
            all_results.append(df_sub)
            df_sub.to_csv(
                os.path.join(OUTPUT_DIR, f'sub_factor_analysis_{parent}.csv'),
                encoding='utf-8-sig', index=False
            )
    
    if all_results:
        df_all = pd.concat(all_results, ignore_index=True)
        df_all.to_csv(os.path.join(OUTPUT_DIR, 'sub_factor_analysis_all.csv'),
                      encoding='utf-8-sig', index=False)
        
        # Print summary
        print("\nSub-factor Predictive Power Summary (by |IC|):")
        print("-" * 70)
        
        summary = df_all.groupby(['Parent_Factor', 'Sub_Factor', 'Type']).agg({
            'IC': 'mean',
            'Significant': 'sum'
        }).round(4)
        summary = summary.sort_values('IC', key=abs, ascending=False)
        print(summary.head(20).to_string())
        
        # Plot sub-factor IC heatmaps
        plot_sub_factor_heatmaps(df_all)
        
        return df_all
    
    return None


def plot_sub_factor_heatmaps(df_sub_all):
    """Plot sub-factor analysis heatmaps"""
    if df_sub_all is None or df_sub_all.empty:
        return
    
    # Select key assets
    key_assets = []
    for category, assets in KEY_ASSETS.items():
        key_assets.extend(assets)
    key_assets = [a for a in key_assets if a in df_sub_all['Asset'].unique()]
    
    if not key_assets:
        return
    
    df_plot = df_sub_all[df_sub_all['Asset'].isin(key_assets)]
    
    if df_plot.empty:
        return
    
    # 1. Top Sub-factors IC Heatmap (by |IC| mean)
    sub_factor_ic = df_plot.groupby('Sub_Factor')['IC'].mean().abs().sort_values(ascending=False)
    top_sub_factors = sub_factor_ic.head(15).index.tolist()
    
    df_top = df_plot[df_plot['Sub_Factor'].isin(top_sub_factors)]
    
    if not df_top.empty:
        pivot_ic = df_top.pivot(index='Asset', columns='Sub_Factor', values='IC')
        # Reorder columns by IC strength
        pivot_ic = pivot_ic[top_sub_factors]
        
        fig, ax = plt.subplots(figsize=(16, max(8, len(key_assets) * 0.5)))
        sns.heatmap(pivot_ic, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                    annot_kws={'size': 7})
        ax.set_title('Top 15 分项因子预测能力 (Rank IC, T→T+1)\n按 |IC| 均值排序')
        ax.set_xlabel('分项因子')
        ax.set_ylabel('资产')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 'sub_factor_top15_ic_heatmap.png'), dpi=150)
        plt.close()
    
    # 2. Separate plots for Level and Change sub-factors
    for factor_type in ['Level', 'Change']:
        df_type = df_plot[df_plot['Type'] == factor_type]
        
        if df_type.empty:
            continue
        
        # Get top 10 sub-factors of this type
        type_ic = df_type.groupby('Sub_Factor')['IC'].mean().abs().sort_values(ascending=False)
        top_type_factors = type_ic.head(10).index.tolist()
        
        df_type_top = df_type[df_type['Sub_Factor'].isin(top_type_factors)]
        
        if not df_type_top.empty:
            pivot_type = df_type_top.pivot(index='Asset', columns='Sub_Factor', values='IC')
            pivot_type = pivot_type[top_type_factors]
            
            fig, ax = plt.subplots(figsize=(14, max(6, len(key_assets) * 0.4)))
            sns.heatmap(pivot_type, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
                        annot_kws={'size': 8})
            ax.set_title(f'Top 10 {factor_type} 分项因子预测能力 (Rank IC)')
            ax.set_xlabel('分项因子')
            ax.set_ylabel('资产')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, f'sub_factor_top10_{factor_type}_ic_heatmap.png'), dpi=150)
            plt.close()
    
    # 3. Summary bar chart - Top factors by mean IC
    fig, ax = plt.subplots(figsize=(14, 8))
    
    summary_ic = df_plot.groupby(['Sub_Factor', 'Type'])['IC'].mean().reset_index()
    summary_ic['abs_IC'] = summary_ic['IC'].abs()
    summary_ic = summary_ic.sort_values('abs_IC', ascending=True).tail(20)
    
    colors = ['#2ecc71' if ic > 0 else '#e74c3c' for ic in summary_ic['IC']]
    bars = ax.barh(range(len(summary_ic)), summary_ic['IC'], color=colors, alpha=0.8)
    ax.set_yticks(range(len(summary_ic)))
    ax.set_yticklabels([f"{row['Sub_Factor']} ({row['Type'][0]})" 
                        for _, row in summary_ic.iterrows()], fontsize=9)
    ax.set_xlabel('平均 Rank IC')
    ax.set_title('Top 20 分项因子预测能力排名 (按 |IC| 排序)\n正值=因子上升时资产上涨, L=Level, C=Change')
    ax.axvline(x=0, color='black', linewidth=0.5)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'sub_factor_ranking_bar.png'), dpi=150)
    plt.close()


# =============================================================================
# Summary Report
# =============================================================================

def generate_summary_report(returns, factors, df_sub_factors=None):
    """Generate comprehensive analysis report"""
    
    print("\n" + "=" * 70)
    print("Macro Factor vs Asset Return Analysis Report")
    print("=" * 70)
    
    # 0. Overview - Correlation Matrix
    print("\n" + "-" * 70)
    print("Part 0: Correlation Overview")
    print("-" * 70)
    
    df_corr = analyze_correlation_matrix(returns, factors)
    
    # 1. Level Factor Analysis
    print("\n" + "-" * 70)
    print("Part 1: Level Factor Analysis (State-driven)")
    print("-" * 70)
    
    # Quadrant Analysis (Growth x Inflation)
    df_quadrant = analyze_level_quadrant(returns, factors)
    
    # Beta Exposure Analysis
    df_beta = analyze_level_beta_exposure(returns, factors)
    
    # Quantile Analysis (for key factors)
    for factor_name in ['CN_Growth_Level', 'CN_Inflation_Level']:
        if factor_name in factors.columns:
            df_regime = analyze_level_regime(returns, factors, factor_name, n_quantiles=5)
            if df_regime is not None:
                df_regime.to_csv(
                    os.path.join(OUTPUT_DIR, f'level_regime_{factor_name}.csv'),
                    encoding='utf-8-sig', index=False
                )
                print(f"  {factor_name} quantile analysis saved")
    
    # 2. Change Factor Analysis
    print("\n" + "-" * 70)
    print("Part 2: Change Factor Analysis (Momentum-driven)")
    print("-" * 70)
    
    # IC Analysis (T+1 and T+2)
    df_ic_t1 = analyze_change_predictive_ic(returns, factors, lead=1)
    df_ic_t2 = analyze_change_predictive_ic(returns, factors, lead=2)
    
    # Predictive Regression
    df_reg_t1 = analyze_change_predictive_regression(returns, factors, lead=1)
    df_reg_t2 = analyze_change_predictive_regression(returns, factors, lead=2)
    
    # 3. Joint Analysis
    print("\n" + "-" * 70)
    print("Part 3: Level + Change Joint Analysis")
    print("-" * 70)
    
    df_joint = analyze_joint_model(returns, factors, lead=1)
    
    # Level x Change Quadrant
    for factor_name in ['CN_Growth', 'CN_Inflation']:
        df_lc = analyze_level_change_quadrant(returns, factors, factor_name, lead=1)
        if df_lc is not None:
            df_lc.to_csv(
                os.path.join(OUTPUT_DIR, f'level_change_quadrant_{factor_name}.csv'),
                encoding='utf-8-sig', index=False
            )
            plot_level_change_quadrant(df_lc, factor_name)
            print(f"  {factor_name} Level x Change quadrant analysis saved")
    
    # 4. Sub-factor Analysis (if available)
    print("\n" + "-" * 70)
    print("Part 4: Sub-factor Analysis")
    print("-" * 70)
    
    df_sub_all = analyze_all_sub_factors(returns, df_sub_factors)
    
    print("\n" + "=" * 70)
    print("Analysis Complete! Results saved to:", OUTPUT_DIR)
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

def main():
    """Main function"""
    try:
        # Load data
        returns, factors, sub_factors = load_and_process_data()
        
        if returns.empty or factors.empty:
            print("Data is empty, cannot analyze.")
            return
        
        # Generate report
        generate_summary_report(returns, factors, sub_factors)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
