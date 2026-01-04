# -*- coding: utf-8 -*-
"""
@Project: ETF Strategy
@File   : Study_macro_factor_exposure.py
@Description: 
    研究各类资产指数 (Index) 对合成宏观因子 (Macro Factors) 的暴露关系。
    
    适配 V2 版本因子 (Level + Change 双维度):
    - Level (绝对水位): 当前处于什么位置
    - Change (边际变化): 趋势在改善还是恶化
    
    因子列表:
    - CN_Growth_Level, CN_Growth_Change
    - CN_Inflation_Level, CN_Inflation_Change
    - US_Growth_Level, US_Growth_Change
    - US_Inflation_Level, US_Inflation_Change
    - CN_RiskAppetite_Level, CN_RiskAppetite_Change
    - CN_Monetary_Level, CN_Monetary_Change
    
    研究方法:
    1. 数据对齐: 将日频指数收益率转换为月频收益率，与月频宏观因子对齐。
    2. 相关性分析 (Correlation): 计算资产与因子的相关系数热力图。
    3. 单因子回归分析 (Single-Factor Regression):
       - 对每个因子、每个资产做回归。
       - 使用预测框架: 当期宏观因子 F_t 回归未来第2个月资产收益 R_{t+2}。
       - 输出 Beta 矩阵 & t 值矩阵，并画热力图。
    4. 宏观情景分析 (Regime Analysis): 
       - 根据 CN_Growth_Level / CN_Inflation_Level 象限划分 Regime。
       - 统计各资产在不同象限下的平均收益率和胜率。
    5. 因子 IR 代理 (Factor IR Proxy):
       - 对各因子单因子回归 F_t → R_{t+2}。
       - 用 Beta / Residual Std 作为因子对资产/四象限组合的 IR 代理。
       - 对结果画条形图。
"""

import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm

# --- 路径配置 ---
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

DATA_DIR = os.path.join(ROOT, 'data')
OUTPUT_DIR = os.path.join(ROOT, 'analysis_results')

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

INDEX_RETURNS_FILE = os.path.join(DATA_DIR, 'index_daily_simple_returns.csv')
MACRO_FACTORS_FILE = os.path.join(DATA_DIR, 'all_macro_factors.csv')  # V2 版本因子
SUB_FACTORS_FILE = os.path.join(DATA_DIR, 'all_macro_sub_factors.csv')
MIMICKING_FACTORS_FILE = os.path.join(DATA_DIR, 'macro_factors_mimicking.csv')

# --- 绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文
plt.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# --- 资产池配置 ---
# 说明：
# - 这里预先填好当前数据文件中的全部资产名称，后续你只需要改下面这个 dict / key
# - main() 会根据 ASSET_UNIVERSE_KEY 过滤 returns，只分析选中的资产
ASSET_UNIVERSE_KEY = 'CUSTOM'  # 可改为 'CUSTOM' 或你自己新增的 key

ASSET_UNIVERSES = {
    'ALL': [
        '沪深300指数',
        '上证50指数',
        '中证小盘500指数',
        '创业板指数',
        '中证1000指数',
        '上证科创板50成份指数',
        '标普中国A股大盘红利低波50指数',
        '中证红利质量指数',
        '中证全指证券公司指数',
        '中证主要消费指数',
        '中证全指半导体产品与设备指数',
        '中证银行指数',
        '中证军工指数',
        '沪深300医药卫生指数',
        '中证申万有色金属指数',
        '中证煤炭指数',
        '中证全指房地产指数',
        '中证全指电力与电网指数',
        '国证半导体芯片',
        '中证机器人指数',
        '中证人工智能主题指数',
        '中证细分化工产业主题指数',
        '沪深300非银行金融指数',
        '国证新能源车电池指数',
        '中证光伏产业指数',
        '中证创新药产业指数',
        '中证全指通信设备指数',
        '中证动漫游戏指数',
        '中证金融科技主题指数',
        '上证5年期国债指数',
        '上证10年期国债指数(净价)',
        '中债-30年期国债指数',
        '上证城投债指数',
        '中债-中高等级公司债利差因子指数',
        '上证基准做市公司债指数',
        '中证可转债及可交换债券指数',
        '上海期货交易所有色金属期货价格指数',
        '大连商品交易所豆粕期货价格指数',
        '易盛能化A',
        'SGE黄金999',
        '中证短融指数',
    ],

    # 示例：你后续可以在这里自定义任意资产组合
    'CUSTOM': [
         '沪深300指数',
        '中证小盘500指数',
        '中证1000指数',
        '上证科创板50成份指数',
        '创业板指数',     
        '标普中国A股大盘红利低波50指数',
        '中证红利质量指数',
        '上证5年期国债指数',
        '上证10年期国债指数(净价)',
        '中债-30年期国债指数',
        '中债-中高等级公司债利差因子指数',
        '上证基准做市公司债指数',
        '中证可转债及可交换债券指数',
        '上海期货交易所有色金属期货价格指数',
        '大连商品交易所豆粕期货价格指数',
        '易盛能化A',
        'SGE黄金999',
    ],
}

def filter_returns_by_universe(returns: pd.DataFrame, universe_key: str) -> pd.DataFrame:
    if universe_key not in ASSET_UNIVERSES:
        print(f"[Assets] 未找到资产池 key={universe_key}，将使用 returns 全部列")
        return returns

    asset_list = ASSET_UNIVERSES[universe_key]
    if not asset_list:
        print(f"[Assets] 资产池 key={universe_key} 为空，将使用 returns 全部列")
        return returns

    keep = [a for a in asset_list if a in returns.columns]
    missing = [a for a in asset_list if a not in returns.columns]
    if missing:
        print(f"[Assets] 资产池中有 {len(missing)} 个资产不在数据里，将忽略: {missing}")
    if not keep:
        print(f"[Assets] 资产池 key={universe_key} 在数据中命中为空，将使用 returns 全部列")
        return returns

    print(f"[Assets] 使用资产池 {universe_key}: {len(keep)} / {returns.shape[1]} assets")
    return returns[keep].copy()


def load_and_process_data():
    """
    加载并预处理数据
    返回: (df_monthly_returns, df_monthly_factors)
    """
    print("正在加载数据...")
    
    # 1. 加载指数日频收益率
    if not os.path.exists(INDEX_RETURNS_FILE):
        raise FileNotFoundError(f"找不到文件: {INDEX_RETURNS_FILE}")
    
    df_daily_ret = pd.read_csv(INDEX_RETURNS_FILE, index_col=0, parse_dates=True)
    
    # 2. 加载宏观因子 (V2 版本: 月频 Level + Change)
    if not os.path.exists(MACRO_FACTORS_FILE):
        raise FileNotFoundError(f"找不到文件: {MACRO_FACTORS_FILE}")
    
    df_monthly_factors = pd.read_csv(MACRO_FACTORS_FILE, index_col=0, parse_dates=True)

    # 3. Load sub-factors (if exists)
    if not os.path.exists(SUB_FACTORS_FILE):
        raise FileNotFoundError(f"找不到文件: {SUB_FACTORS_FILE}")
    df_monthly_sub_factors = pd.read_csv(SUB_FACTORS_FILE, index_col=0, parse_dates=True)

    # 4. Load MIMICKING-factors (if exists)
    if not os.path.exists(MIMICKING_FACTORS_FILE):
        raise FileNotFoundError(f"找不到文件: {MIMICKING_FACTORS_FILE}")
    df_monthly_mimicking_factors = pd.read_csv(MIMICKING_FACTORS_FILE, index_col=0, parse_dates=True)

    # 4. 转换为月频数据
    print("正在转换为月频数据...")
    
    # 指数: 月度累计简单收益率 (1+r).prod() - 1
    df_monthly_ret = (1 + df_daily_ret).resample('ME').prod() - 1
  
    # 因子: 已经是月频数据，无需转换
    print(f"因子列: {list(df_monthly_factors.columns)}")
    
    # 5. 对齐时间段
    common_index = df_monthly_ret.index.intersection(df_monthly_factors.index)
    df_monthly_ret = df_monthly_ret.loc[common_index]
    df_monthly_factors = df_monthly_factors.loc[common_index]
    df_monthly_sub_factors = df_monthly_sub_factors.loc[common_index]
    print(f"数据对齐完成: {len(common_index)} 个月 ({common_index[0].date()} 至 {common_index[-1].date()})")
    # 去除因子全为空的行
    df_monthly_factors = df_monthly_factors.dropna(how='all')
    df_monthly_sub_factors = df_monthly_sub_factors.dropna(how='all')
    common_index = df_monthly_ret.index.intersection(df_monthly_factors.index)
    
    df_monthly_ret = df_monthly_ret.loc[common_index]
    df_monthly_factors = df_monthly_factors.loc[common_index]
    df_monthly_sub_factors = df_monthly_sub_factors.loc[common_index]
    print(f"数据对齐完成: {len(common_index)} 个月 ({common_index[0].date()} 至 {common_index[-1].date()})")
    
    return df_monthly_ret, df_monthly_factors, df_monthly_sub_factors,df_monthly_mimicking_factors



def analyze_correlations_and_single_factor_regressions(returns, factors, T_shift=0):
    """
    分析相关性（时间序列层面的资产 vs 因子相关系数）
    同时计算 Pearson 与 Spearman：使用 T+T_shift 期资产收益与当期因子值的相关性。
    T_shift=0: 当期相关性 (同步)
    T_shift=1: 预测性相关性 (因子领先1期)
    """
    print(f"\n正在进行相关性分析 (T_shift={T_shift})...")
    
    correlations_pearson = pd.DataFrame(index=returns.columns, columns=factors.columns)
    correlations_spearman = pd.DataFrame(index=returns.columns, columns=factors.columns)
    
    # 如果 T_shift != 0，对收益率进行 shift
    # shift(-k) 表示将未来的数据平移到当前，用于对齐 F_t 和 R_{t+k}
    if T_shift != 0:
        shifted_returns = returns.shift(-T_shift)
    else:
        shifted_returns = returns
    
    for asset in returns.columns:
        for factor in factors.columns:
            valid_data = pd.concat([shifted_returns[asset], factors[factor]], axis=1).dropna()
            if len(valid_data) < 12:
                correlations_pearson.loc[asset, factor] = np.nan
                correlations_spearman.loc[asset, factor] = np.nan
            else:
                correlations_pearson.loc[asset, factor] = valid_data.corr(method='pearson').iloc[0, 1]
                correlations_spearman.loc[asset, factor] = valid_data.corr(method='spearman').iloc[0, 1]

    correlations_pearson = correlations_pearson.astype(float)
    correlations_spearman = correlations_spearman.astype(float)

    # 保存
    suffix = f"_T+{T_shift}" if T_shift > 0 else ""
    corr_file_pearson = os.path.join(OUTPUT_DIR, f'exposure_correlations_pearson{suffix}.csv')
    corr_file_spearman = os.path.join(OUTPUT_DIR, f'exposure_correlations_spearman{suffix}.csv')
    correlations_pearson.to_csv(corr_file_pearson, encoding='utf-8-sig')
    correlations_spearman.to_csv(corr_file_spearman, encoding='utf-8-sig')
    print(f"Pearson 相关性矩阵已保存: {corr_file_pearson}")
    print(f"Spearman 相关性矩阵已保存: {corr_file_spearman}")
    
    # 绘图
    fig_corr = plt.figure(figsize=(12, 10))
    sns.heatmap(correlations_pearson, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    title_suffix = f" (T+{T_shift})" if T_shift > 0 else ""
    plt.title(f'资产收益率与宏观因子的相关性 (Pearson){title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'exposure_heatmap_pearson{suffix}.png'))

    fig_corr2 = plt.figure(figsize=(12, 10))
    sns.heatmap(correlations_spearman, annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f'资产收益率与宏观因子的相关性 (Spearman){title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'exposure_heatmap_spearman{suffix}.png'))


    """
    分析相关性（时间序列层面的资产 vs 因子相关系数）
    使用 T+T_shift 期资产收益与当期因子水平的 pearson 相关性。
    T_shift=0: 当期相关性 (同步)
    T_shift=1: 预测性相关性 (因子领先1期)
    """
    print(f"\n正在进行相关性分析 (T_shift={T_shift})...")


    numeric_factor_cols = factors.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_factor_cols:
        print("未找到数值型因子列，跳过回归分析。")
        return None, None
    
    beta_mat = pd.DataFrame(index=returns.columns, columns=numeric_factor_cols, dtype=float)
    t_mat = pd.DataFrame(index=returns.columns, columns=numeric_factor_cols, dtype=float)

    # 如果 T_shift != 0，对收益率进行 shift
    # shift(-k) 表示将未来的数据平移到当前，用于对齐 F_t 和 R_{t+k}
    shifted_returns = returns.shift(-T_shift) if T_shift != 0 else returns

    for factor_col in numeric_factor_cols:
        F = factors[factor_col]

        for asset in returns.columns:
            # T+T_shift 单月收益（已按需 shift）
            y = shifted_returns[asset]
            X = pd.DataFrame({'factor': F})
            X = sm.add_constant(X)
            
            data = pd.concat([y, X], axis=1).dropna()
            if len(data) < 24:
                continue
            
            y_clean = data.iloc[:, 0]
            X_clean = data.iloc[:, 1:]
            
            try:
                model = sm.OLS(y_clean, X_clean).fit()
                beta = model.params.get('factor', np.nan)
                tval = model.tvalues.get('factor', np.nan)
                
                beta_mat.loc[asset, factor_col] = beta
                t_mat.loc[asset, factor_col] = tval

            except Exception:
                continue
    
    suffix = f"_T+{T_shift}" if T_shift > 0 else ""
    beta_file = os.path.join(OUTPUT_DIR, f'single_factor_betas{suffix}.csv')
    t_file = os.path.join(OUTPUT_DIR, f'single_factor_tvalues{suffix}.csv')

    beta_mat.to_csv(beta_file, encoding='utf-8-sig')
    t_mat.to_csv(t_file, encoding='utf-8-sig')

    print(f"单因子 Beta 矩阵已保存: {beta_file}")
    print(f"单因子 t 值矩阵已保存: {t_file}")

    

    # # 画 Beta 热力图
    # fig_beta = plt.figure(figsize=(14, 10))
    # sns.heatmap(beta_mat.astype(float), annot=False, cmap='coolwarm', center=0)
    # title_suffix = f" (T+{T_shift})" if T_shift > 0 else ""
    # plt.title(f'单因子回归 Beta (F_t -> R_{{t+{T_shift}}}){title_suffix}')
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, f'single_factor_beta_heatmap{suffix}.png'))

    
    # 画 t 值热力图
    fig_t = plt.figure(figsize=(14, 10))
    sns.heatmap(t_mat.astype(float), annot=True, cmap='coolwarm', center=0)
    plt.title(f'单因子回归 t 值 (F_t -> R_{{t+{T_shift}}}){title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'single_factor_tvalue_heatmap{suffix}.png'))


    # 筛选：Pearson 相关系数 > 0.1 且 t-stat > 1.5（为包含负相关/负t，这里用绝对值）
    try:
        corr_for_reg = correlations_pearson[numeric_factor_cols]
    except Exception:
        corr_for_reg = correlations_pearson

    corr_long = corr_for_reg.stack(dropna=False).rename('PearsonCorr')
    t_long = t_mat.stack(dropna=False).rename('Tstat')
    screen_df = pd.concat([corr_long, t_long], axis=1).reset_index()
    screen_df.columns = ['Asset', 'Factor', 'PearsonCorr', 'Tstat']
    screen_df = screen_df.dropna(subset=['PearsonCorr', 'Tstat'], how='any')
    screen_df['AbsPearsonCorr'] = screen_df['PearsonCorr'].abs()
    screen_df['AbsTstat'] = screen_df['Tstat'].abs()
    screen_df = screen_df[(screen_df['AbsPearsonCorr'] >= 0.1) & (screen_df['AbsTstat'] >= 1.5)].copy()
    screen_df = screen_df.sort_values(['AbsPearsonCorr', 'AbsTstat'], ascending=False)

    print(f"\n筛选结果 (|Pearson|>=0.1 & |t|>=1.5) 命中: {len(screen_df)} 条")
    if not screen_df.empty:
        print(screen_df[['Asset', 'Factor', 'PearsonCorr', 'Tstat']].head(100).to_string(index=False))

    excel_path = os.path.join(OUTPUT_DIR, f'screen_pearson_corr_ge_0.1_tstat_ge_1.5{suffix}.xlsx')
    try:
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            screen_df[['Asset', 'Factor', 'PearsonCorr', 'Tstat']].to_excel(writer, sheet_name='screen', index=False)
    except Exception:
        with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
            screen_df[['Asset', 'Factor', 'PearsonCorr', 'Tstat']].to_excel(writer, sheet_name='screen', index=False)
    print(f"筛选结果已导出: {excel_path}")

    plt.show()
    # plt.close()
    
    return beta_mat, t_mat


def analyze_regimes(returns, factors, T_shift=1):
    """
    非线性分组分析 (Regime Analysis) - 批量处理
    对每个因子和每个资产进行固定阈值分组分析（基于因子数值区间）。
    参考用户提供的单因子分析逻辑进行批量处理。
    
    :param returns: 资产收益率 DataFrame (T时刻)
    :param factors: 因子值 DataFrame (T时刻)
    :param q: 兼容保留（不再使用；分组固定为 6 档区间）
    :param T_shift: 收益率滞后期数，默认1 (即使用 T时刻因子 预测 T+1时刻收益)
    """
    print(f"\n正在进行 Regime Analysis (Fixed-Bin Analysis), T_shift={T_shift}...")
    
    regime_dir = os.path.join(OUTPUT_DIR, 'regime_analysis')
    if not os.path.exists(regime_dir):
        os.makedirs(regime_dir)
        
    # 准备收益率数据: T+shift
    # 如果 T_shift > 0, 我们需要将 returns 向前 shift (shift(-k)) 以对齐 T 时刻的因子
    if T_shift != 0:
        next_returns = returns.shift(-T_shift)
    else:
        next_returns = returns

    summary_stats = []

    # 遍历所有因子和资产
    for factor_col in factors.columns:
        factor_dir = os.path.join(regime_dir, factor_col)
        if not os.path.exists(factor_dir):
            os.makedirs(factor_dir)
        # 跳过非数值因子
        if not pd.api.types.is_numeric_dtype(factors[factor_col]):
            continue
            
        factor_series = factors[factor_col]
        
        for asset_col in returns.columns:
            asset_ret = next_returns[asset_col]
            
            # --- 核心逻辑开始 (参考用户提供的代码) ---
            # 1. 合并数据
            df = pd.concat([factor_series, asset_ret], axis=1).dropna()
            if len(df) < 24: # 样本太少跳过
                continue
                
            df.columns = ['Factor', 'Next_Ret']
            
            # 2. 固定区间分组（6 档）：-2以下，-2--1，-1-0，0-1，1-2，2以上
            # 用 pd.cut 避免 np.select 在字符串标签 + np.nan default 时的 dtype 冲突
            f = df['Factor'].astype(float)
            ordered_bins = ['<=-1', '-1--0.3', '-0.3-0', '0-0.3', '0.3-1', '>=1']


            df['Group'] = pd.cut(
                f,
                bins=[-np.inf, -1.0, -0.3, 0.0, 0.3, 1.0, np.inf],
                labels=ordered_bins,
                right=True,
                include_lowest=True,
            )
            try:
                df.loc[f >= 2.0, 'Group'] = '>=2'
            except Exception:
                pass

            df['Group'] = pd.Categorical(df['Group'], categories=ordered_bins, ordered=True)

            # 3. 统计核心指标
            # observed=False 确保包含所有定义的类别，即使某些组为空
            stats = df.groupby('Group', observed=False)['Next_Ret'].agg([
                ('Avg_Ret', 'mean'),           # 平均收益
                ('Win_Rate', lambda x: (x>0).mean()), # 胜率
                ('Count', 'count')             # 样本数
            ])
            
            # 年化处理 (假设月频)
            stats['Ann_Ret'] = stats['Avg_Ret'] * 12
            

            
            # 收集汇总数据
            summary_stats.append({
                'Factor': factor_col,
                'Asset': asset_col,
            })

            # 5. 绘图 (保存而不是显示)

            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # 柱状图：平均收益
            colors = ['green' if x > 0 else 'red' for x in stats['Avg_Ret']]
            bars = ax1.bar(stats.index, stats['Ann_Ret'], color=colors, alpha=0.7, label='Ann Return')
            ax1.set_ylabel('Annualized Return')
            ax1.axhline(0, color='black', linewidth=0.8)
            
            # 折线图：胜率 (右轴)
            ax2 = ax1.twinx()
            ax2.plot(stats.index, stats['Win_Rate'], color='blue', marker='o', linestyle='--', linewidth=2, label='Win Rate')
            ax2.set_ylabel('Win Rate')
            ax2.set_ylim(0, 1.0)
            
            # 标注
            plt.title(f'Regime Analysis: {factor_col} -> {asset_col} (T+{T_shift})')
            
            # 保存
            safe_asset = asset_col.replace('/', '_').replace('\\', '_')
            safe_factor = factor_col.replace('/', '_')
            plt.tight_layout()
            plt.savefig(os.path.join(factor_dir, f'{safe_factor}_vs_{safe_asset}.png'))
            # plt.show()
            plt.close()
            
            # --- 核心逻辑结束 ---

    # 保存汇总 CSV
    if summary_stats:
        df_summary = pd.DataFrame(summary_stats)
        
        summary_file = os.path.join(OUTPUT_DIR, f'regime_stats_summary_T+{T_shift}.csv')
        df_summary.to_csv(summary_file, encoding='utf-8-sig', index=False)
        print(f"Regime Analysis 汇总表已保存: {summary_file}")


def analyze_two_factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_x: str,
    factor_y: str,
    T_shift: int = 0,
):
    """双因子回归（可选因子，默认用于“增长+通胀”）。

    回归形式：R_{t+T_shift} = a + b1*X_t + b2*Y_t + e
    严格使用 shift(-T_shift) 对齐，避免未来函数。
    输出：beta/t/R2 CSV + heatmap 图片。
    """
    if factor_x not in factors.columns or factor_y not in factors.columns:
        print(f"\n[Two-Factor] 因子列不存在，跳过：{factor_x}, {factor_y}")
        return None

    print(f"\n[Two-Factor] 双因子回归: ({factor_x}, {factor_y}), T_shift={T_shift}")

    out_dir = os.path.join(OUTPUT_DIR, 'two_factor_regression')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    y_returns = returns.shift(-T_shift) if T_shift != 0 else returns
    X = factors[[factor_x, factor_y]].copy()

    beta = pd.DataFrame(index=returns.columns, columns=[factor_x, factor_y], dtype=float)
    tval = pd.DataFrame(index=returns.columns, columns=[factor_x, factor_y], dtype=float)
    r2 = pd.Series(index=returns.columns, dtype=float)
    nobs = pd.Series(index=returns.columns, dtype=float)
    resid_var = pd.Series(index=returns.columns, dtype=float)

    for asset in returns.columns:
        y = y_returns[asset].rename('y')
        data = pd.concat([y, X], axis=1).dropna()
        if data.shape[0] < 24:
            continue

        y_clean = data['y']
        X_clean = sm.add_constant(data[[factor_x, factor_y]])

        try:
            model = sm.OLS(y_clean, X_clean).fit()
        except Exception:
            continue

        beta.loc[asset, factor_x] = model.params.get(factor_x, np.nan)
        beta.loc[asset, factor_y] = model.params.get(factor_y, np.nan)
        tval.loc[asset, factor_x] = model.tvalues.get(factor_x, np.nan)
        tval.loc[asset, factor_y] = model.tvalues.get(factor_y, np.nan)
        r2.loc[asset] = model.rsquared
        nobs.loc[asset] = model.nobs
        resid_var.loc[asset] = getattr(model, 'mse_resid', np.nan)

    suffix = f"_T+{T_shift}" if T_shift > 0 else ""
    beta_path = os.path.join(out_dir, f"two_factor_betas{suffix}.csv")
    t_path = os.path.join(out_dir, f"two_factor_tvalues{suffix}.csv")
    r2_path = os.path.join(out_dir, f"two_factor_r2{suffix}.csv")
    nobs_path = os.path.join(out_dir, f"two_factor_nobs{suffix}.csv")
    resid_path = os.path.join(out_dir, f"two_factor_resid_var{suffix}.csv")
    beta.to_csv(beta_path, encoding='utf-8-sig')
    tval.to_csv(t_path, encoding='utf-8-sig')
    r2.to_frame('R2').to_csv(r2_path, encoding='utf-8-sig')
    nobs.to_frame('Nobs').to_csv(nobs_path, encoding='utf-8-sig')
    df_resid = pd.DataFrame({
        'ResidVar_Monthly': resid_var,
        'ResidStd_Monthly': np.sqrt(resid_var),
        'ResidStd_Annualized': np.sqrt(resid_var) * np.sqrt(12.0),
        'R2': r2,
        'Nobs': nobs,
    })
    df_resid.to_csv(resid_path, encoding='utf-8-sig')
    print(f"[Two-Factor] 已保存: {beta_path}")

    # # 图片：beta 与 t 值热力图
    # plt.figure(figsize=(8, max(6, 0.25 * len(returns.columns))))
    # sns.heatmap(beta.astype(float), annot=True, fmt='.3f', cmap='coolwarm', center=0)
    # plt.title(f"Two-Factor Betas: ({factor_x}, {factor_y}) -> R_(t+{T_shift})")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"two_factor_beta_heatmap{suffix}.png"))
    # # plt.close()

    plt.figure(figsize=(8, max(6, 0.25 * len(returns.columns))))
    sns.heatmap(tval.astype(float), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f"Two-Factor t-stats: ({factor_x}, {factor_y}) -> R_(t+{T_shift})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"two_factor_tvalue_heatmap{suffix}.png"))
    # plt.close()

    # R2 条形图（辅助）
    r2_plot = r2.dropna().sort_values(ascending=False)
    if not r2_plot.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(r2_plot))))
        sns.barplot(x=r2_plot.values, y=r2_plot.index)
        plt.title(f"Two-Factor R2 (T+{T_shift})")
        plt.xlabel('R2')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"two_factor_r2_bar{suffix}.png"))
        plt.show()
        # plt.close()

    # Residual Std 条形图：宏观因子之外的“无法解释波动”
    resid_std_ann = (np.sqrt(resid_var) * np.sqrt(12.0)).dropna().sort_values(ascending=False)
    if not resid_std_ann.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(resid_std_ann))))
        sns.barplot(x=resid_std_ann.values, y=resid_std_ann.index)
        plt.title(f"Two-Factor Residual Std (Annualized), T+{T_shift}")
        plt.xlabel('Residual Std (ann.)')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"two_factor_resid_std_ann_bar{suffix}.png"))
        plt.show()

    return {
        'beta': beta,
        'tval': tval,
        'r2': r2,
        'nobs': nobs,
        'resid_var': resid_var,
        'out_dir': out_dir,
    }

def analyze_three_factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_x: str,
    factor_y: str,
    factor_z: str,
    T_shift: int = 0,
):
    """三因子回归（可选因子，默认用于“增长+通胀”）。

    回归形式：R_{t+T_shift} = a + b1*X_t + b2*Y_t +b3*Z_t + e
    严格使用 shift(-T_shift) 对齐，避免未来函数。
    输出：beta/t/R2 CSV + heatmap 图片。
    """
    if factor_x not in factors.columns or factor_y not in factors.columns or factor_z not in factors.columns:
        print(f"\n[Three-Factor] 因子列不存在，跳过：{factor_x}, {factor_y}, {factor_z}")
        return None

    print(f"\n[Three-Factor] 三因子回归: ({factor_x}, {factor_y}, {factor_z}), T_shift={T_shift}")

    out_dir = os.path.join(OUTPUT_DIR, 'three_factor_regression')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    y_returns = returns.shift(-T_shift) if T_shift != 0 else returns
    X = factors[[factor_x, factor_y, factor_z]].copy()

    beta = pd.DataFrame(index=returns.columns, columns=[factor_x, factor_y, factor_z], dtype=float)
    tval = pd.DataFrame(index=returns.columns, columns=[factor_x, factor_y, factor_z], dtype=float)
    r2 = pd.Series(index=returns.columns, dtype=float)
    nobs = pd.Series(index=returns.columns, dtype=float)
    resid_var = pd.Series(index=returns.columns, dtype=float)

    for asset in returns.columns:
        y = y_returns[asset].rename('y')
        data = pd.concat([y, X], axis=1).dropna()
        if data.shape[0] < 24:
            continue

        y_clean = data['y']
        X_clean = sm.add_constant(data[[factor_x, factor_y, factor_z]])

        try:
            model = sm.OLS(y_clean, X_clean).fit()
        except Exception:
            continue

        beta.loc[asset, factor_x] = model.params.get(factor_x, np.nan)
        beta.loc[asset, factor_y] = model.params.get(factor_y, np.nan)
        beta.loc[asset, factor_z] = model.params.get(factor_z, np.nan)
        tval.loc[asset, factor_x] = model.tvalues.get(factor_x, np.nan)
        tval.loc[asset, factor_y] = model.tvalues.get(factor_y, np.nan)
        tval.loc[asset, factor_z] = model.tvalues.get(factor_z, np.nan)
        r2.loc[asset] = model.rsquared
        nobs.loc[asset] = model.nobs
        resid_var.loc[asset] = getattr(model, 'mse_resid', np.nan)

    suffix = f"_T+{T_shift}" if T_shift > 0 else ""
    beta_path = os.path.join(out_dir, f"three_factor_betas{suffix}.csv")
    t_path = os.path.join(out_dir, f"three_factor_tvalues{suffix}.csv")
    r2_path = os.path.join(out_dir, f"three_factor_r2{suffix}.csv")
    nobs_path = os.path.join(out_dir, f"three_factor_nobs{suffix}.csv")
    resid_path = os.path.join(out_dir, f"three_factor_resid_var{suffix}.csv")
    beta.to_csv(beta_path, encoding='utf-8-sig')
    tval.to_csv(t_path, encoding='utf-8-sig')
    r2.to_frame('R2').to_csv(r2_path, encoding='utf-8-sig')
    nobs.to_frame('Nobs').to_csv(nobs_path, encoding='utf-8-sig')
    df_resid = pd.DataFrame({
        'ResidVar_Monthly': resid_var,
        'ResidStd_Monthly': np.sqrt(resid_var),
        'ResidStd_Annualized': np.sqrt(resid_var) * np.sqrt(12.0),
        'R2': r2,
        'Nobs': nobs,
    })
    df_resid.to_csv(resid_path, encoding='utf-8-sig')
    print(f"[Three-Factor] 已保存: {beta_path}")

    # # 图片：beta 与 t 值热力图
    # plt.figure(figsize=(8, max(6, 0.25 * len(returns.columns))))
    # sns.heatmap(beta.astype(float), annot=True, fmt='.3f', cmap='coolwarm', center=0)
    # plt.title(f"Two-Factor Betas: ({factor_x}, {factor_y}) -> R_(t+{T_shift})")
    # plt.tight_layout()
    # plt.savefig(os.path.join(out_dir, f"two_factor_beta_heatmap{suffix}.png"))
    # # plt.close()

    plt.figure(figsize=(10, max(6, 0.25 * len(returns.columns))))
    sns.heatmap(tval.astype(float), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f"Three-Factor t-stats: ({factor_x}, {factor_y}, {factor_z}) -> R_(t+{T_shift})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"three_factor_tvalue_heatmap{suffix}.png"))
    # plt.close()

    # R2 条形图（辅助）
    r2_plot = r2.dropna().sort_values(ascending=False)
    if not r2_plot.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(r2_plot))))
        sns.barplot(x=r2_plot.values, y=r2_plot.index)
        plt.title(f"Three-Factor R2 (T+{T_shift})")
        plt.xlabel('R2')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"three_factor_r2_bar{suffix}.png"))
        plt.show()
        # plt.close()

    resid_std_ann = (np.sqrt(resid_var) * np.sqrt(12.0)).dropna().sort_values(ascending=False)
    if not resid_std_ann.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(resid_std_ann))))
        sns.barplot(x=resid_std_ann.values, y=resid_std_ann.index)
        plt.title(f"Three-Factor Residual Std (Annualized), T+{T_shift}")
        plt.xlabel('Residual Std (ann.)')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"three_factor_resid_std_ann_bar{suffix}.png"))
        plt.show()

    return {
        'beta': beta,
        'tval': tval,
        'r2': r2,
        'nobs': nobs,
        'resid_var': resid_var,
        'out_dir': out_dir,
    }

def analyze_four_factor_regression(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
    factor_x: str,
    factor_y: str,
    factor_z: str,
    factor_w: str,
    T_shift: int = 0,
):
    """四因子回归（可选因子）。

    回归形式：R_{t+T_shift} = a + b1*X_t + b2*Y_t + b3*Z_t + b4*W_t + e
    严格使用 shift(-T_shift) 对齐，避免未来函数。
    输出：beta/t/R2/Nobs CSV + beta/t heatmap + R2 bar。
    """
    missing = [c for c in [factor_x, factor_y, factor_z, factor_w] if c not in factors.columns]
    if missing:
        print(f"\n[Four-Factor] 因子列不存在，跳过：{missing}")
        return None

    print(f"\n[Four-Factor] 四因子回归: ({factor_x}, {factor_y}, {factor_z}, {factor_w}), T_shift={T_shift}")

    out_dir = os.path.join(OUTPUT_DIR, 'four_factor_regression')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    y_returns = returns.shift(-T_shift) if T_shift != 0 else returns
    X = factors[[factor_x, factor_y, factor_z, factor_w]].copy()

    cols = [factor_x, factor_y, factor_z, factor_w]
    beta = pd.DataFrame(index=returns.columns, columns=cols, dtype=float)
    tval = pd.DataFrame(index=returns.columns, columns=cols, dtype=float)
    r2 = pd.Series(index=returns.columns, dtype=float)
    nobs = pd.Series(index=returns.columns, dtype=float)
    resid_var = pd.Series(index=returns.columns, dtype=float)

    for asset in returns.columns:
        y = y_returns[asset].rename('y')
        data = pd.concat([y, X], axis=1).dropna()
        if data.shape[0] < 24:
            continue

        y_clean = data['y']
        X_clean = sm.add_constant(data[cols])
        try:
            model = sm.OLS(y_clean, X_clean).fit()
        except Exception:
            continue

        for c in cols:
            beta.loc[asset, c] = model.params.get(c, np.nan)
            tval.loc[asset, c] = model.tvalues.get(c, np.nan)
        r2.loc[asset] = model.rsquared
        nobs.loc[asset] = model.nobs
        resid_var.loc[asset] = getattr(model, 'mse_resid', np.nan)

    suffix = f"_T+{T_shift}" if T_shift > 0 else ""
    beta_path = os.path.join(out_dir, f"four_factor_betas{suffix}.csv")
    t_path = os.path.join(out_dir, f"four_factor_tvalues{suffix}.csv")
    r2_path = os.path.join(out_dir, f"four_factor_r2{suffix}.csv")
    nobs_path = os.path.join(out_dir, f"four_factor_nobs{suffix}.csv")
    resid_path = os.path.join(out_dir, f"four_factor_resid_var{suffix}.csv")
    beta.to_csv(beta_path, encoding='utf-8-sig')
    tval.to_csv(t_path, encoding='utf-8-sig')
    r2.to_frame('R2').to_csv(r2_path, encoding='utf-8-sig')
    nobs.to_frame('Nobs').to_csv(nobs_path, encoding='utf-8-sig')
    df_resid = pd.DataFrame({
        'ResidVar_Monthly': resid_var,
        'ResidStd_Monthly': np.sqrt(resid_var),
        'ResidStd_Annualized': np.sqrt(resid_var) * np.sqrt(12.0),
        'R2': r2,
        'Nobs': nobs,
    })
    df_resid.to_csv(resid_path, encoding='utf-8-sig')
    print(f"[Four-Factor] 已保存: {beta_path}")



    plt.figure(figsize=(12, max(6, 0.25 * len(returns.columns))))
    sns.heatmap(tval.astype(float), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title(f"Four-Factor t-stats -> R_(t+{T_shift})")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"four_factor_tvalue_heatmap{suffix}.png"))
    # plt.close()

    r2_plot = r2.dropna().sort_values(ascending=False)
    if not r2_plot.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(r2_plot))))
        sns.barplot(x=r2_plot.values, y=r2_plot.index)
        plt.title(f"Four-Factor R2 (T+{T_shift})")
        plt.xlabel('R2')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"four_factor_r2_bar{suffix}.png"))
        plt.show()
        # plt.close()

    resid_std_ann = (np.sqrt(resid_var) * np.sqrt(12.0)).dropna().sort_values(ascending=False)
    if not resid_std_ann.empty:
        plt.figure(figsize=(10, max(4, 0.25 * len(resid_std_ann))))
        sns.barplot(x=resid_std_ann.values, y=resid_std_ann.index)
        plt.title(f"Four-Factor Residual Std (Annualized), T+{T_shift}")
        plt.xlabel('Residual Std (ann.)')
        plt.ylabel('Asset')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"four_factor_resid_std_ann_bar{suffix}.png"))
        plt.show()

    return {
        'beta': beta,
        'tval': tval,
        'r2': r2,
        'nobs': nobs,
        'resid_var': resid_var,
        'out_dir': out_dir,
    }

def main():

    returns, factors, sub_factors, mimicking_factors = load_and_process_data()

    # 根据资产池过滤（便于后续只分析指定资产子集）
    returns = filter_returns_by_universe(returns, ASSET_UNIVERSE_KEY)

    # 将因子按 Level / Change 分组（基于列名后缀）
    factor_cols = list(factors.columns)
    level_cols = [c for c in factor_cols if str(c).endswith('_Level')]
    change_cols = [c for c in factor_cols if str(c).endswith('_Change')]
    other_cols = [c for c in factor_cols if c not in set(level_cols + change_cols)]

    factors_level = factors[level_cols].copy() if level_cols else factors.iloc[:, 0:0].copy()
    factors_change = factors[change_cols].copy() if change_cols else factors.iloc[:, 0:0].copy()

    sub_factors_level = sub_factors[[c for c in sub_factors.columns if str(c).endswith('_Level')]].copy()
    sub_factors_change = sub_factors[[c for c in sub_factors.columns if str(c).endswith('_Change')]].copy()

    # factors_change.corr()
    # factors_change.cov()
    print(f"\n[Factors] Level: {len(level_cols)} cols; Change: {len(change_cols)} cols")
    if other_cols:
        print(f"[Factors] 未归类列(非 _Level/_Change): {other_cols}")
    
    if returns.empty or factors.empty:
        print("数据为空，无法分析。")
        return
    
    # --- 单因子回归（看因子显著性）---
    
    # analyze_correlations_and_single_factor_regressions(returns, sub_factors_change, T_shift=0)

    # --- 双因子回归（默认：中国增长 + 中国通胀；可按需修改）---

    # 优先用 Change；如果你想用 Level，把下面两行改成 *_Level
    factor_x_name = 'CN_Growth_Change'
    factor_y_name = 'CN_Inflation_Change'
    factor_z_name = 'CN_RiskAppetite_Change'
    factor_w_name = 'CN_Monetary_Change'

    # analyze_two_factor_regression(returns,factors_change,factor_x=factor_x_name,factor_y=factor_y_name,T_shift=1)
    # analyze_three_factor_regression(returns,factors_change,factor_x=factor_w_name,factor_y=factor_y_name,factor_z=factor_z_name,T_shift=1)
    # analyze_four_factor_regression(returns,factors_change,factor_x=factor_x_name,factor_y=factor_y_name,factor_z=factor_z_name,factor_w=factor_w_name,T_shift=0)
    analyze_regimes(returns, factors_level,  T_shift=2)

    
    print("\n所有分析完成！结果保存在 analysis_results 文件夹中。")
        



if __name__ == "__main__":
    main()
