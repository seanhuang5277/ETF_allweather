# -*- coding: utf-8 -*-
"""
@Project: ETF Strategy
@File   : macro_factor_mimicking.py
@Description: 
    根据导出的高频宏观因子原始数据 (macro_factors_mimicking_raw.csv) 和配置 (config_factor_mimicking.csv)，
    合成高频宏观因子 (High-Frequency Macro Factors)。

    合成方法 (参考华泰证券研报):
    1. 数据预处理: 
       - 计算日频收益率 (对于价格类数据) 或 差分 (对于利率类数据，视配置而定)。
       - 这里的配置 raw_data_type 均为 "价格"，因此统一计算日频收益率 (pct_change)。
    2. 因子合成:
       - 对于有明确权重 (Weight) 的因子 (如 Term Spread, Credit Spread): 采用加权求和。
       - 对于无权重 (Weight=NaN) 的因子 (如 Growth, Inflation): 采用 PCA (第一主成分) 提取共同趋势。
    3. 因子化:
       - 将合成的日频变化量累积 (Cumsum/Cumprod) 得到因子净值曲线。
       - 标准化 (Z-Score) 以方便比较。

@Usage:
    python macro_factor_mimicking.py
"""

import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# --- 路径配置 ---
try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()

DATA_DIR = os.path.join(ROOT, 'data')
CONFIG_DIR = os.path.join(ROOT, 'config')

RAW_DATA_FILE = os.path.join(DATA_DIR, 'macro_factors_mimicking_raw.csv')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config_factor_mimicking.csv')
OUTPUT_FILE = os.path.join(DATA_DIR, 'macro_factors_mimicking.csv')

def load_data():
    """加载数据和配置"""
    # 读取配置
    try:
        try:
            cfg = pd.read_csv(CONFIG_FILE, encoding='utf-8-sig')
        except UnicodeDecodeError:
            cfg = pd.read_csv(CONFIG_FILE, encoding='gbk')
    except Exception as e:
        raise FileNotFoundError(f"无法读取配置文件: {e}")

    # 读取数据
    if not os.path.exists(RAW_DATA_FILE):
        raise FileNotFoundError(f"找不到原始数据文件: {RAW_DATA_FILE}")
    
    df = pd.read_csv(RAW_DATA_FILE, index_col='date', parse_dates=True)
    
    return cfg, df

def synthesize_pca(returns_df):
    """
    PCA 合成 (取第一主成分)
    returns_df: 资产收益率 DataFrame
    """
    # 去除 NaN (PCA 不支持)
    df_clean = returns_df.dropna()
    if df_clean.empty:
        return pd.Series(np.nan, index=returns_df.index)
    
    # 标准化
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_clean)
    
    # PCA
    pca = PCA(n_components=1)
    pca.fit(df_scaled)
    
    # 获取第一主成分
    # 符号校正: 假设大多数资产与因子正相关 (Loading sum > 0)
    loadings = pca.components_[0]
    direction = np.sign(np.sum(loadings))
    
    factor = pca.transform(df_scaled)[:, 0] * direction
    
    return pd.Series(factor, index=df_clean.index)

def main():
    print("正在加载数据...")
    cfg, df_raw = load_data()
    
    # 1. 计算日频变化 (Returns)
    # 假设所有数据都是 "价格" 类型，计算百分比变化
    # 如果是利率 (Rate)，通常计算差分 (Diff)，但配置里 raw_data_type 都是 "价格"
    # 且 Term Spread 用的是 "总财富指数" (Wealth Index)，所以应该用 pct_change
    print("计算日频收益率...")
    df_returns = df_raw.pct_change()
    
    # 准备结果容器
    factor_results = {}
    
    # 获取所有因子类别
    factor_types = cfg['factor_type'].dropna().unique()
    
    for f_type in factor_types:
        f_type = f_type.strip()
        print(f"\n正在合成因子: {f_type} ...")
        
        # 筛选该因子的成分
        sub_cfg = cfg[cfg['factor_type'] == f_type]
        
        # 获取成分数据
        components = []
        weights = []
        has_weights = True
        
        for _, row in sub_cfg.iterrows():
            name = row['宏观变量名'].strip()
            weight = row['权重'] # 可能为 NaN
            
            if name in df_returns.columns:
                components.append(df_returns[name])
                if pd.isna(weight):
                    has_weights = False
                else:
                    weights.append(weight)
            else:
                print(f"  [Warn] 缺失数据: {name}")
        
        if not components:
            print(f"  [Error] {f_type} 无有效数据")
            continue
            
        df_comp = pd.concat(components, axis=1)
        
        # 2. 合成逻辑
        if has_weights and len(weights) == len(components):
            # 有明确权重 -> 加权求和
            print("  使用加权求和法 (Weighted Sum)")
            # 归一化权重? 通常 Spread 是 1 和 -1，不需要归一化
            # 如果是 Portfolio，可能需要。这里直接用配置的权重。
            factor_ret = df_comp.dot(np.array(weights))
        else:
            # 无权重 -> PCA
            print("  使用 PCA 法 (First Principal Component)")
            factor_ret = synthesize_pca(df_comp)
            
        # 3. 累积得到因子净值 (Factor Index)
        # 简单累加 (log return 近似) 或 累乘
        # PCA 出来的已经是 "Factor Return" (标准化后的)，直接累加即可
        # 加权求和出来的是 Portfolio Return，累乘 (1+r).cumprod()
        
        if has_weights:
            # 累乘
            factor_index = (1 + factor_ret).cumprod()
            # 归一化起始点为 100
            factor_index = factor_index / factor_index.iloc[0] * 100
        else:
            # PCA 结果通常是均值为0的序列，视为 "变化量" 或 "得分"
            # 我们可以对其累加，或者直接作为 "因子值" (State Variable)
            # 研报中 "高频因子" 通常指 "变化率" 还是 "水平"?
            # 如果是 Growth，通常看 YoY 或 趋势。
            # PCA on Returns -> Common Return Factor. Cumsum -> Common Trend.
            # 我们输出 累积值 (Trend)
            factor_index = factor_ret.cumsum()
            
        factor_results[f_type] = factor_index
        
    # 合并结果
    df_factors = pd.DataFrame(factor_results)
    
    # 对齐日期 (ffill)
    df_factors = df_factors.reindex(df_raw.index).ffill()
    
    # 保存
    df_factors.to_csv(OUTPUT_FILE, encoding='utf-8-sig')
    print(f"\n合成完成！")
    print(f"输出文件: {OUTPUT_FILE}")
    print(f"因子列表: {df_factors.columns.tolist()}")

if __name__ == "__main__":
    main()
