# -*- coding: utf-8 -*-
"""
测试协方差估计函数
"""
import numpy as np
import pandas as pd
import sys
sys.path.append('d:\\OneDrive\\投研\\ETF策略')

from framework.allocation_utils import estimate_covariance_matrix

# 生成模拟数据
np.random.seed(42)
n_days = 500
n_assets = 3

# 生成有偏态分布的收益率（更多负收益）
returns = np.random.randn(n_days, n_assets) * 0.02
# 添加一些极端负收益（模拟尾部风险）
returns[:50, :] = returns[:50, :] - 0.03  # 前50天有更大负收益

returns_df = pd.DataFrame(
    returns,
    columns=['Asset_A', 'Asset_B', 'Asset_C'],
    index=pd.date_range('2020-01-01', periods=n_days, freq='D')
)

print("="*60)
print("收益率数据统计:")
print(returns_df.describe())
print(f"\n负收益天数: {(returns_df < 0).sum()}")
print(f"正收益天数: {(returns_df > 0).sum()}")

# 测试四种协方差估计方法
methods = ['cov', 'ewm', 'downside_cov', 'downside_ewm', 'oas']

print("\n" + "="*60)
print("协方差矩阵估计结果对比:")
print("="*60)

for method in methods:
    print(f"\n方法: {method.upper()}")
    print("-"*60)
    try:
        cov_matrix = estimate_covariance_matrix(
            returns_df,
            method=method,
            ewm_span=252,
            min_negative_samples=20
        )
        print(cov_matrix)
        
        # 计算相关系数矩阵（便于比较）
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        print(f"\n相关系数矩阵:")
        print(corr_matrix)
        
        # 计算波动率（标准差）
        vols = np.sqrt(np.diag(cov_matrix))
        print(f"\n各资产波动率（年化，假设252交易日）:")
        for asset, vol in zip(returns_df.columns, vols):
            annual_vol = vol * np.sqrt(252)
            print(f"  {asset}: {annual_vol:.2%}")
            
    except Exception as e:
        print(f"ERROR: {str(e)}")

# 测试手动收缩方法
print("\n" + "="*60)
print("手动收缩方法 (Shrunk) 对比:")
print("="*60)

shrunk_targets = ['diagonal', 'constant_correlation', 'identity']
for target in shrunk_targets:
    print(f"\n收缩目标: {target.upper()} (强度=0.5)")
    print("-"*60)
    try:
        cov_matrix = estimate_covariance_matrix(
            returns_df,
            method='shrunk',
            shrinkage_target=target,
            shrinkage_intensity=0.5
        )
        print(cov_matrix)
        
        std = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std, std)
        print(f"\n相关系数矩阵:")
        print(corr_matrix)
        
    except Exception as e:
        print(f"ERROR: {str(e)}")

# 测试下半协方差vs标准协方差的差异
print("\n" + "="*60)
print("下半协方差 vs 标准协方差 - 波动率对比:")
print("="*60)

cov_standard = estimate_covariance_matrix(returns_df, method='cov')
cov_downside = estimate_covariance_matrix(returns_df, method='downside_cov')

vol_standard = np.sqrt(np.diag(cov_standard)) * np.sqrt(252)
vol_downside = np.sqrt(np.diag(cov_downside)) * np.sqrt(252)

comparison_df = pd.DataFrame({
    '标准波动率': vol_standard,
    '下半波动率': vol_downside,
    '比率(下/标)': vol_downside / vol_standard
}, index=returns_df.columns)

print(comparison_df)

print("\n" + "="*60)
print("解释:")
print("- 下半波动率 < 标准波动率: 负收益的波动性更小")
print("- 下半波动率 > 标准波动率: 负收益的波动性更大（尾部风险更高）")
print("- 比率接近1: 收益分布对称")
print("="*60)
