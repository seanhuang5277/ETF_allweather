# -*- coding: utf-8 -*-
"""
演示如何在全天候策略中使用不同的协方差估计方法

协方差估计方法说明：
1. 'cov' - 标准样本协方差
   - 使用全部历史数据
   - 对正负收益一视同仁
   - 适合收益分布对称的资产

2. 'ewm' - 指数加权移动协方差
   - 给予近期数据更高权重
   - 适合市场环境快速变化的情况
   - ewm_span=252 表示半衰期为一年

3. 'downside_cov' - 下半协方差（半方差）
   - 仅使用负收益计算
   - 更关注下行风险
   - 适合风险厌恶型投资者
   - 能更好地识别熊市中的资产相关性

5. 'oas' - Oracle Approximating Shrinkage (sklearn)
   - 自动确定最佳收缩强度
   - 收缩目标为 Scaled Identity (常方差对角阵)
   - 均方误差 (MSE) 优于 Ledoit-Wolf
   - 适合样本量较少或噪声较大的情况

6. 'shrunk' - 手动收缩估计
   - 支持自定义收缩目标和强度
   - 目标可选：
     - 'diagonal': 收缩向对角阵（保留方差，协方差为0）
     - 'constant_correlation': 收缩向常相关系数矩阵
     - 'identity': 收缩向单位阵
   - 适合有明确先验假设的场景

使用建议：
- 顶层（象限间）: 建议使用 'ewm' 或 'downside_ewm'
  原因：宏观环境变化较快，需要动态调整

- 底层（ETF间）: 建议使用 'oas' 或 'shrunk' (constant_correlation)
  原因：同类资产相关性较稳定，收缩估计能减少估计误差
"""

# 示例1：保守配置（关注下行风险）
conservative_config = {
    'top_cov_estimate_ways': 'downside_ewm',  # 顶层使用下半EWM
    'bottom_cov_estimate_ways': 'downside_cov',  # 底层使用下半协方差
    'ewm_span_days': 180,  # 半年半衰期（更敏感）
}

# 示例2：激进配置（快速响应市场变化）
aggressive_config = {
    'top_cov_estimate_ways': 'ewm',  # 顶层使用EWM
    'bottom_cov_estimate_ways': 'ewm',  # 底层也使用EWM
    'ewm_span_days': 126,  # 半年半衰期
}

# 示例3：均衡配置（长期稳定）
balanced_config = {
    'top_cov_estimate_ways': 'ewm',  # 顶层使用EWM
    'bottom_cov_estimate_ways': 'cov',  # 底层使用标准协方差
    'ewm_span_days': 252,  # 一年半衰期
}

# 示例4：稳健收缩配置（减少估计误差）
robust_shrinkage_config = {
    'top_cov_estimate_ways': 'oas',  # 顶层使用OAS自动收缩
    'bottom_cov_estimate_ways': 'shrunk',  # 底层使用手动收缩
    # 注意：目前 strategy_AllWeather.py 尚未直接暴露 shrinkage_target 参数，
    # 若需使用 'shrunk' 的特定目标，需修改 strategy 代码或使用默认值
}

print("="*80)
print("全天候策略协方差估计方法配置示例")
print("="*80)

configs = {
    '保守配置（关注下行风险）': conservative_config,
    '激进配置（快速响应）': aggressive_config,
    '均衡配置（长期稳定）': balanced_config,
    '稳健收缩配置（减少误差）': robust_shrinkage_config,
}

for name, config in configs.items():
    print(f"\n{name}:")
    print("-"*80)
    print(f"  顶层方法: {config['top_cov_estimate_ways']}")
    print(f"  底层方法: {config['bottom_cov_estimate_ways']}")
    print(f"  EWM半衰期: {config['ewm_span_days']}天 ({config['ewm_span_days']/252:.1f}年)")
    
print("\n" + "="*80)
print("在 strategy_AllWeather.py 中使用:")
print("="*80)
print("""
res = run_strategy(
    internal_method='HRP',
    start_date="2018-11-30",
    end_date="2025-11-30",
    
    # 选择配置（例如：保守配置）
    top_cov_estimate_ways='downside_ewm',  # 顶层使用下半EWM
    bottom_cov_estimate_ways='downside_cov',  # 底层使用下半协方差
    ewm_span_days=180,  # 半年半衰期
    
    # 其他参数...
)
""")

print("\n" + "="*80)
print("方法对比实验建议:")
print("="*80)
print("""
1. 回测不同配置在不同市场环境下的表现
2. 比较指标：
   - 最大回撤（下半协方差方法理论上应该更小）
   - 夏普比率
   - Calmar比率
   - 下行标准差
   
3. 分时段分析：
   - 牛市（2019-2021）：激进配置可能表现更好
   - 熊市（2022）：保守配置应该回撤更小
   - 震荡市（2023-2024）：均衡配置可能最稳定
""")
