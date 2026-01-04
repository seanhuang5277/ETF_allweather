# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 23:29:32 2025

@author: Sean
"""

import pandas as pd
from WindPy import w
import numpy as np
# --- 1. 配置参数 ---

# [!] 最终确定的 12 只代理 ETF 列表
proxy_etf_list = [
    # 沪深300 (4支)
    '510300.SH', '510310.SH', '510330.SH', '159919.SZ',
    # 科创50 (2支)
    '588000.SH', '588080.SH',
    # 创业板指 (1支)
    '159915.SZ',
    # 5年国债 (1支)
    '511010.SH',
    # 黄金 (4支)
    '518880.SH', '159937.SZ', '159934.SZ', '518800.SH'
]

# 1.2 [!] 关键：定义聚合规则
# (这必须与我们的收益率文件列名完全一致)
meta = {
    '沪深300指数': ['510300.SH', '510310.SH', '510330.SH', '159919.SZ'],
    '科创50指数': ['588000.SH', '588080.SH'],
    '创业板指': ['159915.SZ'],
    '上证5年期国债指数': ['511010.SH'],
    '上海黄金AU9999': ['518880.SH', '159937.SZ', '159934.SZ', '518800.SH']
}

# AUM 数据我们从2015年开始获取
start_date = '2015-01-01'
end_date = pd.Timestamp.today().strftime('%Y-%m-%d')

output_filename = 'proxy_etfs_aum_monthly.csv'

# --- 2. Wind API 初始化与登录 ---
try:
    if not w.isconnected():
        w.start()
        print("Wind API 尝试连接...")
    if w.isconnected():
        print("Wind API 连接成功。")
    else:
        print("Wind API 连接失败，请检查客户端状态。")
        exit()
except Exception as e:
    print(f"Wind API 初始化错误: {e}")
    exit()

# --- 3. 获取 AUM 数据 (月度) ---
all_aum_data_raw = pd.DataFrame()

# Wind API 字段：ETF总市值(亿元)



print(f"开始获取 {len(proxy_etf_list)} 只 ETF 的月度 AUM 数据...")

for ticker in proxy_etf_list:
    print(f"  正在获取: {ticker} ...")
    
    # 使用 WSD 函数获取月度 AUM
    wsd_data = w.wsd(ticker, 
                     "val_mv_ARD", 
                     start_date, 
                     end_date, 
                     "unit=1;Period=M;Days=Trading") # Period=M 获取月度数据
    
    if wsd_data.ErrorCode != 0:
        print(f"    获取 {ticker} 数据失败，错误代码: {wsd_data.ErrorCode}")
        continue

    # 将数据转换为 DataFrame
    df = pd.DataFrame(wsd_data.Data, index=wsd_data.Fields, columns=wsd_data.Times).T
    #转化为亿
    df["VAL_MV_ARD"] = pd.to_numeric(df["VAL_MV_ARD"], errors='coerce') / 100000000
    # 重命名列
    df.columns = [ticker]
    
    # 合并到主 DataFrame
    if all_aum_data_raw.empty:
        all_aum_data_raw = df
    else:
        all_aum_data_raw = all_aum_data_raw.join(df, how='outer')

print("\nAUM 数据获取完成，正在进行清洗和保存...")

# --- 4. [!] 关键：合并数据 ---
# 创建一个新的 DataFrame 用于存储聚合后的 5 列数据
aggregated_aum_data = pd.DataFrame(index=all_aum_data_raw.index)

for asset_name, etf_list in meta.items():
    print(f"  正在合并: {asset_name} (基于 {len(etf_list)} 支 ETF)")
    
    # 确保所有 ETF 都在我们已下载的列中
    available_etfs = [etf for etf in etf_list if etf in all_aum_data_raw.columns]
    
    if not available_etfs:
        print(f"    警告: {asset_name} 的所有代理 ETF 均无数据。")
        aggregated_aum_data[asset_name] = np.nan
        continue
# [!] 核心合并逻辑
    # .sum(axis=1) 会按行求和
    # skipna=True 是默认设置，它会自动处理 NaN (即未上市的 ETF)
    aggregated_aum_data[asset_name] = all_aum_data_raw[available_etfs].sum(axis=1)

aggregated_aum_data.replace(0, np.nan, inplace=True)

print("\n数据合并完成。")
# --- 4. 数据清洗与保存 ---

# 1. 清理：只有当一行的所有列都为 NaN 时 (how='all')，才删除该行
aggregated_aum_data.dropna(how='all', inplace=True)

# 2. 索引命名
aggregated_aum_data.index.name = '日期'

# 3. 保存：保留 datetime 索引，使用 utf-8-sig 编码
aggregated_aum_data.to_csv(output_filename, 
                    header=True, 
                    index=True, 
                    encoding='utf-8-sig')

print(f"\n成功将 {len(proxy_etf_list)} 只 ETF 的月度 AUM 数据保存到文件:")
print(f"{output_filename}")
print(f"数据起始日期: {aggregated_aum_data.index.min().strftime('%Y-%m-%d')}")
print(f"数据截止日期: {aggregated_aum_data.index.max().strftime('%Y-%m-%d')}")
print(f"总月份数量: {len(aggregated_aum_data)}")

print(f"\n成功将 {len(meta)} 个资产类别的【聚合后】AUM 数据保存到文件:")
print(f"{output_filename}")
print("\n聚合 AUM 数据预览 (单位：亿元)：")
print(aggregated_aum_data.tail())

# --- 5. 关闭 Wind API ---
# w.stop()
