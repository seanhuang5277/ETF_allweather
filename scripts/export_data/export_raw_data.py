# -*- coding: utf-8 -*-
"""
从配置 CSV 读取 ETF-指数一一映射，导出两份总表：
    - ETF_daily_simple_returns.csv       (所有 ETF 日度简单收益宽表)
    - index_daily_simple_returns.csv     (所有指数 日度简单收益宽表)

说明：
    - CSV 文件：config/config_export_etf_and_index_rawdata.csv
    - 固定列名（精确匹配）：ETF代码, ETF名称, Index代码, Index名称, 资产类别, 是否调用回测[Y/N]
    - 本脚本忽略“是否调用回测[Y/N]”列，不再按资产类别拆文件
"""

import os
import re
import pandas as pd
from WindPy import w


# --- 1. 基本参数 ---
start_date = '2010-01-01'
end_date = '2025-12-31'
# pd.Timestamp.today().strftime('%Y-%m-%d')

try:
    ROOT = os.path.dirname(os.path.abspath(__file__))
except NameError:
    ROOT = os.getcwd()
DATA_DIR = os.path.join(ROOT, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_DIR = os.path.join(ROOT, 'config')
# 配置文件（CSV）路径
CONFIG_CSV = os.path.join(CONFIG_DIR, 'config_export_etf_and_index_rawdata.csv')
# --- 2. 简单工具函数 ---



def _normalize_code(x: str) -> str:
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return ''
    s = re.sub(r"\.0$", "", s)
    s = s.upper()
    return s


def _load_config_from_df(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how='all')
    raw = raw.loc[:, ~raw.columns.astype(str).str.strip().duplicated()]

    # 精确匹配所需列（与配置 CSV 完全一致）
    expected_cols = ['ETF代码', 'ETF名称', 'Index代码', 'Index名称', '资产类别']
    missing = [c for c in expected_cols if c not in raw.columns]
    if missing:
        existing_cols = ', '.join(map(str, raw.columns.tolist()))
        raise ValueError(f"配置缺少必要列: {', '.join(missing)} | 已检测到列: {existing_cols}")

    df = pd.DataFrame({
        'category': raw['资产类别'].astype(str).str.strip(),
        'etf_code': raw['ETF代码'].map(_normalize_code),
        'etf_name': raw['ETF名称'].astype(str).str.strip(),
        'index_code': raw['Index代码'].map(_normalize_code),
        'index_name': raw['Index名称'].astype(str).str.strip(),
    })

    df = df[(df['etf_code'] != '') & (df['index_code'] != '')]
    df = df.drop_duplicates(subset=['category', 'etf_code', 'index_code'], keep='first')
    return df


def load_config() -> pd.DataFrame:
    if not os.path.isfile(CONFIG_CSV):
        raise FileNotFoundError(f"未找到配置 CSV 文件: {CONFIG_CSV}")
    last_err = None
    raw = None
    for enc in ('utf-8-sig', 'gbk', 'cp936', 'utf-8'):
        try:
            raw = pd.read_csv(CONFIG_CSV, encoding=enc)
            break
        except Exception as e:
            last_err = e
    if raw is None:
        raise ValueError(f"CSV 读取失败: {CONFIG_CSV} | {last_err}")
    return _load_config_from_df(raw)


def fetch_series_for_ticker(ticker: str, display_name: str) -> pd.Series | None:
    print(f"正在获取 {ticker} ({display_name}) 的数据...")
    wsd_data = w.wsd(ticker, "PCT_CHG", start_date, end_date, "Period=D;Days=Trading")
    if wsd_data.ErrorCode != 0:
        print(f"获取 {ticker} 数据失败，错误代码: {wsd_data.ErrorCode}")
        return None
    df = pd.DataFrame(wsd_data.Data, index=wsd_data.Fields, columns=wsd_data.Times).T
    # 转换为小数收益率
    sr = df['PCT_CHG'] / 100.0
    sr.name = display_name
    return sr


# --- 3. Wind API 初始化 ---
try:
    if not w.isconnected():
        w.start()
        print("Wind API 尝试连接...")
    if w.isconnected():
        print("Wind API 连接成功。")
    else:
        print("Wind API 连接失败，请检查客户端状态。")
        raise SystemExit(1)
except Exception as e:
    print(f"Wind API 初始化错误: {e}")
    raise SystemExit(1)


# --- 4. 读取配置并导出合并总表 ---
try:
    cfg = load_config()
except Exception as e:
    print(f"读取配置失败: {e}")
    w.stop()
    raise SystemExit(1)

# 1) ETF 宽表（按配置顺序累积所有 ETF）
etf_df = pd.DataFrame()
for _, r in cfg.iterrows():
    s = fetch_series_for_ticker(r['etf_code'], r.get('etf_name', r['etf_code']))
    if s is None:
        continue
    etf_df = s.to_frame() if etf_df.empty else etf_df.join(s, how='outer')

# 2) 指数宽表（与 ETF 一一对应，顺序相同）
idx_df = pd.DataFrame(index=etf_df.index if not etf_df.empty else None)
for _, r in cfg.iterrows():
    s = fetch_series_for_ticker(r['index_code'], r.get('index_name', r['index_code']))
    if s is None:
        continue
    idx_df = s.to_frame() if idx_df.empty else idx_df.join(s, how='outer')

# 写出
if etf_df.empty and idx_df.empty:
    print("[WARN] 无可写出的数据，已跳过。")
else:
    if not etf_df.empty:
        etf_df.index.name = '日期'
        etf_df = etf_df.sort_index()
        out_etf = os.path.join(DATA_DIR, "ETF_daily_simple_returns.csv")
        etf_df.to_csv(out_etf, header=True, index=True, encoding='utf-8-sig')
        print(f"已写出 ETF 文件: {out_etf} | 行数={len(etf_df)} 列数={etf_df.shape[1]}")
    if not idx_df.empty:
        idx_df.index.name = '日期'
        idx_df = idx_df.sort_index()
        out_idx = os.path.join(DATA_DIR, "index_daily_simple_returns.csv")
        idx_df.to_csv(out_idx, header=True, index=True, encoding='utf-8-sig')
        print(f"已写出 指数 文件: {out_idx} | 行数={len(idx_df)} 列数={idx_df.shape[1]}")

print("\n数据获取完成，已写出 ETF 与指数总表到 data/ 目录。")

# --- 5. 关闭 Wind API ---
w.stop()