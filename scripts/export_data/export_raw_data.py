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

import re
import time
import random
from pathlib import Path
import pandas as pd
from WindPy import w


# --- 1. 基本参数 ---
start_date = '2010-01-01'
end_date = '2025-12-31'
# pd.Timestamp.today().strftime('%Y-%m-%d')

# 新增：导出模式和休眠设置
MODE = 'incr'  # 'full' = 全量导出, 'incr' = 增量导出（只更新新数据）
SLEEP_SEC = 0.8  # 每次请求后休眠秒数，友好对待Wind配额

ROOT = Path(__file__).resolve().parents[2]  # 项目根目录 ETF_allweather/
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)
CONFIG_CSV = ROOT / 'config' / 'config_export_etf_and_index_rawdata.csv'
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
    if not CONFIG_CSV.is_file():
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


def fetch_series_for_ticker(ticker: str, display_name: str, start: str = None, end: str = None) -> pd.Series | None:
    """获取单个标的数据，支持自定义日期范围"""
    _start = start or start_date
    _end = end or end_date
    print(f"正在获取 {ticker} ({display_name}) | {_start} -> {_end} ...")
    wsd_data = w.wsd(ticker, "PCT_CHG", _start, _end, "Period=D;Days=Trading")
    
    # 友好休眠，避免触发Wind配额限制
    time.sleep(random.uniform(SLEEP_SEC * 0.8, SLEEP_SEC * 1.2))
    
    if wsd_data.ErrorCode != 0:
        print(f"  [WARN] 获取 {ticker} 数据失败，错误代码: {wsd_data.ErrorCode}")
        return None
    if not wsd_data.Times:
        print(f"  [WARN] {ticker} 无数据返回")
        return None
    df = pd.DataFrame(wsd_data.Data, index=wsd_data.Fields, columns=wsd_data.Times).T
    # 转换为小数收益率
    sr = df['PCT_CHG'] / 100.0
    sr.name = display_name
    print(f"  [OK] 获取 {len(sr)} 条数据")
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

print(f"\n导出模式: {'全量' if MODE == 'full' else '增量'} | 休眠: {SLEEP_SEC}s")

# 增量模式：读取现有文件，计算起始日期
etf_start, idx_start = start_date, start_date
existing_etf, existing_idx = None, None

if MODE == 'incr':
    out_etf_path = DATA_DIR / "ETF_daily_simple_returns.csv"
    out_idx_path = DATA_DIR / "index_daily_simple_returns.csv"
    if out_etf_path.exists():
        existing_etf = pd.read_csv(out_etf_path, index_col=0, parse_dates=True)
        last_dt = pd.to_datetime(existing_etf.index.max())
        etf_start = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')  # 回填7天
        print(f"ETF 增量模式：从 {etf_start} 开始（现有 {len(existing_etf)} 行）")
    if out_idx_path.exists():
        existing_idx = pd.read_csv(out_idx_path, index_col=0, parse_dates=True)
        last_dt = pd.to_datetime(existing_idx.index.max())
        idx_start = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')
        print(f"指数 增量模式：从 {idx_start} 开始（现有 {len(existing_idx)} 行）")

# 1) ETF 宽表（按配置顺序累积所有 ETF）
etf_df = pd.DataFrame()
for _, r in cfg.iterrows():
    s = fetch_series_for_ticker(r['etf_code'], r.get('etf_name', r['etf_code']), etf_start, end_date)
    if s is None:
        continue
    etf_df = s.to_frame() if etf_df.empty else etf_df.join(s, how='outer')

# 2) 指数宽表（与 ETF 一一对应，顺序相同）
idx_df = pd.DataFrame(index=etf_df.index if not etf_df.empty else None)
for _, r in cfg.iterrows():
    s = fetch_series_for_ticker(r['index_code'], r.get('index_name', r['index_code']), idx_start, end_date)
    if s is None:
        continue
    idx_df = s.to_frame() if idx_df.empty else idx_df.join(s, how='outer')

# 增量模式：合并新旧数据
if MODE == 'incr':
    if existing_etf is not None and not etf_df.empty:
        etf_df = etf_df.combine_first(existing_etf)
    elif existing_etf is not None:
        etf_df = existing_etf
    if existing_idx is not None and not idx_df.empty:
        idx_df = idx_df.combine_first(existing_idx)
    elif existing_idx is not None:
        idx_df = existing_idx

# 写出
if etf_df.empty and idx_df.empty:
    print("[WARN] 无可写出的数据，已跳过。")
else:
    if not etf_df.empty:
        etf_df.index.name = '日期'
        etf_df = etf_df.sort_index()
        out_etf = DATA_DIR / "ETF_daily_simple_returns.csv"
        etf_df.to_csv(out_etf, header=True, index=True, encoding='utf-8-sig')
        print(f"已写出 ETF 文件: {out_etf} | 行数={len(etf_df)} 列数={etf_df.shape[1]}")
    if not idx_df.empty:
        idx_df.index.name = '日期'
        idx_df = idx_df.sort_index()
        out_idx = DATA_DIR / "index_daily_simple_returns.csv"
        idx_df.to_csv(out_idx, header=True, index=True, encoding='utf-8-sig')
        print(f"已写出 指数 文件: {out_idx} | 行数={len(idx_df)} 列数={idx_df.shape[1]}")

print("\n数据获取完成，已写出 ETF 与指数总表到 data/ 目录。")

# --- 5. 关闭 Wind API ---
w.stop()