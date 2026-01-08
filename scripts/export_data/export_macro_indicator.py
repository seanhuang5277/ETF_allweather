# -*- coding: utf-8 -*-
"""
@Project: ETF Strategy
@File   : export_macro_indicator.py
@Description: 
    根据 config/config_macro_factor.csv 配置，从 Wind 导出宏观因子数据。
    结合了 export_wind.py 的 EDB 批处理逻辑和 export_raw_data.py 的简单结构。
    输出文件: data/macro_factors.csv (日频，非日频数据已前向填充)

@Usage:
    python export_macro_indicator.py
"""

from __future__ import annotations
import sys
import time
import math
import random
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# ---- WindPy ----
try:
    from WindPy import w
except ImportError:
    print("请先安装并配置 WindPy：pip install WindPy")
    sys.exit(1)

# --- 配置 ---
START_DATE = '2005-01-01'
END_DATE = datetime.today().strftime('%Y-%m-%d')
BATCH_SIZE = 20  # EDB 每次请求的代码数量

# 新增：导出模式和休眠设置
EXPORT_MODE = 'incr'  # 'full' = 全量导出, 'incr' = 增量导出
SLEEP_SEC = 0.8  # 每次请求后休眠秒数

# 路径设置 (Pathlib)
ROOT = Path(__file__).resolve().parents[2]  # 项目根目录 ETF_allweather/
DATA_DIR = ROOT / 'data'
CONFIG_DIR = ROOT / 'config'
CONFIG_FILE = CONFIG_DIR / 'config_macro_factor.csv'
CONFIG_MIMICKING_FILE = CONFIG_DIR / 'config_factor_mimicking.csv'
OUTPUT_FILE = DATA_DIR / 'macro_factors_raw.csv'
OUTPUT_MIMICKING_FILE = DATA_DIR / 'macro_factors_mimicking_raw.csv'

DATA_DIR.mkdir(exist_ok=True)


def ensure_wind():
    """确保 Wind API 已连接"""
    if not w.isconnected():
        print("正在连接 Wind API ...")
        w.start()
        time.sleep(0.5)
    
    if not w.isconnected():
        print("Wind API 连接失败，请检查 Wind 终端是否已登录。")
        sys.exit(1)
    print("Wind API 连接成功。")


def load_config(config_path: Path) -> pd.DataFrame:
    """读取配置文件"""
    if not config_path.exists():
        print(f"错误: 找不到配置文件 {config_path}")
        sys.exit(1)
        
    try:
        # 尝试不同编码读取
        try:
            df = pd.read_csv(config_path, encoding='utf-8-sig')
        except UnicodeDecodeError:
            df = pd.read_csv(config_path, encoding='gbk')
            
        # 检查必要列
        required_cols = ['宏观变量代码', '宏观变量名', 'api']
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"配置文件缺少必要列: {missing}")
            
        # 清洗数据
        df['code'] = df['宏观变量代码'].astype(str).str.strip()
        df['name'] = df['宏观变量名'].astype(str).str.strip()
        df['api'] = df['api'].astype(str).str.strip().str.lower()
        
        # 过滤无效行
        df = df[df['code'] != 'nan']
        
        print(f"成功读取配置 {config_path.name}，共 {len(df)} 个指标。")
        return df
        
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        sys.exit(1)


def edb_fetch_batch(codes: list[str], start: str, end: str) -> pd.DataFrame:
    """
    批量拉取 EDB 数据 (参考 export_wind.py)
    返回: DataFrame (index=date, columns=codes)
    """
    if not codes:
        return pd.DataFrame()
        
    codes_str = ",".join(codes)
    # EDB 默认 field='close' 即值
    # 修改: 宏观数据不应使用前值填充 (Fill=Previous)，否则会导致累计值跨年填充错误
    # 应使用 Fill=Blank (缺失值用 NaN 表示)
    res = w.edb(codes_str, start, end, "Fill=Blank") 
    
    if res.ErrorCode != 0:
        print(f"[Error] w.edb 失败 (Error={res.ErrorCode}): {res.Data}")
        return pd.DataFrame()
        
    if not res.Times:
        return pd.DataFrame()
        
    dates = pd.to_datetime(res.Times)
    df = pd.DataFrame(index=dates)
    
    # 处理返回数据结构
    # res.Data 是 list of list (按列)
    if len(res.Data) == len(res.Codes):
        for i, code in enumerate(res.Codes):
            # 注意：Wind 返回的 Code 可能大小写不一致，这里尽量匹配
            df[code] = res.Data[i]
    else:
        print(f"[Warn] EDB 数据维度不匹配: Codes={len(res.Codes)}, Data={len(res.Data)}")
        
    return df


def wsd_fetch_single(code: str, start: str, end: str, field: str = "close") -> pd.Series | None:
    """
    单个拉取 WSD 数据
    """
    # 默认参数
    options = "Period=D;Days=Trading;Fill=Previous"
    
    # 如果 field 是 value，对于 WSD 可能是 close 或其他，这里默认 close，除非配置指定
    # 简单起见，如果配置里写了 value，我们尝试用 close，或者直接传 field
    wind_field = field if field.lower() != 'value' else "close"
    
    res = w.wsd(code, wind_field, start, end, options)
    
    if res.ErrorCode != 0:
        print(f"[Error] w.wsd 失败 {code} (Error={res.ErrorCode})")
        return None
        
    if not res.Times:
        return None
        
    dates = pd.to_datetime(res.Times)
    return pd.Series(res.Data[0], index=dates, name=code)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Export Macro Indicators from Wind')
    parser.add_argument('--mode', type=str, default='all', choices=['macro', 'mimicking', 'all'],
                        help='Export mode: macro (original), mimicking (high-freq), or all (both)')
    args = parser.parse_args()

    ensure_wind()

    tasks = []
    if args.mode in ['macro', 'all']:
        tasks.append((CONFIG_FILE, OUTPUT_FILE, "Macro Factors (Original)"))
    if args.mode in ['mimicking', 'all']:
        tasks.append((CONFIG_MIMICKING_FILE, OUTPUT_MIMICKING_FILE, "High-Freq Mimicking Factors"))

    for cfg_path, out_path, desc in tasks:
        print(f"\n{'='*20} Running Task: {desc} {'='*20}")
        run_export_task(cfg_path, out_path)

    w.stop()

def run_export_task(config_path: Path, output_path: Path):
    print(f"Loading config from: {config_path}")
    df_config = load_config(config_path)
    
    # 增量模式：读取现有文件，计算起始日期
    existing_df = None
    actual_start = START_DATE
    print(f"\n导出模式: {'全量' if EXPORT_MODE == 'full' else '增量'} | 休眠: {SLEEP_SEC}s")
    
    if EXPORT_MODE == 'incr' and output_path.exists():
        existing_df = pd.read_csv(output_path, index_col=0, parse_dates=True)
        last_dt = pd.to_datetime(existing_df.index.max())
        actual_start = (last_dt - pd.Timedelta(days=7)).strftime('%Y-%m-%d')  # 回填7天
        print(f"增量模式：从 {actual_start} 开始（现有 {len(existing_df)} 行）")
    
    # 1. 分类 API
    edb_rows = df_config[df_config['api'] == 'edb']
    wsd_rows = df_config[df_config['api'] == 'wsd'] # 假设非 EDB 都是 WSD，或者明确标记
    # 如果有其他类型，需额外处理。这里简单处理。
    
    all_series_list = []
    
    # 2. 批量获取数据
    
    # --- 处理 EDB ---
    if not edb_rows.empty:
        edb_codes = edb_rows['code'].tolist()
        edb_names = edb_rows['name'].tolist()
        code_map = dict(zip(edb_codes, edb_names))
        
        total_edb = len(edb_codes)
        print(f"\n开始下载 EDB 数据 (共 {total_edb} 个)...")
        
        for i in range(0, total_edb, BATCH_SIZE):
            batch_codes = edb_codes[i : i + BATCH_SIZE]
            print(f"  > Fetching batch {i//BATCH_SIZE + 1}/{math.ceil(total_edb/BATCH_SIZE)}: {len(batch_codes)} codes")
            
            try:
                df_batch = edb_fetch_batch(batch_codes, actual_start, END_DATE)
                if not df_batch.empty:
                    # 重命名列为中文名称
                    # 注意：Wind 返回的列名可能是大写，需要不区分大小写匹配
                    renamed_cols = {}
                    for col in df_batch.columns:
                        # 尝试直接匹配
                        if col in code_map:
                            renamed_cols[col] = code_map[col]
                        else:
                            # 尝试忽略大小写匹配
                            for c_cfg in code_map:
                                if c_cfg.upper() == col.upper():
                                    renamed_cols[col] = code_map[c_cfg]
                                    break
                    
                    df_batch = df_batch.rename(columns=renamed_cols)
                    
                    # 将每一列转为 Series 加入列表
                    for col_name in df_batch.columns:
                        all_series_list.append(df_batch[col_name])
                        
            except Exception as e:
                print(f"  [Error] Batch failed: {e}")
            
            time.sleep(random.uniform(SLEEP_SEC * 0.8, SLEEP_SEC * 1.2))

    # --- 处理 WSD ---
    if not wsd_rows.empty:
        print(f"\n开始下载 WSD 数据 (共 {len(wsd_rows)} 个)...")
        for _, row in wsd_rows.iterrows():
            code = row['code']
            name = row['name']
            field = row.get('field', 'close') # 默认取收盘价，如果 config 有 field 列则用 field
            
            print(f"  > Fetching {code} ({name})...")
            try:
                s = wsd_fetch_single(code, actual_start, END_DATE, field)
                if s is not None:
                    s.name = name
                    all_series_list.append(s)
            except Exception as e:
                print(f"  [Error] Failed {code}: {e}")
            
            time.sleep(random.uniform(SLEEP_SEC * 0.5, SLEEP_SEC * 0.8))

    # 3. 合并数据
    if not all_series_list:
        print("\n[Warn] 未获取到任何数据。")
        return

    print("\n正在合并数据...")
    # 使用 outer join 合并所有序列
    full_df = pd.concat(all_series_list, axis=1).sort_index()
    
    # 4. 增量模式：合并新旧数据
    if EXPORT_MODE == 'incr' and existing_df is not None:
        full_df = full_df.combine_first(existing_df)
    
    # 截取时间段
    full_df = full_df[START_DATE:END_DATE]
    
    # 5. 保存
    full_df.index.name = 'date'
    full_df.to_csv(output_path, encoding='utf-8-sig')
    
    print(f"\n导出完成！")
    print(f"文件路径: {output_path}")
    print(f"数据范围: {full_df.index[0].date()} 至 {full_df.index[-1].date()}")
    print(f"包含指标: {len(full_df.columns)} 个")
    print(f"数据行数: {len(full_df)}")

if __name__ == "__main__":
    main()
