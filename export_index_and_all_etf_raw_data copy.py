# -*- coding: utf-8 -*-
"""
从配置 Excel(xlsx) 读取 ETF-指数一一映射，导出两份总表：
    - ETF_daily_simple_returns.csv       (所有 ETF 日度简单收益宽表)
    - index_daily_simple_returns.csv     (所有指数 日度简单收益宽表)

说明：
    - Excel 文件：config/config_export_etf_and_index_rawdata.xlsx
    - 固定列名（精确匹配）：ETF代码, ETF名称, Index代码, Index名称, 资产类别, 是否调用回测[Y/N]
    - 本脚本忽略“是否调用回测[Y/N]”列，不再按资产类别拆文件
"""

import os
import re
import pandas as pd
import numpy as np
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
# 配置文件（XLSX）路径
CONFIG_XLSX = os.path.join(CONFIG_DIR, 'config_export_etf_and_index_rawdata.xlsx')
# Excel 工作表：默认第 1 张表（如需指定名称可改成字符串）
CONFIG_SHEET = 0

# 配置新增列：一个指数对应的全部跟踪基金列表（逗号分隔）
TRACKING_FUNDS_COL = '全部跟踪该指数的基金'

# 只保留国内场内 ETF（剔除 .OF 等场外基金）
DOMESTIC_ETF_SUFFIX = ('.SH', '.SZ')
# --- 2. 简单工具函数 ---



def _normalize_code(x: str) -> str:
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return ''
    s = re.sub(r"\.0$", "", s)
    s = s.upper()
    return s


def _parse_code_list(x) -> list[str]:
    if x is None:
        return []
    s = str(x).strip()
    if s == '' or s.lower() == 'nan':
        return []
    # 去掉可能的引号包裹
    s = s.strip('"').strip("'")
    parts = [p.strip() for p in s.split(',') if str(p).strip()]
    codes: list[str] = []
    for p in parts:
        c = _normalize_code(p)
        if c:
            codes.append(c)
    return codes


def _load_config_from_df(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.dropna(how='all')
    raw = raw.loc[:, ~raw.columns.astype(str).str.strip().duplicated()]

    # 精确匹配所需列（与配置 CSV 完全一致）
    expected_cols = ['ETF代码', 'ETF名称', 'Index代码', 'Index名称', '资产类别']
    missing = [c for c in expected_cols if c not in raw.columns]
    if missing:
        existing_cols = ', '.join(map(str, raw.columns.tolist()))
        raise ValueError(f"配置缺少必要列: {', '.join(missing)} | 已检测到列: {existing_cols}")

    tracking_raw = raw[TRACKING_FUNDS_COL] if TRACKING_FUNDS_COL in raw.columns else ''

    df = pd.DataFrame({
        'category': raw['资产类别'].astype(str).str.strip(),
        'etf_code': raw['ETF代码'].map(_normalize_code),
        'etf_name': raw['ETF名称'].astype(str).str.strip(),
        'index_code': raw['Index代码'].map(_normalize_code),
        'index_name': raw['Index名称'].astype(str).str.strip(),
        'tracking_funds_all': tracking_raw,
    })

    df['tracking_funds_all'] = df['tracking_funds_all'].apply(_parse_code_list)

    df = df[(df['etf_code'] != '') & (df['index_code'] != '')]
    df = df.drop_duplicates(subset=['category', 'etf_code', 'index_code'], keep='first')
    return df


def load_config() -> pd.DataFrame:
    if not os.path.isfile(CONFIG_XLSX):
        raise FileNotFoundError(f"未找到配置 Excel 文件: {CONFIG_XLSX}")
    try:
        raw = pd.read_excel(CONFIG_XLSX, sheet_name=CONFIG_SHEET)
    except Exception as e:
        raise ValueError(f"Excel 读取失败: {CONFIG_XLSX} | sheet={CONFIG_SHEET} | {e}")
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


def _wsd_fetch_series(ticker: str, field: str, display_name: str) -> pd.Series | None:
    res = w.wsd(ticker, field, start_date, end_date, "Period=D;Days=Trading")
    if res.ErrorCode != 0:
        return None
    df = pd.DataFrame(res.Data, index=res.Fields, columns=res.Times).T
    col = res.Fields[0]
    s = pd.to_numeric(df[col], errors='coerce')
    if s.notna().sum() == 0:
        return None
    s.name = display_name
    return s


def _wsd_fetch_first_available(ticker: str, field_candidates: list[str], display_name: str) -> tuple[str | None, pd.Series | None]:
    last = None
    for f in field_candidates:
        s = _wsd_fetch_series(ticker, f, display_name)
        if s is not None:
            return f, s
        last = f
    return last, None


def _filter_domestic_etfs(codes: list[str]) -> list[str]:
    return [c for c in codes if c.endswith(DOMESTIC_ETF_SUFFIX)]


def _safe_sum(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    return df.sum(axis=1, skipna=True)


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

# # 1) ETF 宽表（按配置顺序累积所有 ETF）
# etf_df = pd.DataFrame()
# etf_ret_by_code: dict[str, pd.Series] = {}
# for _, r in cfg.iterrows():
#     s = fetch_series_for_ticker(r['etf_code'], r.get('etf_name', r['etf_code']))
#     if s is None:
#         continue
#     etf_ret_by_code[r['etf_code']] = s.copy()
#     etf_df = s.to_frame() if etf_df.empty else etf_df.join(s, how='outer')

# # 2) 指数宽表（与 ETF 一一对应，顺序相同）
# idx_df = pd.DataFrame(index=etf_df.index if not etf_df.empty else None)
# idx_ret_by_code: dict[str, pd.Series] = {}
# for _, r in cfg.iterrows():
#     s = fetch_series_for_ticker(r['index_code'], r.get('index_name', r['index_code']))
#     if s is None:
#         continue
#     idx_ret_by_code[r['index_code']] = s.copy()
#     idx_df = s.to_frame() if idx_df.empty else idx_df.join(s, how='outer')

# # 写出
# if etf_df.empty and idx_df.empty:
#     print("[WARN] 无可写出的数据，已跳过。")
# else:
#     if not etf_df.empty:
#         etf_df.index.name = '日期'
#         etf_df = etf_df.sort_index()
#         out_etf = os.path.join(DATA_DIR, "ETF_daily_simple_returns.csv")
#         etf_df.to_csv(out_etf, header=True, index=True, encoding='utf-8-sig')
#         print(f"已写出 ETF 文件: {out_etf} | 行数={len(etf_df)} 列数={etf_df.shape[1]}")
#     if not idx_df.empty:
#         idx_df.index.name = '日期'
#         idx_df = idx_df.sort_index()
#         out_idx = os.path.join(DATA_DIR, "index_daily_simple_returns.csv")
#         idx_df.to_csv(out_idx, header=True, index=True, encoding='utf-8-sig')
#         print(f"已写出 指数 文件: {out_idx} | 行数={len(idx_df)} 列数={idx_df.shape[1]}")

# print("\n数据获取完成，已写出 ETF 与指数总表到 data/ 目录。")

# --- 4.1 额外导出 (1)：跟踪该指数的全部基金（筛选 ETF）规模分解聚合 ---
print("\n开始导出：指数-跟踪ETF规模分解聚合数据...")

aum_fields = ["VAL_MV_ARD", "VAL_MV", "FUND_NETASSET", "NETASSET_TOTAL"]
shares_fields = ["UNIT_TOTAL", "FUND_SHARE", "FUND_TOTALSHARE", "TOTAL_SHARE"]
nav_fields = ["NAV", "NAV_ADJ", "FUND_NAV", "NAV_UNIT"]

fund_metric_cache: dict[str, pd.DataFrame] = {}
impact_rows: list[pd.DataFrame] = []

for _, r in cfg.iterrows():
    index_code = r['index_code']
    index_name = r.get('index_name', index_code)
    all_funds = r.get('tracking_funds_all', [])
    etf_funds = _filter_domestic_etfs(all_funds)
    etf_funds = sorted(set(etf_funds))

    if not etf_funds:
        print(f"  [SKIP] {index_code}({index_name}) 跟踪基金列为空或无场内ETF")
        continue

    print(f"  处理 {index_code}({index_name}) | ETF数量={len(etf_funds)}")
    per_fund_metrics: dict[str, pd.DataFrame] = {}
    for fund_code in etf_funds:
        if fund_code in fund_metric_cache:
            per_fund_metrics[fund_code] = fund_metric_cache[fund_code]
            continue

        f_aum, s_aum = _wsd_fetch_first_available(fund_code, aum_fields, "aum")
        f_shr, s_shr = _wsd_fetch_first_available(fund_code, shares_fields, "shares")
        f_nav, s_nav = _wsd_fetch_first_available(fund_code, nav_fields, "nav")

        if s_aum is None or s_shr is None or s_nav is None:
            print(f"    [WARN] {fund_code} 指标缺失 | aum={f_aum if s_aum is not None else 'NA'} shares={f_shr if s_shr is not None else 'NA'} nav={f_nav if s_nav is not None else 'NA'}")
            continue

        dfm = pd.concat([s_aum, s_shr, s_nav], axis=1)
        dfm = dfm.sort_index()
        dfm.columns = ["aum", "shares", "nav"]
        fund_metric_cache[fund_code] = dfm
        per_fund_metrics[fund_code] = dfm

    if not per_fund_metrics:
        print(f"    [SKIP] {index_code}({index_name}) 无可用ETF数据")
        continue

    # 逐基金计算影响，再按日期聚合
    aum_mat = pd.DataFrame({c: m['aum'] for c, m in per_fund_metrics.items()})
    shares_mat = pd.DataFrame({c: m['shares'] for c, m in per_fund_metrics.items()})
    nav_mat = pd.DataFrame({c: m['nav'] for c, m in per_fund_metrics.items()})

    aum_diff = aum_mat.diff()
    shares_diff = shares_mat.diff()
    shares_effect = nav_mat * shares_diff
    nav_effect = aum_diff - shares_effect

    out = pd.DataFrame(index=aum_mat.index)
    out['总基金规模'] = _safe_sum(aum_mat)
    out['每日基金规模变化'] = _safe_sum(aum_diff)
    out['每日份额对规模变化影响'] = _safe_sum(shares_effect)
    out['每日净值对规模变化影响'] = _safe_sum(nav_effect)
    out['Index代码'] = index_code
    out['Index名称'] = index_name
    out['ETF数量'] = len(per_fund_metrics)

    out = out.dropna(how='all', subset=['总基金规模', '每日基金规模变化', '每日份额对规模变化影响', '每日净值对规模变化影响'])
    impact_rows.append(out)

if impact_rows:
    impact_long = pd.concat(impact_rows, axis=0)
    impact_long.index.name = '日期'
    impact_long = impact_long.reset_index()
    out_path = os.path.join(DATA_DIR, 'index_tracking_etf_impact_long.csv')
    impact_long.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已写出：{out_path} | 行数={len(impact_long)}")
else:
    print("[WARN] 未生成任何指数-跟踪ETF规模分解数据")


# --- 4.2 额外导出 (2)：指数本身总成交额 ---
print("\n开始导出：指数总成交额...")

idx_amt_rows: list[pd.DataFrame] = []
idx_amt_field_candidates = ["AMT", "TURNOVER", "S_DQ_AMOUNT"]

for _, r in cfg.iterrows():
    index_code = r['index_code']
    index_name = r.get('index_name', index_code)
    _, s_amt = _wsd_fetch_first_available(index_code, idx_amt_field_candidates, "amt")
    if s_amt is None:
        print(f"  [WARN] {index_code}({index_name}) 无法获取成交额")
        continue
    tmp = s_amt.to_frame(name='总成交额').copy()
    tmp['Index代码'] = index_code
    tmp['Index名称'] = index_name
    idx_amt_rows.append(tmp)

if idx_amt_rows:
    idx_amt_long = pd.concat(idx_amt_rows, axis=0)
    idx_amt_long.index.name = '日期'
    idx_amt_long = idx_amt_long.reset_index()
    out_path = os.path.join(DATA_DIR, 'index_turnover_amount_long.csv')
    idx_amt_long.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已写出：{out_path} | 行数={len(idx_amt_long)}")
else:
    print("[WARN] 未生成任何指数成交额数据")


# --- 4.3 额外导出 (3)：主ETF折溢价、成交额、跟踪误差 ---
print("\n开始导出：主ETF折溢价/成交额/跟踪误差...")

main_rows: list[pd.DataFrame] = []
etf_amt_field_candidates = ["AMT", "TURNOVER", "S_DQ_AMOUNT"]
etf_nav_field_candidates = ["NAV", "NAV_ADJ", "FUND_NAV", "NAV_UNIT"]
etf_close_field_candidates = ["CLOSE", "S_DQ_CLOSE"]

for _, r in cfg.iterrows():
    etf_code = r['etf_code']
    etf_name = r.get('etf_name', etf_code)
    index_code = r['index_code']
    index_name = r.get('index_name', index_code)

    # 折溢价 = 收盘价 / (净值或IOPV近似) - 1
    _, s_close = _wsd_fetch_first_available(etf_code, etf_close_field_candidates, "close")
    _, s_nav = _wsd_fetch_first_available(etf_code, etf_nav_field_candidates, "nav")
    _, s_amt = _wsd_fetch_first_available(etf_code, etf_amt_field_candidates, "amt")

    if s_close is None or s_nav is None:
        print(f"  [WARN] {etf_code}({etf_name}) 无法获取 close/nav，跳过折溢价")
        premium = None
    else:
        premium = (s_close / s_nav) - 1.0
        premium.name = '折溢价'

    etf_ret = etf_ret_by_code.get(etf_code)
    idx_ret = idx_ret_by_code.get(index_code)
    if etf_ret is None or idx_ret is None:
        te = None
    else:
        diff_ret = (etf_ret - idx_ret).dropna()
        te = diff_ret.rolling(252).std() * np.sqrt(252)
        te.name = '跟踪误差_252日'

    cols: list[pd.Series] = []
    if premium is not None:
        cols.append(premium)
    if s_amt is not None:
        s_amt2 = s_amt.copy()
        s_amt2.name = '成交额'
        cols.append(s_amt2)
    if te is not None:
        cols.append(te)

    if not cols:
        continue

    tmp = pd.concat(cols, axis=1)
    tmp['ETF代码'] = etf_code
    tmp['ETF名称'] = etf_name
    tmp['Index代码'] = index_code
    tmp['Index名称'] = index_name
    main_rows.append(tmp)

if main_rows:
    main_long = pd.concat(main_rows, axis=0)
    main_long.index.name = '日期'
    main_long = main_long.reset_index()
    out_path = os.path.join(DATA_DIR, 'main_etf_metrics_long.csv')
    main_long.to_csv(out_path, index=False, encoding='utf-8-sig')
    print(f"已写出：{out_path} | 行数={len(main_long)}")
else:
    print("[WARN] 未生成任何主ETF指标数据")

# --- 5. 关闭 Wind API ---
w.stop()