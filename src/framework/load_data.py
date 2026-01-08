# -*- coding: utf-8 -*-
"""data.py

数据加载与预处理工具：统一读数、清洗、对齐与 AUM 月转日。

函数:
    load_returns_and_aum(returns_path, aum_path, *,
                         returns_index_col='日期', 
                         aum_index_col='日期') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]
返回:
    all_simple_returns_df: 简单收益率(日)
    all_log_returns_df: 对数收益率(日，已清洗)
    market_aum_daily_df: AUM 日频（由月频前向填充）
    all_assets: 资产代码列表（按列顺序）
"""
from __future__ import annotations
import os
import pandas as pd
import numpy as np

try:
    # 作为包被导入时（推荐）：使用相对导入
    from .logging_config import get_logger  # type: ignore
except Exception:
    # 当直接运行本文件时（无父包信息）：回退为绝对导入
    from logging_config import get_logger  # type: ignore

logger = get_logger("framework.data")


def _to_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.apply(pd.to_numeric, errors='coerce')


def _read_csv_robust(path: str) -> pd.DataFrame:
    last_err: Exception | None = None
    for enc in ("utf-8-sig", "gbk", "cp936", "utf-8"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取CSV失败: {path} | {last_err}")


def load_etf_index_returns_by_category(
    data_dir: str = 'data',
    *,
    returns_index_col: str = '日期',
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """从合并总表与配置生成分类ETF与指数日度简单收益。

    - 读取 `data/ETF_daily_simple_returns.csv` 与 `data/index_daily_simple_returns.csv`
    - 读取 `config/config_export_etf_and_index_rawdata.csv`，仅保留 F 列为 'Y'
    - 按 `资产类别` 输出两个字典：{类别: DataFrame}
    列依据配置中的 `ETF名称`/`Index名称` 精确匹配。
    """
    # 路径
    etf_path = os.path.join(data_dir, 'ETF_daily_simple_returns.csv')
    idx_path = os.path.join(data_dir, 'index_daily_simple_returns.csv')
    if not os.path.isfile(etf_path):
        raise FileNotFoundError(f"未找到ETF返回总表: {etf_path}")
    if not os.path.isfile(idx_path):
        raise FileNotFoundError(f"未找到指数返回总表: {idx_path}")

    # 读取总表
    etf_all = pd.read_csv(etf_path, index_col=returns_index_col, parse_dates=True)
    idx_all = pd.read_csv(idx_path, index_col=returns_index_col, parse_dates=True)
    etf_all = _to_numeric_df(etf_all).sort_index()
    idx_all = _to_numeric_df(idx_all).sort_index()

    # 读取配置
    from pathlib import Path
    # 默认相对工程根目录的 config 路径
    root = Path(__file__).resolve().parents[2]  # framework -> src -> repo root
    config_csv = root / 'config' / 'config_export_etf_and_index_rawdata.csv'
    if not os.path.isfile(config_csv):
        raise FileNotFoundError(f"未找到配置文件: {config_csv}")
    cfg = _read_csv_robust(config_csv)

    # 校验必要列
    required = ['ETF名称', 'Index名称', '资产类别', '是否调用回测[Y/N]']
    miss = [c for c in required if c not in cfg.columns]
    if miss:
        raise ValueError(f"配置缺少必要列: {', '.join(miss)} | 已检测到列: {', '.join(map(str, cfg.columns))}")

    # 过滤 F 列为 Y
    mask_y = cfg['是否调用回测[Y/N]'].astype(str).str.upper().str.strip() == 'Y'
    cfg_y = cfg.loc[mask_y].copy()
    if cfg_y.empty:
        logger.warning("配置中未找到标记为 'Y' 的资产，返回空结果。")
        return {}, {}

    # 按资产类别分类收集列
    etf_by_cat: dict[str, pd.DataFrame] = {}
    idx_by_cat: dict[str, pd.DataFrame] = {}
    for cat, sub in cfg_y.groupby('资产类别'):
        etf_cols = [n for n in sub['ETF名称'].astype(str) if n in etf_all.columns]
        idx_cols = [n for n in sub['Index名称'].astype(str) if n in idx_all.columns]

        # 日历对齐：使用各自已有索引
        etf_cat = etf_all[etf_cols].copy() if etf_cols else pd.DataFrame(index=etf_all.index)
        idx_cat = idx_all[idx_cols].copy() if idx_cols else pd.DataFrame(index=idx_all.index)
        etf_by_cat[str(cat)] = etf_cat.sort_index()
        idx_by_cat[str(cat)] = idx_cat.sort_index()

        # 记录可能缺失的列，便于排查
        missing_etf = [n for n in sub['ETF名称'].astype(str) if n not in etf_all.columns]
        missing_idx = [n for n in sub['Index名称'].astype(str) if n not in idx_all.columns]
        if missing_etf:
            logger.warning(f"分类 {cat}: ETF 列未在总表中找到: {missing_etf}")
        if missing_idx:
            logger.warning(f"分类 {cat}: 指数 列未在总表中找到: {missing_idx}")

    logger.info(f"按资产类别载入完成 | 类别数={len(idx_by_cat)}")
    return etf_by_cat, idx_by_cat


def load_category_returns(
    data_dir: str = 'data',
    returns_index_col: str = '日期',
) -> dict[str, pd.DataFrame]:
    """按分类读取原始日度简单收益。

    返回字典：{category: DataFrame}，DataFrame 可能为空（只有索引无列）。
    不再合并为单一总表，保持分类的独立性，便于策略分别处理。
    文件命名约定：index_daily_simple_returns_<category>.csv
    """
    categories = ['equity', 'bond', 'commodity', 'gold', 'currency']
    out: dict[str, pd.DataFrame] = {}
    union_index = None
    for cat in categories:
        fp = os.path.join(data_dir, f'index_daily_simple_returns_{cat}.csv')
        if os.path.isfile(fp):
            try:
                df = pd.read_csv(fp, index_col=returns_index_col, parse_dates=True)
                df = _to_numeric_df(df)
                out[cat] = df
                union_index = df.index if union_index is None else union_index.union(df.index)
                logger.info(f"分类 {cat} 加载成功: {fp} | 形状={df.shape}")
            except Exception as e:
                logger.warning(f"分类 {cat} 读取失败 {fp}: {e}")
        else:
            logger.info(f"分类 {cat} 文件缺失: {fp}")
    if union_index is None:
        raise FileNotFoundError(f"目录 {data_dir} 下未发现任何分类收益文件")
    # 为缺失或读取失败的分类生成空 DataFrame（只有索引），保持统一索引
    for cat in categories:
        if cat not in out:
            out[cat] = pd.DataFrame(index=union_index)
    # 统一按日期排序
    for cat, df in out.items():
        out[cat] = df.sort_index()
    return out


def compute_log_returns(simple_returns_df: pd.DataFrame) -> pd.DataFrame:
    """从简单收益 DataFrame 生成对数收益并清洗无效值。"""
    log_df = np.log1p(simple_returns_df)
    log_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    log_df.dropna(how='all', inplace=True)
    return _to_numeric_df(log_df)


def load_market_aum(
    aum_path: str,
    *,
    aum_index_col: str = '日期',
    target_index: pd.Index | None = None,
) -> pd.DataFrame:
    """读取月度（或日度）AUM，并可选对齐到指定的 target_index（日频）前向填充。"""
    aum_df = pd.read_csv(aum_path, index_col=aum_index_col, parse_dates=True)
    aum_df = _to_numeric_df(aum_df)
    if target_index is not None:
        aum_df = aum_df.reindex(target_index, method='ffill')
    return aum_df


def load_returns_and_aum(
    returns_path: str | None,
    aum_path: str,
    returns_index_col: str = '日期',
    aum_index_col: str = '日期',
    separate: bool = False,
) -> tuple[pd.DataFrame | dict[str, pd.DataFrame], pd.DataFrame | dict[str, pd.DataFrame], pd.DataFrame, list]:
    """兼容旧接口的总加载函数：

    - 如果 separate=False：保持旧行为（多分类合并为一个 simple DataFrame）。
    - 如果 separate=True：返回分类字典 {cat: simple_df} 与对应的 {cat: log_df}。
    - AUM 通过独立函数加载并对齐到（合并后或 union）索引。
    """
    if separate:
        cat_simple = load_category_returns('data', returns_index_col=returns_index_col)
        # 计算每个分类的 log returns
        cat_log: dict[str, pd.DataFrame] = {k: compute_log_returns(v) if not v.empty else v for k, v in cat_simple.items()}
        # 统一索引（所有分类的并集）
        union_index = None
        for df in cat_log.values():
            union_index = df.index if union_index is None else union_index.union(df.index)
        aum_daily_df = load_market_aum(aum_path, aum_index_col=aum_index_col, target_index=union_index)
        # 资产列表：合并所有非空分类的列
        assets = []
        for df in cat_log.values():
            assets.extend([c for c in df.columns if c not in assets])
        logger.info(f"数据加载完成 (分类模式) | 分类数: {len(cat_simple)} | 资产数: {len(assets)}")
        return cat_simple, cat_log, aum_daily_df, assets

    # 非分类模式：按旧逻辑，如果传入目录/None 自动合并所有分类文件
    if returns_path is None or (isinstance(returns_path, str) and os.path.isdir(returns_path)):
        cat_simple = load_category_returns(returns_path or 'data', returns_index_col=returns_index_col)
        combined = None
        for df in cat_simple.values():
            if df.empty:
                continue
            combined = df if combined is None else combined.join(df, how='outer')
        if combined is None:
            raise ValueError("无任何可合并的分类收益数据")
        all_simple_returns_df = combined.sort_index()
    else:
        raw = pd.read_csv(returns_path, index_col=returns_index_col, parse_dates=True)
        all_simple_returns_df = raw.copy()

    all_log_returns_df = compute_log_returns(all_simple_returns_df)
    # 对齐 simple 到 log 索引
    all_simple_returns_df = all_simple_returns_df.loc[all_log_returns_df.index]
    aum_daily_df = load_market_aum(aum_path, aum_index_col=aum_index_col, target_index=all_log_returns_df.index)
    assets = all_log_returns_df.columns.tolist()
    logger.info(f"数据加载完成 | 资产数: {len(assets)} | 交易日: {len(all_log_returns_df)}")
    return all_simple_returns_df, all_log_returns_df, aum_daily_df, assets


__all__ = [
    'load_category_returns',
    'compute_log_returns',
    'load_market_aum',
    'load_returns_and_aum',
    'load_etf_index_returns_by_category',
]
