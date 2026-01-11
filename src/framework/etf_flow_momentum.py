# -*- coding: utf-8 -*-
"""etf_flow_momentum.py

ETF份额动量择时模块：
基于ETF份额变化（资金流入/流出）来调整股票类ETF的风险权重。

核心逻辑：
1. ETF总市值 = 份额 × 单位净值
2. 净值导致的市值变化 = 上月份额 × (当月净值 - 上月净值)
3. 份额导致的市值变化 = 当月净值 × (当月份额 - 上月份额)
   这个指标反映了资金的真实流入/流出

4. 份额动量信号：
   - 份额增加（资金流入）→ 市场情绪积极 → 增加股票ETF权重
   - 份额减少（资金流出）→ 市场情绪消极 → 减少股票ETF权重
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple
import os

try:
    from .logging_config import get_logger
except ImportError:
    from logging_config import get_logger

logger = get_logger("framework.etf_flow_momentum")


def load_etf_aum_data(
    data_dir: str = 'data',
    amount_file: str = 'ETF_aum_amount.csv',
    nav_file: str = 'ETF_aum_unit_value.csv',
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载ETF份额和单位净值数据。
    
    Parameters
    ----------
    data_dir : str
        数据目录
    amount_file : str
        份额数据文件名
    nav_file : str  
        单位净值数据文件名
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        (份额数据, 单位净值数据)
    """
    amount_path = os.path.join(data_dir, amount_file)
    nav_path = os.path.join(data_dir, nav_file)
    
    if not os.path.exists(amount_path):
        raise FileNotFoundError(f"未找到ETF份额数据文件: {amount_path}")
    if not os.path.exists(nav_path):
        raise FileNotFoundError(f"未找到ETF净值数据文件: {nav_path}")
    
    # 读取份额数据 (跳过标题行，第1行是日期行)
    amount_df = pd.read_csv(amount_path, skiprows=[0], index_col=0, parse_dates=True)
    # 读取净值数据 (跳过标题行，第1行是日期行)
    nav_df = pd.read_csv(nav_path, skiprows=[0, 1], index_col=0, parse_dates=True)
    
    # 转换为数值型
    amount_df = amount_df.apply(pd.to_numeric, errors='coerce')
    nav_df = nav_df.apply(pd.to_numeric, errors='coerce')
    
    # 按日期排序
    amount_df = amount_df.sort_index()
    nav_df = nav_df.sort_index()
    
    logger.info(f"加载ETF AUM数据: 份额 {amount_df.shape}, 净值 {nav_df.shape}")
    
    return amount_df, nav_df


def calculate_flow_driven_aum_change(
    amount_df: pd.DataFrame,
    nav_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    计算份额驱动的ETF市值变化。
    
    公式: 份额导致的市值变化 = 当月净值 × (当月份额 - 上月份额)
    
    Parameters
    ----------
    amount_df : pd.DataFrame
        ETF份额数据 (月频)
    nav_df : pd.DataFrame
        ETF单位净值数据 (月频)
        
    Returns
    -------
    pd.DataFrame
        份额驱动的市值变化 (亿元)
    """
    # 对齐两个DataFrame
    common_cols = amount_df.columns.intersection(nav_df.columns)
    common_idx = amount_df.index.intersection(nav_df.index)
    
    amount_aligned = amount_df.loc[common_idx, common_cols]
    nav_aligned = nav_df.loc[common_idx, common_cols]
    
    # 计算份额变化
    share_change = amount_aligned.diff()  # 当月份额 - 上月份额
    
    # 计算份额驱动的市值变化
    # 份额变化 × 当月净值 = 资金流入/流出金额
    flow_driven_change = share_change * nav_aligned
    
    logger.debug(f"计算份额驱动市值变化: {flow_driven_change.shape}")
    
    return flow_driven_change


def calculate_etf_flow_momentum(
    flow_driven_change: pd.DataFrame,
    lookback_months: int = 3,
    method: str = 'sum',  # 'sum', 'mean', 'weighted'
) -> pd.Series:
    """
    计算ETF资金流动量动量信号。
    
    Parameters
    ----------
    flow_driven_change : pd.DataFrame
        份额驱动的市值变化
    lookback_months : int
        回看月数
    method : str
        聚合方法:
        - 'sum': 累计流入
        - 'mean': 平均流入
        - 'weighted': 时间加权 (近期权重更高)
        
    Returns
    -------
    pd.Series
        资金流动量信号 (正=流入, 负=流出)，index为日期
    """
    if method == 'sum':
        # 简单累计
        momentum = flow_driven_change.rolling(lookback_months).sum().sum(axis=1)
    elif method == 'mean':
        # 平均流入
        momentum = flow_driven_change.rolling(lookback_months).mean().sum(axis=1)
    elif method == 'weighted':
        # 时间加权 (指数衰减)
        weights = np.array([2 ** i for i in range(lookback_months)])
        weights = weights / weights.sum()
        
        def weighted_sum(x):
            if len(x) < lookback_months:
                return np.nan
            return np.sum(x[-lookback_months:] * weights)
        
        momentum = flow_driven_change.sum(axis=1).rolling(lookback_months).apply(weighted_sum, raw=True)
    else:
        raise ValueError(f"不支持的聚合方法: {method}")
    
    return momentum


def calculate_stock_etf_aggregate_flow(
    flow_driven_change: pd.DataFrame,
    stock_etf_keywords: List[str] = None,
    lookback_months: int = 3,
) -> pd.Series:
    """
    计算股票类ETF的聚合资金流动量。
    
    Parameters
    ----------
    flow_driven_change : pd.DataFrame
        份额驱动的市值变化
    stock_etf_keywords : List[str], optional
        股票类ETF的名称关键词，用于筛选
    lookback_months : int
        回看月数
        
    Returns
    -------
    pd.Series
        股票类ETF聚合资金流动量
    """
    if stock_etf_keywords is None:
        # 默认的股票类ETF关键词
        stock_etf_keywords = [
            '300', '500', '1000', '50', 'A500',
            '创业板', '科创', '红利', '证券', '券商',
            '消费', '银行', '军工', '医药', '半导体', 
            '芯片', '机器人', '人工智能', '化工', '电池',
            '光伏', '创新药', '金融科技', '电力',
            '房地产', '有色', '煤炭',
        ]
    
    # 筛选股票类ETF列
    stock_cols = []
    for col in flow_driven_change.columns:
        if any(kw in col for kw in stock_etf_keywords):
            stock_cols.append(col)
    
    if not stock_cols:
        logger.warning("未找到股票类ETF列，使用全部列")
        stock_cols = list(flow_driven_change.columns)
    
    logger.debug(f"筛选到 {len(stock_cols)} 个股票类ETF")
    
    # 计算聚合流动量
    stock_flow = flow_driven_change[stock_cols].sum(axis=1)
    
    # 滚动累计
    aggregate_flow = stock_flow.rolling(lookback_months).sum()
    
    return aggregate_flow


def get_flow_momentum_signal(
    aggregate_flow: pd.Series,
    dt: pd.Timestamp,
    signal_method: str = 'percentile',  # 'percentile', 'zscore', 'sign'
    lookback_for_signal: int = 12,  # 用于计算分位数/z-score的回看月数
) -> float:
    """
    根据资金流动量计算择时信号。
    
    Parameters
    ----------
    aggregate_flow : pd.Series
        聚合资金流动量序列
    dt : pd.Timestamp
        当前日期
    signal_method : str
        信号计算方法:
        - 'percentile': 历史分位数 (0-1)
        - 'zscore': 标准化分数
        - 'sign': 简单正负判断 (1, 0, -1)
    lookback_for_signal : int
        计算信号的历史回看月数
        
    Returns
    -------
    float
        择时信号值
    """
    # 获取截止到dt的数据
    dt_prev_month_end = (pd.to_datetime(dt) - pd.offsets.MonthEnd(1))
    available_flow = aggregate_flow[aggregate_flow.index <= dt_prev_month_end]
    
    if len(available_flow) < 3:
        logger.debug(f"[{dt.date()}] 资金流数据不足，返回中性信号")
        return 0.5 if signal_method == 'percentile' else 0.0
    
    current_flow = available_flow.iloc[-1]
    
    if pd.isna(current_flow):
        return 0.5 if signal_method == 'percentile' else 0.0
    
    # 用于计算信号的历史数据
    hist_flow = available_flow.iloc[-lookback_for_signal:] if len(available_flow) >= lookback_for_signal else available_flow
    
    if signal_method == 'percentile':
        # 计算当前值在历史中的分位数 (0-1)
        rank = (hist_flow < current_flow).sum() / len(hist_flow)
        return float(rank)
    
    elif signal_method == 'zscore':
        # 计算z-score
        mean = hist_flow.mean()
        std = hist_flow.std()
        if std == 0 or pd.isna(std):
            return 0.0
        zscore = (current_flow - mean) / std
        return float(np.clip(zscore, -3, 3))  # 限制在[-3, 3]
    
    elif signal_method == 'sign':
        # 简单正负判断
        if current_flow > 0:
            return 1.0
        elif current_flow < 0:
            return -1.0
        else:
            return 0.0
    
    else:
        raise ValueError(f"不支持的信号方法: {signal_method}")


def get_stock_etf_weight_adjustment(
    aggregate_flow: pd.Series,
    dt: pd.Timestamp,
    base_weight: float = 1.0,
    max_overweight: float = 0.2,  # 最大超配比例
    max_underweight: float = 0.2,  # 最大低配比例
    signal_method: str = 'percentile',
    lookback_for_signal: int = 12,
) -> float:
    """
    根据资金流动量计算股票ETF权重调整系数。
    
    Parameters
    ----------
    aggregate_flow : pd.Series
        聚合资金流动量序列
    dt : pd.Timestamp
        当前日期
    base_weight : float
        基础权重系数 (1.0表示不调整)
    max_overweight : float
        最大超配比例 (如0.2表示最多超配20%)
    max_underweight : float
        最大低配比例 (如0.2表示最多低配20%)
    signal_method : str
        信号计算方法
    lookback_for_signal : int
        信号回看月数
        
    Returns
    -------
    float
        权重调整系数 (如1.1表示增配10%)
    """
    signal = get_flow_momentum_signal(
        aggregate_flow, dt, signal_method, lookback_for_signal
    )
    
    if signal_method == 'percentile':
        # 分位数在0-1之间
        # 0.5为中性，<0.5低配，>0.5超配
        # 线性映射到权重调整
        if signal >= 0.5:
            # 超配
            adjustment = (signal - 0.5) * 2 * max_overweight
        else:
            # 低配
            adjustment = (signal - 0.5) * 2 * max_underweight
        
    elif signal_method == 'zscore':
        # z-score通常在[-3, 3]之间
        # 0为中性，正值超配，负值低配
        if signal >= 0:
            adjustment = (signal / 3) * max_overweight
        else:
            adjustment = (signal / 3) * max_underweight
            
    elif signal_method == 'sign':
        # 简单的+1/-1信号
        if signal > 0:
            adjustment = max_overweight
        elif signal < 0:
            adjustment = -max_underweight
        else:
            adjustment = 0.0
    else:
        adjustment = 0.0
    
    weight_factor = base_weight + adjustment
    weight_factor = np.clip(weight_factor, base_weight - max_underweight, base_weight + max_overweight)
    
    logger.debug(f"[{dt.date()}] 资金流信号={signal:.3f}, 权重调整={weight_factor:.3f}")
    
    return float(weight_factor)


class ETFFlowMomentumTimer:
    """
    ETF份额动量择时器。
    
    封装资金流动量计算和择时信号生成的完整流程。
    
    Parameters
    ----------
    data_dir : str
        数据目录
    lookback_months : int
        计算动量的回看月数
    signal_method : str
        信号计算方法 ('percentile', 'zscore', 'sign')
    max_overweight : float
        最大超配比例
    max_underweight : float
        最大低配比例
    stock_etf_keywords : List[str], optional
        股票类ETF关键词
    """
    
    def __init__(
        self,
        data_dir: str = 'data',
        lookback_months: int = 3,
        signal_method: str = 'percentile',
        max_overweight: float = 0.2,
        max_underweight: float = 0.2,
        stock_etf_keywords: List[str] = None,
        lookback_for_signal: int = 12,
    ):
        self.data_dir = data_dir
        self.lookback_months = lookback_months
        self.signal_method = signal_method
        self.max_overweight = max_overweight
        self.max_underweight = max_underweight
        self.stock_etf_keywords = stock_etf_keywords
        self.lookback_for_signal = lookback_for_signal
        
        # 初始化数据
        self._load_and_process_data()
    
    def _load_and_process_data(self):
        """加载并处理数据"""
        try:
            self.amount_df, self.nav_df = load_etf_aum_data(self.data_dir)
            self.flow_driven_change = calculate_flow_driven_aum_change(
                self.amount_df, self.nav_df
            )
            self.aggregate_flow = calculate_stock_etf_aggregate_flow(
                self.flow_driven_change,
                self.stock_etf_keywords,
                self.lookback_months,
            )
            self.is_valid = True
            logger.info(f"ETF资金流数据加载成功: {len(self.aggregate_flow)} 个月")
        except Exception as e:
            logger.error(f"加载ETF资金流数据失败: {e}")
            self.is_valid = False
            self.aggregate_flow = pd.Series(dtype=float)
    
    def get_weight_adjustment(self, dt: pd.Timestamp) -> float:
        """
        获取指定日期的股票ETF权重调整系数。
        
        Parameters
        ----------
        dt : pd.Timestamp
            当前日期
            
        Returns
        -------
        float
            权重调整系数 (1.0表示不调整)
        """
        if not self.is_valid:
            return 1.0
        
        return get_stock_etf_weight_adjustment(
            self.aggregate_flow,
            dt,
            base_weight=1.0,
            max_overweight=self.max_overweight,
            max_underweight=self.max_underweight,
            signal_method=self.signal_method,
            lookback_for_signal=self.lookback_for_signal,
        )
    
    def get_signal(self, dt: pd.Timestamp) -> float:
        """
        获取原始择时信号值。
        
        Parameters
        ----------
        dt : pd.Timestamp
            当前日期
            
        Returns
        -------
        float
            信号值
        """
        if not self.is_valid:
            return 0.5 if self.signal_method == 'percentile' else 0.0
        
        return get_flow_momentum_signal(
            self.aggregate_flow,
            dt,
            self.signal_method,
            self.lookback_for_signal,
        )
    
    def get_flow_data(self) -> pd.DataFrame:
        """获取完整的资金流数据用于分析"""
        return pd.DataFrame({
            'aggregate_flow': self.aggregate_flow,
        })


def identify_stock_etf_columns(
    etf_columns: List[str],
    stock_keywords: List[str] = None,
) -> List[str]:
    """
    从ETF列名中识别股票类ETF。
    
    Parameters
    ----------
    etf_columns : List[str]
        所有ETF列名
    stock_keywords : List[str], optional
        股票类ETF关键词
        
    Returns
    -------
    List[str]
        股票类ETF列名列表
    """
    if stock_keywords is None:
        stock_keywords = [
            '300ETF', '500ETF', '1000ETF', '50ETF', 'A500',
            '创业板', '科创', '红利',
        ]
    
    stock_etfs = []
    for col in etf_columns:
        if any(kw in col for kw in stock_keywords):
            stock_etfs.append(col)
    
    return stock_etfs


__all__ = [
    'load_etf_aum_data',
    'calculate_flow_driven_aum_change',
    'calculate_etf_flow_momentum',
    'calculate_stock_etf_aggregate_flow',
    'get_flow_momentum_signal',
    'get_stock_etf_weight_adjustment',
    'ETFFlowMomentumTimer',
    'identify_stock_etf_columns',
]
