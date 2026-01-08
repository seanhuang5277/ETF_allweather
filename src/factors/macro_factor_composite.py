# -*- coding: utf-8 -*-
"""
@Project: ETF Strategy
@File   : macro_factor_composite_v2.py
@Description: 
    宏观因子 (Macro Factors) 构造模块 - V2 版本
    
    核心改进: 每个因子导出两个维度
    ============================================
    1. Level (绝对水位): 当前处于什么位置
       - 用于判断: 高通胀/低通胀, 强增长/弱增长
       - 适合: 象限判断, 资产配置大方向
       
    2. Change (边际变化): 趋势在改善还是恶化
       - 用于判断: 通胀升温/降温, 增长加速/减速
       - 适合: 战术调仓, 趋势跟踪
    
    最终导出因子:
    ============================================
    | 因子名称          | Level列名              | Change列名              |
    |-------------------|------------------------|-------------------------|
    | CN_Growth         | CN_Growth_Level        | CN_Growth_Change        |
    | CN_Inflation      | CN_Inflation_Level     | CN_Inflation_Change     |
    | US_Growth         | US_Growth_Level        | US_Growth_Change        |
    | US_Inflation      | US_Inflation_Level     | US_Inflation_Change     |
    | CN_RiskAppetite   | CN_RiskAppetite_Level  | CN_RiskAppetite_Change  |
    | CN_Monetary       | CN_Monetary_Level      | CN_Monetary_Change      |

@Usage:
    python macro_factor_composite_v2.py
"""

import os
import sys
import io
from pathlib import Path

# 设置标准输出编码为 UTF-8（解决 VS Code 中文乱码）
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- 路径配置 ---
ROOT = Path(__file__).resolve().parents[2]  # src/factors -> src -> 项目根目录

# 确保 sys.path 包含根目录和 src 目录，支持 VS Code 直接运行
root_str = str(ROOT)
src_str = str(ROOT / 'src')
if root_str not in sys.path:
    sys.path.insert(0, root_str)
if src_str not in sys.path:
    sys.path.insert(0, src_str)

DATA_DIR = ROOT / 'data'
OUTPUT_ALL_FACTORS_FILE = DATA_DIR / 'all_macro_factors.csv'
PLOT_DIR = DATA_DIR / 'plots' / 'macro_factors'

# 确保目录存在
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# Matplotlib 中文配置
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 参数配置 ---
EWM_SPAN_DAILY = 20     # 日频数据平滑窗口
EWM_SPAN_MONTHLY = 3    # 月频数据平滑窗口
ZSCORE_WINDOW_LEVEL = 36      # Z-Score 标准化窗口 (月)
ZSCORE_WINDOW_CHANGE = 18      # Z-Score 标准化窗口 (月)

# =============================================================================
# 因子权重配置
# =============================================================================

# 增长因子权重
GROWTH_WEIGHTS = {
    'PMI_neworder': 0.3,       # PMI 新订单
    'yield_spread': 0.00,       # 10Y-1Y 利差
    'TSF_yoy': 0.00,            # 社融同比
    'industrial': 0.3,         # 发电量同比
    'commodity': 0.00,          # 大宗商品
    'export': 0.25,             # 出口指标（新增）
    'real_estate': 0.15,        # 房地产指标（新增）
}

# 通胀因子权重
INFLATION_WEIGHTS = {
    'CPI': 0.4,
    'PPI': 0.3,
    'commodity': 0.0,
    'PMI_price': 0.3,
    'FX': 0.0,
    'pig': 0.00,
}

# 美国通胀因子权重
US_INFLATION_WEIGHTS = {
    'BEI': 0.40,
    'commodity': 0.30,
    'official': 0.20,
    'FX': 0.10,
}

# 美国增长因子权重
US_GROWTH_WEIGHTS = {
    'ISM': 0.35,
    'yield_curve': 0.25,
    'copper_gold': 0.25,
    'real_activity': 0.15,
}

# 中国风险偏好因子权重
CN_RISK_APPETITE_WEIGHTS = {
    'AAA_spread': 0.50,
    'AA_spread': 0.50,
}

# 中国货币因子权重
CN_MONETARY_WEIGHTS = {
    'DR007': 0.50,
    'Shibor_3M': 0.30,
    'R007': 0.20,
}


# =============================================================================
# 核心工具函数
# =============================================================================

def load_raw_data():
    """加载原始宏观数据"""
    raw_path = os.path.join(DATA_DIR, 'macro_factors_raw.csv')
    
    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"找不到原始数据文件: {raw_path}")
    
    df = pd.read_csv(raw_path, index_col='date', parse_dates=True)
    df = df.sort_index()
    
    print(f"[Data] 加载原始数据: {raw_path}")
    print(f"[Data] 时间范围: {df.index.min().date()} ~ {df.index.max().date()}")
    print(f"[Data] 列数: {len(df.columns)}")
    
    return df


def rolling_zscore(series, window, min_periods=12):
    """
    滚动 Z-Score 标准化 (避免未来函数)
    """
    mean = series.rolling(window=window, min_periods=min_periods).mean()
    std = series.rolling(window=window, min_periods=min_periods).std()
    
    z = (series - mean) / std
    z = z.clip(-3, 3)
    
    return z

# =============================================================================
# 六类数据处理函数（按 raw_data_type 分类）
# =============================================================================

def process_diffusion(series: pd.Series,
                      *,
                      center_value: float = 50.0,
                      level_ewm_span: int = None,
                      level_rolling_span: int = None,
                      level_zscore_window: int = ZSCORE_WINDOW_LEVEL,
                      change_ewm_span: int = None,
                      change_rolling_span: int = None,
                      change_zscore_window: int = ZSCORE_WINDOW_CHANGE) -> tuple[pd.Series, pd.Series]:
    """
    处理扩散指标（如 PMI）- 月度数据
    
    特点:
    - 扩散指标有天然中枢（通常是 50）
    - Level: 相对于中枢的偏离程度的 Z-Score
    - Change: 月度环比变化的 Z-Score
    
    Args:
        series: 原始月度序列（如 PMI = 51.2）
        center_value: 扩散指标的中枢值（PMI 为 50）
        level_ewm_span: 平滑窗口（月），用于降噪
        level_rolling_span: 滚动窗口（月），用于降噪(用于季节性数据)
        level_zscore_window: Level 的 Z-Score 窗口
        change_ewm_span: 平滑窗口（月），用于降噪
        change_rolling_span: 滚动窗口（月），用于降噪(用于季节性数据)
        change_zscore_window: Change 的 Z-Score 窗口
    
    Returns:
        (level_signal, change_signal)
    """
    if series is None or len(series.dropna()) < 12:
        return None, None
    
    # 确保月度频率
    monthly = series.resample('ME').last().dropna()
    
    # Level: 相对中枢的偏离
    deviation = monthly - center_value
    if level_ewm_span is not None:
        deviation_smooth = deviation.ewm(span=level_ewm_span).mean()
    elif level_rolling_span is not None:
        deviation_smooth = deviation.rolling(window=level_rolling_span).mean()
    else:
        deviation_smooth = deviation
    level_signal = rolling_zscore(deviation_smooth, window=level_zscore_window)
    
    # Change: 月度环比变化
    change = deviation_smooth.diff()
    if change_ewm_span is not None:
        change_smooth = change.ewm(span=change_ewm_span).mean()
    elif change_rolling_span is not None:
        change_smooth = change.rolling(window=change_rolling_span).mean()
    else:
        change_smooth = change
    change_signal = rolling_zscore(change_smooth, window=change_zscore_window)    

    return level_signal, change_signal


def process_monthly_value(series: pd.Series,
                          ewm_span_level: int = None,
                          rolling_span_level: int = None,
                          zscore_window_level: int = ZSCORE_WINDOW_LEVEL,
                          ewm_span_change: int = None,
                          rolling_span_change: int = None,
                          zscore_window_change: int = ZSCORE_WINDOW_CHANGE) -> tuple[pd.Series, pd.Series]:
    """
    处理当月值指标（如社融当月值、发电量当月值）- 月度数据
    
    特点:
    - 原始数据是绝对量，需要转换为同比或增速
    - Level: 同比增速的 Z-Score
    - Change: 同比增速的月度变化的 Z-Score
    
    Args:
        series: 原始月度序列（绝对值）
        use_yoy: 是否转换为同比（默认 True）
        ewm_span_level: 平滑窗口
        rolling_span_level: 滚动窗口
        zscore_window_level: Level 的 Z-Score 窗口
        ewm_span_change: 平滑窗口
        rolling_span_change: 滚动窗口
        zscore_window_change: Change 的 Z-Score 窗口
    
    Returns:
        (level_signal, change_signal)
    """
    if series is None or len(series.dropna()) < 24:  # 需要至少 2 年数据计算同比
        return None, None
    
    monthly = series.resample('ME').last().dropna()
    

    transformed = monthly.pct_change(periods=12, fill_method=None) * 100

    if ewm_span_level is not None:
        transformed_smooth = transformed.ewm(span=ewm_span_level).mean()
    elif rolling_span_level is not None:
        transformed_smooth = transformed.rolling(window=rolling_span_level).mean()
    else:
        transformed_smooth = transformed
    
    # Level
    level_signal = rolling_zscore(transformed_smooth, window=zscore_window_level)
    
    if ewm_span_change is not None:
        transformed_smooth = transformed_smooth.ewm(span=ewm_span_change).mean()
    elif rolling_span_change is not None:
        transformed_smooth = transformed_smooth.rolling(window=rolling_span_change).mean()
    else:
        transformed_smooth = transformed_smooth
    # Change
    change = transformed_smooth.diff()
    change_signal = rolling_zscore(change, window=zscore_window_change)
    
    return level_signal, change_signal


def process_yoy(series: pd.Series,
                *,
                ewm_span_level: int = None,
                rolling_span_level: int = None,
                zscore_window_level: int = ZSCORE_WINDOW_LEVEL,
                ewm_span_change: int = None,
                rolling_span_change: int = None,
                zscore_window_change: int = ZSCORE_WINDOW_CHANGE) -> tuple[pd.Series, pd.Series]:
    """
    处理同比指标（如 CPI 同比、PPI 同比）- 月度数据
    
    特点:
    - 数据本身已经是同比，直接使用
    - Level: 同比值的 Z-Score
    - Change: 同比值的月度变化的 Z-Score
    
    Args:
        series: 同比序列（如 CPI 同比 = 2.5%）
        ewm_span: 平滑窗口
        zscore_window_level: Level 的 Z-Score 窗口
        zscore_window_change: Change 的 Z-Score 窗口
    
    Returns:
        (level_signal, change_signal)
    """
    if series is None or len(series.dropna()) < 12:
        return None, None
    
    monthly = series.resample('ME').last().dropna()

    if ewm_span_level is not None:
        monthly_smooth = monthly.ewm(span=ewm_span_level).mean()
    elif rolling_span_level is not None:
        monthly_smooth = monthly.rolling(window=rolling_span_level).mean()
    else:
        monthly_smooth = monthly
    
    # Level: 直接对同比值做 Z-Score
    level_signal = rolling_zscore(monthly_smooth, window=zscore_window_level)
    
    if ewm_span_change is not None:
        monthly_smooth = monthly_smooth.ewm(span=ewm_span_change).mean()
    elif rolling_span_change is not None:
        monthly_smooth = monthly_smooth.rolling(window=rolling_span_change).mean()
    else:
        monthly_smooth = monthly_smooth
    # Change: 同比值的变化
    change = monthly_smooth.diff()
    change_signal = rolling_zscore(change, window=zscore_window_change)
    
    return level_signal, change_signal


def process_ytd_cumulative(series: pd.Series,
                           ewm_span_level: int = None,
                           zscore_window_level: int = ZSCORE_WINDOW_LEVEL,
                           ewm_span_change: int = None,
                           zscore_window_change: int = ZSCORE_WINDOW_CHANGE) -> tuple[pd.Series, pd.Series]:
    """
    处理当年累计值指标（如房地产投资累计值）- 月度数据
    
    特点:
    - 累计值需要先转换为当月值，再计算同比
    - Level: 当月同比增速的 Z-Score
    - Change: 同比增速的月度变化的 Z-Score
    
    Args:
        series: 当年累计值序列
        ewm_span_level: Level 的平滑窗口
        zscore_window_level: Level 的 Z-Score 窗口
        ewm_span_change: Change 的平滑窗口
        zscore_window_change: Change 的 Z-Score 窗口
    
    Returns:
        (level_signal, change_signal)
    """
    if series is None or len(series.dropna()) < 24:
        return None, None
    
    monthly = series.resample('ME').last().dropna()
    
    # 累计值转当月值：当月值 = 本月累计 - 上月累计
    # 但1月份的累计值就是当月值，需要特殊处理
    monthly_value = monthly.copy()
    
    for i in range(1, len(monthly_value)):
        current_date = monthly_value.index[i]
        prev_date = monthly_value.index[i - 1]
        
        # 如果是1月，当月值 = 累计值
        if current_date.month == 1:
            continue  # 保持原值
        # 如果上月是12月（跨年），当月值 = 累计值
        elif prev_date.month == 12:
            continue  # 保持原值（1月的累计值就是当月值）
        else:
            # 当月值 = 本月累计 - 上月累计
            monthly_value.iloc[i] = monthly.iloc[i] - monthly.iloc[i - 1]
    
    # 计算同比
    yoy = monthly_value.pct_change(periods=12, fill_method=None) * 100
    yoy_smooth = yoy.ewm(span=ewm_span_level).mean()
    
    # Level
    level_signal = rolling_zscore(yoy_smooth, window=zscore_window_level)
    
    # Change
    change_smooth = yoy.ewm(span=ewm_span_change).mean()
    change = change_smooth.diff()
    change_signal = rolling_zscore(change, window=zscore_window_change)
    
    return level_signal, change_signal


def process_price(series_daily: pd.Series,
                  momentum_windows: list = [60, 120],
                  zscore_window_level: int = ZSCORE_WINDOW_LEVEL * 20,
                  ewm_span_change: int = None,
                  zscore_window_change: int = ZSCORE_WINDOW_CHANGE * 20) -> tuple[pd.Series, pd.Series]:
    """
    处理价格指标（如商品价格、汇率）- 日度数据
    
    特点:
    - 日度数据，需要在日频上处理后再降到月度
    - Level: 基于动量（N日变化率）的 Z-Score
    - Change: 动量的日度变化的 Z-Score，然后取月末
    
    Args:
        series_daily: 日度价格序列
        ewm_span_level: 日频平滑窗口
        ewm_span_change: 日频平滑窗口
        momentum_windows: 动量窗口列表（日），如 [60, 120] 表示 3 个月和 6 个月
        zscore_window_level: Level 的 Z-Score 窗口（日）
        zscore_window_change: Change 的 Z-Score 窗口（日）
    
    Returns:
        (level_signal, change_signal) - 月度频率
    """
    if series_daily is None or len(series_daily.dropna()) < 252:  # 至少 1 年日度数据
        return None, None
    
    daily = series_daily.dropna()
    
    # 计算多窗口动量并平均
    momentums = []
    for window in momentum_windows:
        mom = np.log(daily / daily.shift(window))
        momentums.append(mom)
    
    momentum_combined = pd.concat(momentums, axis=1).mean(axis=1)

    # Level: 动量的 Z-Score（日频计算）
    level_daily = rolling_zscore(momentum_combined, window=zscore_window_level)
    
    # Change: 动量变化的 Z-Score（日频计算）
    
    if ewm_span_change is not None:
        series_smooth = series_daily.ewm(span=ewm_span_change).mean()
    else:
        series_smooth = series_daily
    change_daily = series_smooth.diff(20)  # 20 日 ≈ 1 个月变化
    change_zscore_daily = rolling_zscore(change_daily, window=zscore_window_change)
    
    # 重采样为月度
    level_signal = level_daily.resample('ME').last()
    change_signal = change_zscore_daily.resample('ME').last()
    
    return level_signal, change_signal


def process_rate(series_daily: pd.Series,
                 *,
                 ewm_span_level: int = 20,
                 zscore_window_level: int = ZSCORE_WINDOW_LEVEL * 20,
                 ewm_span_change: int = 20,
                 zscore_window_change: int = ZSCORE_WINDOW_CHANGE * 20) -> tuple[pd.Series, pd.Series]:
    """
    处理利率指标（如国债收益率、DR007）- 日度数据
    
    特点:
    - 日度数据，利率本身就是水平值
    - Level: 利率水平的 Z-Score
    - Change: 利率变化（日度差分）的 Z-Score
    
    Args:
        series_daily: 日度利率序列（如 10 年国债收益率 = 2.65%）
        ewm_span_daily: 日频平滑窗口
        zscore_window_level: Level 的 Z-Score 窗口（日）
        zscore_window_change: Change 的 Z-Score 窗口（日）
    
    Returns:
        (level_signal, change_signal) - 月度频率
    """
    if series_daily is None or len(series_daily.dropna()) < 252:
        return None, None
    
    daily = series_daily.dropna()
    
    # 平滑利率水平
    if ewm_span_level is not None:
        level_smooth = daily.ewm(span=ewm_span_level).mean()
    else:
        level_smooth = daily
    
    # Level: 利率水平的 Z-Score（日频计算）
    level_daily = rolling_zscore(level_smooth, window=zscore_window_level)
    
    # Change: 利率变化（20日差分，约1个月）
    change_daily = level_smooth.diff(20)
    if ewm_span_change is not None:
        change_daily_smooth = change_daily.ewm(span=ewm_span_change).mean()
    else:
        change_daily_smooth = change_daily
    change_zscore_daily = rolling_zscore(change_daily_smooth, window=zscore_window_change)
    
    # 重采样为月度
    level_signal = level_daily.resample('ME').last()
    change_signal = change_zscore_daily.resample('ME').last()
    
    return level_signal, change_signal


def process_rate_spread(series1_daily: pd.Series,
                        series2_daily: pd.Series,
                        *,
                        ewm_span_level: int = 20,
                        zscore_window_level: int = ZSCORE_WINDOW_LEVEL * 20,
                        ewm_span_change: int = 20,
                        zscore_window_change: int = ZSCORE_WINDOW_CHANGE * 20) -> tuple[pd.Series, pd.Series]:
    """
    处理利差指标（如 10Y-1Y 期限利差，信用利差）- 日度数据
    
    Args:
        series1_daily: 第一个利率序列（如 10Y 国债）
        series2_daily: 第二个利率序列（如 1Y 国债）
        其他参数同 process_rate
    
    Returns:
        (level_signal, change_signal) - 月度频率，正值表示利差缩小，负值表示利差扩大
    """
    if series1_daily is None or series2_daily is None:
        return None, None
    
    # 计算利差
    spread = -1 * (series1_daily - series2_daily).dropna()
    
    if len(spread) < 252:
        return None, None
    
    # 使用利率处理函数
    return  process_rate(spread, ewm_span_level = ewm_span_level, 
                                zscore_window_level = zscore_window_level, 
                                ewm_span_change = ewm_span_change, 
                                zscore_window_change = zscore_window_change)


def process_daily_real_estate(series_daily: pd.Series,
                              level_rolling_window: int = 60,
                              change_rolling_window: int = 7,
                              change_diff_window: int = 7,
                              zscore_window_change: int = 365) -> tuple[pd.Series, pd.Series]:
    """
    处理日频房地产成交数据（30大中城市商品房成交面积）
    
    特殊处理逻辑：
    - Level: 季度平均热度（Rolling(60)均值），用 Expanding Z-Score 比全历史
    - Change: 周度改善速度（Rolling(7)平滑后 Diff(7)），用 1年窗口 Z-Score
    
    为什么这样处理：
    - 周末网签暂停，直接用 diff(1) 会产生周期性假信号
    - Rolling(7) 可以完美消除周末效应，分母结构恒定
    - 季度均值显示绝对热度，周度变化显示边际改善
    
    Args:
        series_daily: 日度成交面积序列
        level_rolling_window: Level 的平滑窗口（默认 60 天，约一季度）
        change_rolling_window: Change 的平滑窗口（默认 7 天，一周）
        change_diff_window: Change 的差分窗口（默认 7 天，环比上周）
        zscore_window_change: Change 的 Z-Score 窗口（默认 365 天，1年）
    
    Returns:
        (level_signal, change_signal) - 月度频率
    """
    if series_daily is None or len(series_daily.dropna()) < 365:
        return None, None
    
    daily = series_daily.dropna()
    
    # ========== Level: 季度平均热度 ==========
    # 过去 60 天的日均成交面积，平滑掉所有节假日和周末
    level_smooth = daily.rolling(window=level_rolling_window, min_periods=30).mean()
    
    # 用 Expanding Z-Score（全历史），因为地产是大周期，要跟历史长河比
    level_mean = level_smooth.expanding(min_periods=252).mean()
    level_std = level_smooth.expanding(min_periods=252).std()
    level_zscore_daily = (level_smooth - level_mean) / level_std
    level_zscore_daily = level_zscore_daily.clip(-3, 3)
    
    # ========== Change: 周度边际变化 ==========
    # 1. 先算 7 日移动平均，消除周末效应
    weekly_avg = daily.rolling(window=change_rolling_window, min_periods=5).mean()
    
    # 2. 拿"今天的周均值"和"7天前的周均值"比，相当于本周 vs 上周
    change_daily = weekly_avg.diff(change_diff_window)
    
    # 3. Z-Score（1年窗口）
    change_zscore_daily = rolling_zscore(change_daily, window=zscore_window_change)
    
    # 重采样为月度
    level_signal = level_zscore_daily.resample('ME').last()
    change_signal = change_zscore_daily.resample('ME').last()
    
    return level_signal, change_signal


# =============================================================================
# 兼容函数（向后兼容）
# =============================================================================



def find_column(df, candidates):
    """在 DataFrame 中查找第一个存在的列名"""
    for col in candidates:
        if col in df.columns:
            return col
    return None


# =============================================================================
# 中国增长因子 (CN_Growth)
# =============================================================================

def build_cn_growth_factor(df_raw):
    """
    构建中国增长因子 - 分别导出 Level 和 Change
    """
    print("\n" + "=" * 60)
    print("中国增长因子 (CN_Growth) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    # 1. PMI 新订单 [扩散指标]
    pmi_col = find_column(df_raw, ['中国:制造业PMI:新订单'])
    if pmi_col:
        pmi = df_raw[pmi_col].dropna()
        level, change = process_diffusion(pmi, 
                                         center_value=50.0,                     
                                         level_ewm_span = 3,
                                         level_zscore_window = ZSCORE_WINDOW_LEVEL,
                                         change_ewm_span = None,
                                         change_rolling_span = None,
                                         change_zscore_window = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['PMI_neworder'] = level
            signals_change['PMI_neworder'] = change
            print(f"  [✓] PMI 新订单: {pmi_col} [扩散指标]")
    
    # 2. 10Y-1Y 利差 [利率]
    y10_col = find_column(df_raw, ['中债国债到期收益率:10年', '中国:国债到期收益率:10年'])
    y1_col = find_column(df_raw, ['中债国债到期收益率:1年', '中国:国债到期收益率:1年'])
    if y10_col and y1_col:
        y10 = df_raw[y10_col].dropna()
        y1 = df_raw[y1_col].dropna()
        level, change = process_rate_spread(y10, y1,
                                            ewm_span_level = 60,
                                            zscore_window_level = ZSCORE_WINDOW_LEVEL * 20,
                                            ewm_span_change = 20,
                                            zscore_window_change= ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['yield_spread'] = level
            signals_change['yield_spread'] = change
            print(f"  [✓] 利差: {y10_col} - {y1_col} [利率]")
    
    # 3. 社融同比 [当月值]
    tsf_col = find_column(df_raw, ['中国:社会融资规模:当月值', '社会融资规模:当月值'])
    if tsf_col:
        tsf = df_raw[tsf_col].dropna()
        level, change = process_monthly_value(tsf, 
                                             rolling_span_level = 3,
                                             zscore_window_level = ZSCORE_WINDOW_LEVEL,
                                             ewm_span_change= 3,
                                             zscore_window_change = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['TSF_yoy'] = level
            signals_change['TSF_yoy'] = change
            print(f"  [✓] 社融同比: {tsf_col} [当月值]")
    
    # 4. 发电量同比 [同比]
    elec_col = find_column(df_raw, ['发电量:同比', '中国:发电量:当月同比'])
    if elec_col:
        elec = df_raw[elec_col].dropna()
        level, change = process_yoy(elec,
                                    rolling_span_level = 3,
                                    zscore_window_level = ZSCORE_WINDOW_LEVEL,
                                    ewm_span_change= 3,
                                    zscore_window_change = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['industrial'] = level
            signals_change['industrial'] = change
            print(f"  [✓] 发电量: {elec_col} [同比]")
    
    # 5. 大宗商品 (螺纹钢) [价格]
    rebar_col = find_column(df_raw, ['中国:钢材价格指数:螺纹钢(Φ16)', '螺纹钢价格'])
    if rebar_col:
        rebar_price = df_raw[rebar_col].dropna()
        level, change = process_price(rebar_price, momentum_windows=[60, 120],
                                      zscore_window_level = ZSCORE_WINDOW_LEVEL * 20,
                                      ewm_span_change= 20,
                                      zscore_window_change = ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['commodity'] = level
            signals_change['commodity'] = change
            print(f"  [✓] 商品: {rebar_col} [价格]")
    
    # 6. 出口指标 [扩散+同比 合成]
    # 6a. PMI 新出口订单 [扩散指标]
    export_signals_level = []
    export_signals_change = []
    export_weights = []
    
    pmi_export_col = find_column(df_raw, ['中国:制造业PMI:新出口订单'])
    if pmi_export_col:
        pmi_export = df_raw[pmi_export_col].dropna()
        l, c = process_diffusion(pmi_export, 
                                center_value=50.0,                     
                                level_ewm_span=3,
                                level_zscore_window=ZSCORE_WINDOW_LEVEL,
                                change_ewm_span=None,
                                change_rolling_span=None,
                                change_zscore_window=ZSCORE_WINDOW_CHANGE)
        if l is not None:
            export_signals_level.append(l)
            export_signals_change.append(c)
            export_weights.append(0.5)
            print(f"  [✓] PMI 新出口订单: {pmi_export_col} [扩散指标]")
    
    # 6b. 出口金额当月同比 [同比]
    export_yoy_col = find_column(df_raw, ['中国:出口金额:当月同比'])
    if export_yoy_col:
        export_yoy = df_raw[export_yoy_col].dropna()
        l, c = process_yoy(export_yoy,
                          rolling_span_level=3,
                          zscore_window_level=ZSCORE_WINDOW_LEVEL,
                          ewm_span_change=3,
                          zscore_window_change=ZSCORE_WINDOW_CHANGE)
        if l is not None:
            export_signals_level.append(l)
            export_signals_change.append(c)
            export_weights.append(0.5)
            print(f"  [✓] 出口金额同比: {export_yoy_col} [同比]")
    
    # 合成出口子因子
    if export_signals_level:
        total_w = sum(export_weights)
        export_weights = [w / total_w for w in export_weights]
        
        df_l = pd.concat(export_signals_level, axis=1)
        df_c = pd.concat(export_signals_change, axis=1)
        
        signals_level['export'] = (df_l * export_weights).sum(axis=1)
        signals_change['export'] = (df_c * export_weights).sum(axis=1)
        print(f"  [✓] 出口指标: 加权合成")
    
    # 7. 房地产 (30大中城市商品房成交面积) [日频流量数据]
    re_col = find_column(df_raw, ['中国:30大中城市:成交面积:商品房', '30大中城市商品房成交面积'])
    if re_col:
        re_data = df_raw[re_col].dropna()
        level, change = process_daily_real_estate(re_data,
                                                   level_rolling_window=60,  # 季度均值
                                                   change_rolling_window=7,  # 周平滑
                                                   change_diff_window=7,     # 环比上周
                                                   zscore_window_change=365) # 1年Z窗口
        if level is not None:
            signals_level['real_estate'] = level
            signals_change['real_estate'] = change
            print(f"  [✓] 房地产: {re_col} [日频流量]")
    
    # 合成因子
    cn_growth_level, cn_growth_change = synthesize_factor(
        signals_level, signals_change, GROWTH_WEIGHTS, 'CN_Growth'
    )
    
    return cn_growth_level, cn_growth_change, signals_level, signals_change


# =============================================================================
# 中国通胀因子 (CN_Inflation)
# =============================================================================

def build_cn_inflation_factor(df_raw):
    """
    构建中国通胀因子 - 分别导出 Level 和 Change
    """
    print("\n" + "=" * 60)
    print("中国通胀因子 (CN_Inflation) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    # 1. 核心 CPI [同比]
    cpi_col = find_column(df_raw, ['中国:CPI:不包括食品和能源(核心CPI):当月同比', '核心CPI:当月同比'])
    if cpi_col:
        cpi = df_raw[cpi_col].dropna()
        level, change = process_yoy(cpi,
                                    rolling_span_level = 3,
                                    zscore_window_level = ZSCORE_WINDOW_LEVEL,
                                    ewm_span_change= 3,
                                    zscore_window_change = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['CPI'] = level
            signals_change['CPI'] = change
            print(f"  [✓] 核心CPI: {cpi_col} [同比]")
    
    # 2. PPI [同比]
    ppi_col = find_column(df_raw, ['中国:PPI:当月同比', 'PPI:当月同比'])
    if ppi_col:
        ppi = df_raw[ppi_col].dropna()
        level, change = process_yoy(ppi,
                                    rolling_span_level = 3,
                                    zscore_window_level = ZSCORE_WINDOW_LEVEL,
                                    ewm_span_change= 3,
                                    zscore_window_change = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['PPI'] = level
            signals_change['PPI'] = change
            print(f"  [✓] PPI: {ppi_col} [同比]")
    
    # 3. 大宗商品 (南华工业品 + 原油 + CRB 加权)
    commodity_signals_level = []
    commodity_signals_change = []
    commodity_weights = []
    
    nanhua_col = find_column(df_raw, ['南华工业品指数', '南华综合指数'])
    if nanhua_col:
        nanhua = df_raw[nanhua_col].dropna()
        l, c = process_price(nanhua, momentum_windows=[60,120], # 3个月、6个月动量
                             zscore_window_level = ZSCORE_WINDOW_LEVEL * 20, 
                             ewm_span_change=20,
                             zscore_window_change = ZSCORE_WINDOW_CHANGE * 20)  
        if l is not None:
            commodity_signals_level.append(l)
            commodity_signals_change.append(c)
            commodity_weights.append(0.4)
    
    oil_col = find_column(df_raw, ['期货结算价(连续):布伦特原油', '布伦特原油'])
    if oil_col:
        oil = df_raw[oil_col].dropna()
        l, c = process_price(oil, momentum_windows=[60,120], # 3个月、6个月动量
                            zscore_window_level = ZSCORE_WINDOW_LEVEL * 20, 
                            ewm_span_change=20,
                            zscore_window_change = ZSCORE_WINDOW_CHANGE * 20)  # 6个月动量
        if l is not None:
            commodity_signals_level.append(l)
            commodity_signals_change.append(c)
            commodity_weights.append(0.4)
    
    crb_col = find_column(df_raw, ['CRB现货指数:综合', 'CRB现货指数'])
    if crb_col:
        crb = df_raw[crb_col].dropna()
        l, c = process_price(crb, momentum_windows=[60,120], # 3个月、6个月动量
                            zscore_window_level = ZSCORE_WINDOW_LEVEL * 20, 
                            ewm_span_change=20,
                            zscore_window_change = ZSCORE_WINDOW_CHANGE * 20)  # 6个月动量
        if l is not None:
            commodity_signals_level.append(l)
            commodity_signals_change.append(c)
            commodity_weights.append(0.2)
    
    if commodity_signals_level:
        # 归一化权重
        total_w = sum(commodity_weights)
        commodity_weights = [w / total_w for w in commodity_weights]
        
        df_l = pd.concat(commodity_signals_level, axis=1)
        df_c = pd.concat(commodity_signals_change, axis=1)
        
        signals_level['commodity'] = (df_l * commodity_weights).sum(axis=1)
        signals_change['commodity'] = (df_c * commodity_weights).sum(axis=1)
        print(f"  [✓] 大宗商品: 加权合成")
    
    # 4. PMI 价格 (购进 + 出厂) [扩散指标]
    pmi_in_col = find_column(df_raw, ['中国:制造业PMI:主要原材料购进价格', 'PMI:原材料购进价格'])
    pmi_out_col = find_column(df_raw, ['中国:制造业PMI:出厂价格', 'PMI:出厂价格'])
    
    pmi_price_level = []
    pmi_price_change = []
    
    if pmi_in_col:
        pmi_in = df_raw[pmi_in_col].dropna()
        level, change = process_diffusion(pmi_in, center_value=50.0, 
                                 level_ewm_span = 3,
                                 level_zscore_window = ZSCORE_WINDOW_LEVEL,
                                 change_ewm_span = None,
                                 change_rolling_span = None,
                                 change_zscore_window = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            pmi_price_level.append(level)
            pmi_price_change.append(change)
    
    if pmi_out_col:
        pmi_out = df_raw[pmi_out_col].dropna()
        level, change = process_diffusion(pmi_out, center_value=50.0, 
                                 level_ewm_span = 3,
                                 level_zscore_window = ZSCORE_WINDOW_LEVEL,
                                 change_ewm_span = None,
                                 change_rolling_span = None,
                                 change_zscore_window = ZSCORE_WINDOW_CHANGE)
        if level is not None:
            pmi_price_level.append(level)
            pmi_price_change.append(change)
    
    if pmi_price_level:
        signals_level['PMI_price'] = pd.concat(pmi_price_level, axis=1).mean(axis=1)
        signals_change['PMI_price'] = pd.concat(pmi_price_change, axis=1).mean(axis=1)
        print(f"  [✓] PMI价格")
    
    # 5. 汇率 (CFETS 反向) [价格]
    fx_col = find_column(df_raw, ['CFETS人民币汇率指数', 'CFETS指数'])
    if fx_col:
        fx = df_raw[fx_col].dropna()
        # 反向: CFETS 下降 = 人民币贬值 = 输入性通胀上行
        level, change = process_price(fx, 
                                        momentum_windows=[60,120],
                                        zscore_window_level=ZSCORE_WINDOW_LEVEL,
                                        ewm_span_change=20,
                                        zscore_window_change=ZSCORE_WINDOW_CHANGE * 20
                                           )  # 3个月动量
        if level is not None:
            signals_level['FX'] = - level
            signals_change['FX'] = - change
            print(f"  [✓] 汇率: {fx_col} (反向) [价格]")
    
    # 6. 猪肉价格 [价格]
    pig_col = find_column(df_raw, ['中国:大宗价:猪肉', '猪肉价格'])
    if pig_col:
        pig = df_raw[pig_col].dropna()
        # 猪肉是周度数据，用价格处理函数，看 12 个月动量，约 250 个交易日
        level, change = process_price(pig, momentum_windows=[250],
                                      zscore_window_level=ZSCORE_WINDOW_LEVEL,
                                        ewm_span_change=60,
                                        zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['pig'] = level
            signals_change['pig'] = change
            print(f"  [✓] 猪肉: {pig_col} [价格]")
    
    # 合成因子
    cn_inflation_level, cn_inflation_change = synthesize_factor(
        signals_level, signals_change, INFLATION_WEIGHTS, 'CN_Inflation'
    )
    
    return cn_inflation_level, cn_inflation_change, signals_level, signals_change


# =============================================================================
# 美国通胀因子 (US_Inflation)
# =============================================================================

def build_us_inflation_factor(df_raw):
    """
    构建美国通胀因子 - 分别导出 Level 和 Change
    """
    print("\n" + "=" * 60)
    print("美国通胀因子 (US_Inflation) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    # 1. BEI (盈亏平衡通胀率)
    bei_signals_level = []
    bei_signals_change = []
    bei_weights = []
    
    bei10_col = find_column(df_raw, ['美国:盈亏平衡通胀率:10年:非季调', '美国:盈亏平衡通胀率:10年'])
    if bei10_col:
        bei10 = df_raw[bei10_col].dropna()
        l, c = process_rate(bei10,
                            ewm_span_level = 20,
                            zscore_window_level= ZSCORE_WINDOW_LEVEL * 20,
                            ewm_span_change= 20,
                            zscore_window_change= ZSCORE_WINDOW_CHANGE * 20)  # BEI 是利率类数据
        if l is not None:
            bei_signals_level.append(l)
            bei_signals_change.append(c)
            bei_weights.append(0.6)
    
    bei5_col = find_column(df_raw, ['美国:盈亏平衡通胀率:5年:非季调', '美国:盈亏平衡通胀率:5年'])
    if bei5_col:
        bei5 = df_raw[bei5_col].dropna()
        l, c = process_rate(bei5,  # BEI 是利率类数据
                            ewm_span_level = 20,
                            zscore_window_level= ZSCORE_WINDOW_LEVEL * 20,
                            ewm_span_change= 20,
                            zscore_window_change= ZSCORE_WINDOW_CHANGE * 20)  # BEI 是利率类数据
        if l is not None:
            bei_signals_level.append(l)
            bei_signals_change.append(c)
            bei_weights.append(0.4)
    
    if bei_signals_level:
        total_w = sum(bei_weights)
        bei_weights = [w / total_w for w in bei_weights]
        
        df_l = pd.concat(bei_signals_level, axis=1)
        df_c = pd.concat(bei_signals_change, axis=1)
        
        signals_level['BEI'] = (df_l * bei_weights).sum(axis=1)
        signals_change['BEI'] = (df_c * bei_weights).sum(axis=1)
        print(f"  [✓] BEI: 加权合成")
    
    # 2. 大宗商品 (WTI + CRB)
    commodity_signals_level = []
    commodity_signals_change = []
    commodity_weights = []
    
    wti_col = find_column(df_raw, ['期货结算价(连续):WTI原油', 'WTI原油'])
    if wti_col:
        wti = df_raw[wti_col].dropna()
        level, change = process_price(wti, momentum_windows=[60, 120],
                             zscore_window_level= ZSCORE_WINDOW_LEVEL* 20,
                             ewm_span_change= 20,
                             zscore_window_change= ZSCORE_WINDOW_CHANGE* 20)  # 3个月和6个月动量
        if level is not None:
            commodity_signals_level.append(level)
            commodity_signals_change.append(change)
            commodity_weights.append(0.5)
    
    crb_col = find_column(df_raw, ['CRB现货指数:综合', 'CRB现货指数'])
    if crb_col:
        crb = df_raw[crb_col].dropna()
        level, change = process_price(crb, momentum_windows=[120])  # 6个月动量
        if level is not None:
            commodity_signals_level.append(level)
            commodity_signals_change.append(change)
            commodity_weights.append(0.5)
    
    if commodity_signals_level:
        total_w = sum(commodity_weights)
        commodity_weights = [w / total_w for w in commodity_weights]
        
        df_l = pd.concat(commodity_signals_level, axis=1)
        df_c = pd.concat(commodity_signals_change, axis=1)
        
        signals_level['commodity'] = (df_l * commodity_weights).sum(axis=1)
        signals_change['commodity'] = (df_c * commodity_weights).sum(axis=1)
        print(f"  [✓] 大宗商品: 加权合成")
    
    # 3. 官方通胀 (CPI) [同比]
    cpi_col = find_column(df_raw, ['美国:CPI:非季调:同比', '美国:核心CPI:同比'])
    if cpi_col:
        cpi = df_raw[cpi_col].dropna()
        l, c = process_yoy(cpi, rolling_span_level= 3,
                           zscore_window_level = ZSCORE_WINDOW_LEVEL,
                           ewm_span_change= 3,
                           zscore_window_change= ZSCORE_WINDOW_CHANGE)
        if l is not None:
            signals_level['official'] = l
            signals_change['official'] = c
            print(f"  [✓] 官方CPI: {cpi_col} [同比]")
    
    # 4. 美元指数 (反向) [价格]
    dxy_col = find_column(df_raw, ['美元指数', 'DXY'])
    if dxy_col:
        dxy = df_raw[dxy_col].dropna()
        # 反向: 美元下跌 = 通胀上行
        level, change = process_price(dxy, momentum_windows=[60,120], # 3个月动量和6个月动量
                                      zscore_window_level= ZSCORE_WINDOW_LEVEL* 20,
                                      ewm_span_change= 20,
                                      zscore_window_change= ZSCORE_WINDOW_CHANGE* 20)  
        if level is not None:
            signals_level['FX'] = -level
            signals_change['FX'] = -change
            print(f"  [✓] 美元指数: {dxy_col} (反向) [价格]")
    
    # 合成因子
    us_inflation_level, us_inflation_change = synthesize_factor(
        signals_level, signals_change, US_INFLATION_WEIGHTS, 'US_Inflation'
    )
    
    return us_inflation_level, us_inflation_change, signals_level, signals_change


# =============================================================================
# 美国增长因子 (US_Growth)
# =============================================================================

def build_us_growth_factor(df_raw):
    """
    构建美国增长因子 - 分别导出 Level 和 Change
    """
    print("\n" + "=" * 60)
    print("美国增长因子 (US_Growth) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    # 1. ISM 新订单 [扩散指标]
    ism_col = find_column(df_raw, ['美国:ISM:制造业PMI:新订单', '美国:ISM制造业PMI:新订单'])
    if ism_col:
        ism = df_raw[ism_col].dropna()
        level, change = process_diffusion(ism, center_value=50.0, level_ewm_span=3,
                                         level_zscore_window=ZSCORE_WINDOW_LEVEL,
                                         change_ewm_span=None,
                                         change_rolling_span=None,
                                         change_zscore_window=ZSCORE_WINDOW_CHANGE)
        if level is not None:
            signals_level['ISM'] = level
            signals_change['ISM'] = change
            print(f"  [✓] ISM: {ism_col} [扩散指标]")
    
    # 2. 10Y-2Y 利率曲线 [利率]
    y10_col = find_column(df_raw, ['美国:国债收益率:10年', '美国:10年国债收益率'])
    y2_col = find_column(df_raw, ['美国:国债收益率:2年', '美国:2年国债收益率'])
    if y10_col and y2_col:
        y10 = df_raw[y10_col].dropna()
        y2 = df_raw[y2_col].dropna()
        level, change = process_rate_spread(y10, y2,
                                            ewm_span_level=60,
                                            zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                            ewm_span_change=20,
                                            zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['yield_curve'] = level
            signals_change['yield_curve'] = change
            print(f"  [✓] 利率曲线: {y10_col} - {y2_col} [利率]")
    
    # 3. 铜金比 [价格]
    copper_col = find_column(df_raw, ['期货收盘价(连续):COMEX铜', '期货结算价(活跃合约):COMEX铜'])
    gold_col = find_column(df_raw, ['期货收盘价(连续):COMEX黄金', '期货结算价(活跃合约):COMEX黄金'])
    if copper_col and gold_col:
        copper = df_raw[copper_col].dropna()
        gold = df_raw[gold_col].dropna()
        aligned = pd.DataFrame({'copper': copper, 'gold': gold}).dropna()
        # 铜金比: 用日度数据计算比值
        copper_gold_ratio = aligned['copper'] / aligned['gold']
        level, change = process_price(copper_gold_ratio, momentum_windows=[60, 120],
                                      zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                      ewm_span_change=20,
                                      zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['copper_gold'] = level
            signals_change['copper_gold'] = change
            print(f"  [✓] 铜金比 [价格]")
    
    # 合成因子
    us_growth_level, us_growth_change = synthesize_factor(
        signals_level, signals_change, US_GROWTH_WEIGHTS, 'US_Growth'
    )
    
    return us_growth_level, us_growth_change, signals_level, signals_change


# =============================================================================
# 中国风险偏好因子 (CN_RiskAppetite)
# =============================================================================

def build_cn_risk_appetite_factor(df_raw):
    """
    构建中国风险偏好因子 - 分别导出 Level 和 Change
    注意: 信用利差需要反向处理 (利差收窄 = 风险偏好强 = 正值)
    """
    print("\n" + "=" * 60)
    print("中国风险偏好因子 (CN_RiskAppetite) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    cgb_col = find_column(df_raw, ['中债国债到期收益率:5年'])
    
    # 1. AAA 信用利差 [利率]
    aaa_col = find_column(df_raw, ['中债企业债到期收益率(AAA):5年'])
    if aaa_col and cgb_col:
        aaa = df_raw[aaa_col].dropna()
        cgb = df_raw[cgb_col].dropna()
        # 反向: 利差收窄 = 风险偏好强，所以传入 cgb - aaa（负利差）
        level, change = process_rate_spread(cgb, aaa,  # cgb - aaa = -spread
                                            ewm_span_level=60,  
                                            zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                            ewm_span_change=20,
                                            zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)  
        if level is not None:
            signals_level['AAA_spread'] = level
            signals_change['AAA_spread'] = change
            print(f"  [✓] AAA利差: {aaa_col} (反向) [利率]")
    
    # 2. AA 信用利差 [利率]
    aa_col = find_column(df_raw, ['中债企业债到期收益率(AA):5年', '中债企业债到期收益率(AA+):5年'])
    if aa_col and cgb_col:
        aa = df_raw[aa_col].dropna()
        cgb = df_raw[cgb_col].dropna()
        # 反向
        level, change = process_rate_spread(cgb, aa, # cgb - aaa = -spread
                                            ewm_span_level=60,  
                                            zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                            ewm_span_change=20,
                                            zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)  # cgb - aa = -spread
        if level is not None:
            signals_level['AA_spread'] = level
            signals_change['AA_spread'] = change
            print(f"  [✓] AA利差: {aa_col} (反向) [利率]")
    
    # 3. 信用曲线斜率 (AA - AAA)
    if 'AAA_spread' in signals_level and 'AA_spread' in signals_level:
        # 信用曲线斜率已经隐含在 AA 和 AAA 的差异中，这里不再单独计算
        pass
    
    # 合成因子
    risk_level, risk_change = synthesize_factor(
        signals_level, signals_change, CN_RISK_APPETITE_WEIGHTS, 'CN_RiskAppetite'
    )
    
    return risk_level, risk_change, signals_level, signals_change


# =============================================================================
# 中国货币因子 (CN_Monetary)
# =============================================================================

def build_cn_monetary_factor(df_raw):
    """
    构建中国货币因子 - 分别导出 Level 和 Change
    约定口径：高值 = 货币宽松, 低值 = 货币紧缩
    """
    print("\n" + "=" * 60)
    print("中国货币因子 (CN_Monetary) 构造")
    print("=" * 60)
    
    signals_level = {}
    signals_change = {}
    
    # 1. DR007 [利率]
    dr007_col = find_column(df_raw, ['DR007'])
    if dr007_col:
        dr007 = df_raw[dr007_col].dropna()
        level, change = process_rate(dr007,
                                    ewm_span_level=20,
                                    zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                    ewm_span_change=20,
                                    zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['DR007'] = level
            signals_change['DR007'] = change
            print(f"  [✓] DR007: {dr007_col} [利率]")
    
    # 2. Shibor 3M [利率]
    shibor_col = find_column(df_raw, ['SHIBOR:3个月', 'Shibor:3个月'])
    if shibor_col:
        shibor = df_raw[shibor_col].dropna()
        level, change = process_rate(shibor,
                                    ewm_span_level=20,
                                    zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                    ewm_span_change=20,
                                    zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['Shibor_3M'] = level
            signals_change['Shibor_3M'] = change
            print(f"  [✓] Shibor 3M: {shibor_col} [利率]")
    
    # 3. R007 [利率]
    r007_col = find_column(df_raw, ['R007'])
    if r007_col:
        r007 = df_raw[r007_col].dropna()
        level, change = process_rate(r007,
                                    ewm_span_level=20,
                                    zscore_window_level=ZSCORE_WINDOW_LEVEL * 20,
                                    ewm_span_change=20,
                                    zscore_window_change=ZSCORE_WINDOW_CHANGE * 20)
        if level is not None:
            signals_level['R007'] = level
            signals_change['R007'] = change
            print(f"  [✓] R007: {r007_col} [利率]")
    
    # 合成因子
    monetary_level, monetary_change = synthesize_factor(
        signals_level, signals_change, CN_MONETARY_WEIGHTS, 'CN_Monetary'
    )

    # 方向统一：利率类指标原始含义通常是“利率高=紧缩”，合成后默认也是“高=紧缩”。
    # 这里整体乘以 -1，确保后续研究中“因子为正 = 更宽松”。
    if monetary_level is not None:
        monetary_level = -monetary_level
    if monetary_change is not None:
        monetary_change = -monetary_change
    
    return monetary_level, monetary_change, signals_level, signals_change


# =============================================================================
# 因子合成函数
# =============================================================================

def synthesize_factor(signals_level, signals_change, weights, factor_name):
    """
    合成因子: 分别对 Level 和 Change 进行加权合成
    
    注意: 由于每个子信号已经是 Z-Score，合成后不再做 Z-Score，
    只做 clip(-3, 3) 保护极端值
    """
    print(f"\n  [Synthesis] 合成 {factor_name}...")
    
    if not signals_level:
        print(f"    [Error] 没有有效信号")
        return None, None
    
    # 只保留有数据的权重
    available_weights = {k: v for k, v in weights.items() if k in signals_level}
    
    if not available_weights:
        print(f"    [Error] 没有匹配的权重")
        return None, None
    
    # 归一化权重
    total = sum(available_weights.values())
    available_weights = {k: v / total for k, v in available_weights.items()}
    
    print(f"    使用信号: {list(available_weights.keys())}")
    print(f"    调整后权重: {available_weights}")
    
    # 合成 Level - 加权求和后只做 clip，不再做 Z-Score
    df_level = pd.DataFrame({k: signals_level[k] for k in available_weights.keys()})
    factor_level = pd.Series(0.0, index=df_level.index)
    for name, weight in available_weights.items():
        factor_level = factor_level.add(df_level[name] * weight, fill_value=0.0)
    # 只做 clip，不再做 Z-Score（因为子信号已经是 Z-Score）
    factor_level = factor_level.clip(-3, 3)
    factor_level.name = f'{factor_name}_Level'
    
    # 合成 Change - 同样只做 clip
    df_change = pd.DataFrame({k: signals_change[k] for k in available_weights.keys()})
    factor_change = pd.Series(0.0, index=df_change.index)
    for name, weight in available_weights.items():
        factor_change = factor_change.add(df_change[name] * weight, fill_value=0.0)
    # 只做 clip，不再做 Z-Score
    factor_change = factor_change.clip(-3, 3)
    factor_change.name = f'{factor_name}_Change'
    
    print(f"    Level 有效数据: {factor_level.count()} 个月")
    print(f"    Change 有效数据: {factor_change.count()} 个月")
    
    return factor_level, factor_change


# =============================================================================
# 绘图函数
# =============================================================================

def plot_factor_dual(factor_level, factor_change, factor_name, save_dir):
    """
    绘制因子的 Level 和 Change 双视图
    """
    if factor_level is None or factor_change is None:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 图1: Level (绝对水位)
    ax1 = axes[0]
    ax1.fill_between(factor_level.index, 0, factor_level,
                     where=factor_level >= 0, color='green', alpha=0.3, label='高位')
    ax1.fill_between(factor_level.index, 0, factor_level,
                     where=factor_level < 0, color='red', alpha=0.3, label='低位')
    ax1.plot(factor_level, color='black', linewidth=1.5)
    ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax1.axhline(1, color='green', linestyle=':', alpha=0.3)
    ax1.axhline(-1, color='red', linestyle=':', alpha=0.3)
    ax1.set_title(f'{factor_name} - Level (绝对水位): 当前处于什么位置')
    ax1.set_ylabel('Z-Score')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-3, 3)
    
    # 图2: Change (边际变化)
    ax2 = axes[1]
    ax2.fill_between(factor_change.index, 0, factor_change,
                     where=factor_change >= 0, color='blue', alpha=0.3, label='改善/上升')
    ax2.fill_between(factor_change.index, 0, factor_change,
                     where=factor_change < 0, color='orange', alpha=0.3, label='恶化/下降')
    ax2.plot(factor_change, color='black', linewidth=1.5)
    ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_title(f'{factor_name} - Change (边际变化): 趋势在改善还是恶化')
    ax2.set_ylabel('Z-Score')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-3, 3)
    
    # 图3: 四象限 (Level vs Change)
    ax3 = axes[2]
    
    # 对齐数据
    df_plot = pd.DataFrame({'Level': factor_level, 'Change': factor_change}).dropna()
    
    if len(df_plot) > 0:
        scatter = ax3.scatter(df_plot['Level'], df_plot['Change'],
                              c=range(len(df_plot)), cmap='viridis', alpha=0.5, s=20)
        
        # 最新点
        ax3.scatter(df_plot['Level'].iloc[-1], df_plot['Change'].iloc[-1],
                    color='red', s=200, marker='*', 
                    label=f'最新: {df_plot.index[-1].strftime("%Y-%m")}')
        
        # 四象限
        ax3.axhline(0, color='gray', linestyle='--', alpha=0.7)
        ax3.axvline(0, color='gray', linestyle='--', alpha=0.7)
        
        ax3.text(1.5, 1.5, 'Ⅰ 高位+改善', ha='center', va='center', fontsize=10, color='green')
        ax3.text(-1.5, 1.5, 'Ⅱ 低位+改善', ha='center', va='center', fontsize=10, color='blue')
        ax3.text(-1.5, -1.5, 'Ⅲ 低位+恶化', ha='center', va='center', fontsize=10, color='red')
        ax3.text(1.5, -1.5, 'Ⅳ 高位+恶化', ha='center', va='center', fontsize=10, color='orange')
        
        ax3.set_xlabel('Level (绝对水位)')
        ax3.set_ylabel('Change (边际变化)')
        ax3.set_title(f'{factor_name} - Level vs Change 四象限')
        ax3.legend(loc='upper right')
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(-3, 3)
        ax3.set_ylim(-3, 3)
        
        plt.colorbar(scatter, ax=ax3, label='时间')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{factor_name}_dual_view.png'), dpi=150)
    plt.close()
    
    print(f"  [Plot] {factor_name} 双视图保存完成")


def plot_all_factors_summary(all_factors_df, save_dir):
    """
    绘制所有因子的汇总图
    """
    if all_factors_df is None or len(all_factors_df) < 12:
        return
    
    # 分别绘制 Level 和 Change
    level_cols = [c for c in all_factors_df.columns if '_Level' in c]
    change_cols = [c for c in all_factors_df.columns if '_Change' in c]
    
    # Level 汇总图
    if level_cols:
        n = len(level_cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
        
        colors = {
            'CN_Growth_Level': 'green',
            'CN_Inflation_Level': 'red',
            'US_Growth_Level': 'blue',
            'US_Inflation_Level': 'orange',
            'CN_RiskAppetite_Level': 'purple',
            'CN_Monetary_Level': 'brown'
        }
        
        for i, col in enumerate(level_cols):
            ax = axes[i]
            data = all_factors_df[col].dropna()
            color = colors.get(col, 'black')
            
            ax.fill_between(data.index, 0, data,
                            where=data >= 0, color=color, alpha=0.3)
            ax.fill_between(data.index, 0, data,
                            where=data < 0, color='gray', alpha=0.3)
            ax.plot(data, color=color, linewidth=1.5)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(col.replace('_Level', ' (Level)'))
            ax.set_ylabel('Z-Score')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'All_Factors_Level.png'), dpi=150)
        plt.close()
        print(f"  [Plot] Level 汇总图保存完成")
    
    # Change 汇总图
    if change_cols:
        n = len(change_cols)
        fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n))
        if n == 1:
            axes = [axes]
        
        for i, col in enumerate(change_cols):
            ax = axes[i]
            data = all_factors_df[col].dropna()
            
            ax.fill_between(data.index, 0, data,
                            where=data >= 0, color='blue', alpha=0.3)
            ax.fill_between(data.index, 0, data,
                            where=data < 0, color='orange', alpha=0.3)
            ax.plot(data, color='black', linewidth=1.5)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
            ax.set_title(col.replace('_Change', ' (Change)'))
            ax.set_ylabel('Z-Score')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-3, 3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'All_Factors_Change.png'), dpi=150)
        plt.close()
        print(f"  [Plot] Change 汇总图保存完成")


def plot_factor_distribution(factor_series: pd.Series, factor_col: str, save_dir: str) -> None:
    """绘制单个因子的分布图（直方图）。

    说明：
    - 因子本脚本输出通常为 Z-Score 且已 clip 到 [-3, 3]
    - 这里用直方图 + 关键参考线（0/均值/±1/±2）辅助检查分布形态
    """
    if factor_series is None:
        return
    s = pd.Series(factor_series).dropna()
    if len(s) < 12:
        return

    os.makedirs(save_dir, exist_ok=True)

    mu = float(s.mean())
    sigma = float(s.std(ddof=1)) if len(s) > 1 else 0.0

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.hist(s.values, bins=40, density=True, alpha=0.7, color='steelblue', edgecolor='white')

    # reference lines
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1.0, alpha=0.8, label='0')
    ax.axvline(mu, color='crimson', linestyle='-', linewidth=1.5, alpha=0.9, label=f"mean={mu:+.2f}")
    for x, lab in [(1.0, '+1'), (-1.0, '-1'), (2.0, '+2'), (-2.0, '-2')]:
        ax.axvline(x, color='gray', linestyle=':', linewidth=1.0, alpha=0.5, label=lab)

    ax.set_title(f"{factor_col} 分布 | n={len(s)}, std={sigma:.2f}")
    ax.set_xlabel('Z-Score')
    ax.set_ylabel('Density')
    ax.grid(True, alpha=0.2)

    # 去重 legend（避免 +1/-1/+2/-2 多次重复）
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    uniq_h, uniq_l = [], []
    for h, l in zip(handles, labels):
        if l in seen:
            continue
        seen.add(l)
        uniq_h.append(h)
        uniq_l.append(l)
    ax.legend(uniq_h, uniq_l, loc='upper right', frameon=False)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{factor_col}_dist.png"), dpi=150)
    plt.close()


def plot_all_factors_distributions(all_factors_df: pd.DataFrame, save_dir: str) -> None:
    """批量绘制所有主因子的分布图。"""
    if all_factors_df is None or all_factors_df.empty:
        return
    dist_dir = os.path.join(save_dir, 'distributions')
    os.makedirs(dist_dir, exist_ok=True)
    for col in all_factors_df.columns:
        try:
            plot_factor_distribution(all_factors_df[col], col, dist_dir)
        except Exception:
            continue


# =============================================================================
# 相关性分析
# =============================================================================

def print_correlation_analysis(all_factors_df):
    """打印因子相关性分析"""
    if all_factors_df is None or len(all_factors_df) < 12:
        return
    
    print("\n" + "=" * 70)
    print("因子相关性分析")
    print("=" * 70)
    
    # 分别分析 Level 和 Change
    level_cols = [c for c in all_factors_df.columns if '_Level' in c]
    change_cols = [c for c in all_factors_df.columns if '_Change' in c]
    
    df_valid = all_factors_df.dropna()
    
    if level_cols and len(df_valid) > 0:
        print("\n📊 Level 因子相关性矩阵:")
        print("-" * 70)
        corr_level = df_valid[level_cols].corr()
        # 简化列名显示
        corr_level.index = [c.replace('_Level', '') for c in corr_level.index]
        corr_level.columns = [c.replace('_Level', '') for c in corr_level.columns]
        print(corr_level.round(3).to_string())
    
    if change_cols and len(df_valid) > 0:
        print("\n📊 Change 因子相关性矩阵:")
        print("-" * 70)
        corr_change = df_valid[change_cols].corr()
        corr_change.index = [c.replace('_Change', '') for c in corr_change.index]
        corr_change.columns = [c.replace('_Change', '') for c in corr_change.columns]
        print(corr_change.round(3).to_string())
    
    # 打印最新值
    print("\n" + "=" * 70)
    print("📌 最新因子值")
    print("=" * 70)
    
    latest = all_factors_df.iloc[-1]
    latest_date = all_factors_df.index[-1].strftime("%Y-%m")
    print(f"截至: {latest_date}\n")
    
    # 按因子分组打印
    factor_names = ['CN_Growth', 'CN_Inflation', 'US_Growth', 'US_Inflation', 
                    'CN_RiskAppetite', 'CN_Monetary']
    
    for name in factor_names:
        level_col = f'{name}_Level'
        change_col = f'{name}_Change'
        
        if level_col in latest.index and change_col in latest.index:
            level_val = latest[level_col]
            change_val = latest[change_col]
            
            if name == 'CN_Monetary':
                # 口径：已在构造时翻转为“高值=宽松”
                if level_val > 1:
                    level_status = "🟢宽松(高)"
                elif level_val > 0:
                    level_status = "🟢偏宽松"
                elif level_val > -1:
                    level_status = "🟠偏紧"
                else:
                    level_status = "🔴紧缩(低)"

                if change_val > 0.5:
                    change_status = "⬆️ 更宽松"
                elif change_val > -0.5:
                    change_status = "➡️ 稳定"
                else:
                    change_status = "⬇️ 更紧缩"
            else:
                # 判断 Level 状态
                if level_val > 1:
                    level_status = "🔺高位"
                elif level_val > 0:
                    level_status = "▲偏高"
                elif level_val > -1:
                    level_status = "▼偏低"
                else:
                    level_status = "🔻低位"
                
                # 判断 Change 状态
                if change_val > 0.5:
                    change_status = "⬆️ 改善"
                elif change_val > -0.5:
                    change_status = "➡️ 稳定"
                else:
                    change_status = "⬇️ 恶化"
            
            print(f"  {name}:")
            print(f"    Level:  {level_val:+.3f}  {level_status}")
            print(f"    Change: {change_val:+.3f}  {change_status}")
            print()
    
    # 经济状态判断
    print("=" * 70)
    print("🎯 当前经济状态判断")
    print("=" * 70)
    
    cn_growth_level = latest.get('CN_Growth_Level', 0)
    cn_inflation_level = latest.get('CN_Inflation_Level', 0)
    
    if cn_growth_level > 0 and cn_inflation_level > 0:
        quadrant = "Ⅰ 过热 (Growth↑ Inflation↑)"
        suggestion = "减配债券，增配商品"
    elif cn_growth_level < 0 and cn_inflation_level > 0:
        quadrant = "Ⅱ 滞胀 (Growth↓ Inflation↑)"
        suggestion = "增配黄金/现金，减配股票"
    elif cn_growth_level < 0 and cn_inflation_level < 0:
        quadrant = "Ⅲ 衰退 (Growth↓ Inflation↓)"
        suggestion = "增配长债，减配商品"
    else:
        quadrant = "Ⅳ 复苏 (Growth↑ Inflation↓)"
        suggestion = "增配股票，减配黄金"
    
    print(f"\n  中国宏观象限 (基于 Level): {quadrant}")
    print(f"  配置建议: {suggestion}")
    
    # 边际变化判断
    cn_growth_change = latest.get('CN_Growth_Change', 0)
    cn_inflation_change = latest.get('CN_Inflation_Change', 0)
    
    print(f"\n  边际变化:")
    print(f"    增长: {'改善中' if cn_growth_change > 0 else '恶化中'} ({cn_growth_change:+.2f})")
    print(f"    通胀: {'上升中' if cn_inflation_change > 0 else '下降中'} ({cn_inflation_change:+.2f})")
    
    print("\n" + "=" * 70)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    print("=" * 70)
    print("宏观因子构造 - V2 (Level + Change 双维度)")
    print("=" * 70)
    
    # 加载数据
    df_raw = load_raw_data()
    
    # 构建所有因子 (同时收集分项因子)
    cn_growth_level, cn_growth_change, cn_growth_sub_l, cn_growth_sub_c = build_cn_growth_factor(df_raw)
    cn_inflation_level, cn_inflation_change, cn_inflation_sub_l, cn_inflation_sub_c = build_cn_inflation_factor(df_raw)
    us_inflation_level, us_inflation_change, us_inflation_sub_l, us_inflation_sub_c = build_us_inflation_factor(df_raw)
    us_growth_level, us_growth_change, us_growth_sub_l, us_growth_sub_c = build_us_growth_factor(df_raw)
    cn_risk_level, cn_risk_change, cn_risk_sub_l, cn_risk_sub_c = build_cn_risk_appetite_factor(df_raw)
    cn_monetary_level, cn_monetary_change, cn_monetary_sub_l, cn_monetary_sub_c = build_cn_monetary_factor(df_raw)
    
    # 汇总所有主因子
    all_factors = {}
    
    # 汇总所有分项因子
    all_sub_factors = {}
    
    if cn_growth_level is not None:
        all_factors['CN_Growth_Level'] = cn_growth_level
        all_factors['CN_Growth_Change'] = cn_growth_change
        plot_factor_dual(cn_growth_level, cn_growth_change, 'CN_Growth', PLOT_DIR)
        # 收集分项因子
        for name, sig in cn_growth_sub_l.items():
            all_sub_factors[f'CN_Growth_{name}_Level'] = sig
        for name, sig in cn_growth_sub_c.items():
            all_sub_factors[f'CN_Growth_{name}_Change'] = sig
    
    if cn_inflation_level is not None:
        all_factors['CN_Inflation_Level'] = cn_inflation_level
        all_factors['CN_Inflation_Change'] = cn_inflation_change
        plot_factor_dual(cn_inflation_level, cn_inflation_change, 'CN_Inflation', PLOT_DIR)
        # 收集分项因子
        for name, sig in cn_inflation_sub_l.items():
            all_sub_factors[f'CN_Inflation_{name}_Level'] = sig
        for name, sig in cn_inflation_sub_c.items():
            all_sub_factors[f'CN_Inflation_{name}_Change'] = sig
    
    if us_inflation_level is not None:
        all_factors['US_Inflation_Level'] = us_inflation_level
        all_factors['US_Inflation_Change'] = us_inflation_change
        plot_factor_dual(us_inflation_level, us_inflation_change, 'US_Inflation', PLOT_DIR)
        # 收集分项因子
        for name, sig in us_inflation_sub_l.items():
            all_sub_factors[f'US_Inflation_{name}_Level'] = sig
        for name, sig in us_inflation_sub_c.items():
            all_sub_factors[f'US_Inflation_{name}_Change'] = sig
    
    if us_growth_level is not None:
        all_factors['US_Growth_Level'] = us_growth_level
        all_factors['US_Growth_Change'] = us_growth_change
        plot_factor_dual(us_growth_level, us_growth_change, 'US_Growth', PLOT_DIR)
        # 收集分项因子
        for name, sig in us_growth_sub_l.items():
            all_sub_factors[f'US_Growth_{name}_Level'] = sig
        for name, sig in us_growth_sub_c.items():
            all_sub_factors[f'US_Growth_{name}_Change'] = sig
    
    if cn_risk_level is not None:
        all_factors['CN_RiskAppetite_Level'] = cn_risk_level
        all_factors['CN_RiskAppetite_Change'] = cn_risk_change
        plot_factor_dual(cn_risk_level, cn_risk_change, 'CN_RiskAppetite', PLOT_DIR)
        # 收集分项因子
        for name, sig in cn_risk_sub_l.items():
            all_sub_factors[f'CN_RiskAppetite_{name}_Level'] = sig
        for name, sig in cn_risk_sub_c.items():
            all_sub_factors[f'CN_RiskAppetite_{name}_Change'] = sig
    
    if cn_monetary_level is not None:
        all_factors['CN_Monetary_Level'] = cn_monetary_level
        all_factors['CN_Monetary_Change'] = cn_monetary_change
        plot_factor_dual(cn_monetary_level, cn_monetary_change, 'CN_Monetary', PLOT_DIR)
        # 收集分项因子
        for name, sig in cn_monetary_sub_l.items():
            all_sub_factors[f'CN_Monetary_{name}_Level'] = sig
        for name, sig in cn_monetary_sub_c.items():
            all_sub_factors[f'CN_Monetary_{name}_Change'] = sig
    
    # 导出主因子
    df_all = None
    if all_factors:
        df_all = pd.DataFrame(all_factors)
        df_all.index.name = 'date'

        # 先画图（即使 CSV 写入失败也能产出图表）
        plot_all_factors_summary(df_all, PLOT_DIR)
        plot_all_factors_distributions(df_all, PLOT_DIR)

        # 再导出（可能被 Excel 占用）
        try:
            df_all.to_csv(OUTPUT_ALL_FACTORS_FILE, encoding='utf-8-sig')
            print(f"\n[Export] 全部因子导出: {OUTPUT_ALL_FACTORS_FILE}")
            print(f"  时间范围: {df_all.index.min().date()} ~ {df_all.index.max().date()}")
            print(f"  因子数量: {len(df_all.columns)}")
        except PermissionError as e:
            print(f"\n[Export][Warning] 无法写入 {OUTPUT_ALL_FACTORS_FILE}（可能文件被占用）: {e}")
            print("  已继续生成图表；请关闭占用文件后重试导出。")
        except Exception as e:
            print(f"\n[Export][Warning] 导出失败: {e}")

        # 相关性分析（控制台输出）
        print_correlation_analysis(df_all)
    
    # 导出分项因子
    if all_sub_factors:
        df_sub = pd.DataFrame(all_sub_factors)
        df_sub.index.name = 'date'
        sub_factors_file = os.path.join(DATA_DIR, 'all_macro_sub_factors.csv')
        df_sub.to_csv(sub_factors_file, encoding='utf-8-sig')
        print(f"\n[Export] 分项因子导出: {sub_factors_file}")
        print(f"  分项因子数量: {len(df_sub.columns)}")
    
    print("\n" + "=" * 70)
    print("因子构造完成!")
    print(f"图表目录: {PLOT_DIR}")
    print("=" * 70)
    
    return df_all


if __name__ == "__main__":
    df_factors = main()
