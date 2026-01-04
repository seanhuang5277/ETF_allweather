# -*- coding: utf-8 -*-
"""logging_config.py

统一日志配置：在框架内集中配置 logging，避免各策略散落的 print。
提供：
- setup_logging(level='INFO', log_to_file=False, filename=None): 配置根日志器及处理器
- get_logger(name): 获取带命名空间的日志器（建议模块/策略名）
"""
from __future__ import annotations
import logging
import os
from typing import Optional

_DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
_DEFAULT_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str | int = "INFO", log_to_file: bool = False, filename: Optional[str] = None) -> None:
    """配置全局日志设置。

    重复调用的行为：仅调整级别，并在需要时补加文件处理器（不会重复添加控制台处理器）。

    Args:
        level: 日志级别（字符串或 logging 常量），如 'DEBUG'/'INFO'。
        log_to_file: 是否写入文件。
        filename: 写文件路径（当 log_to_file=True 时生效）。
    """
    root = logging.getLogger()
    initialized = getattr(root, "_framework_logging_initialized", False)

    # 解析级别
    resolved_level = level if isinstance(level, int) else getattr(logging, str(level).upper(), logging.INFO)

    if initialized :
        # 已初始化场景：只调级别 + 根据需要补加文件处理器
        root.setLevel(resolved_level)
        if log_to_file and filename:
            # 若不存在文件处理器则补加
            if not any(isinstance(h, logging.FileHandler) for h in root.handlers):
                try:
                    dirn = os.path.dirname(filename)
                    if dirn:
                        os.makedirs(dirn, exist_ok=True)
                    fh = logging.FileHandler(filename, mode='w', encoding="utf-8")
                    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)
                    fh.setFormatter(formatter)
                    root.addHandler(fh)
                    root.info(f"追加文件日志处理器: {filename}")
                except Exception as e:
                    root.error(f"追加文件日志处理器失败: {e}")
        return

    # 首次初始化

    root.setLevel(resolved_level)
    formatter = logging.Formatter(fmt=_DEFAULT_FORMAT, datefmt=_DEFAULT_DATEFMT)

    # 控制台处理器
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    root.addHandler(sh)

    # 可选文件输出
    if log_to_file and filename:
        try:
            dirn = os.path.dirname(filename)
            if dirn:
                os.makedirs(dirn, exist_ok=True)
            fh = logging.FileHandler(filename, mode='w', encoding="utf-8")
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except Exception as e:
            root.error(f"文件日志处理器创建失败: {e}")

    root._framework_logging_initialized = True  # type: ignore[attr-defined]


def get_logger(name: str) -> logging.Logger:
    """按名称获取日志器（建议使用模块路径或策略名）。"""
    return logging.getLogger(name)
