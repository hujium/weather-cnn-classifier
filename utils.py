#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工具函数
========
统一的 matplotlib 中文字体配置，供 train.py / experiment.py / app.py 共用。

核心策略：使用 FontProperties(fname=...) 直接指定字体文件，
不依赖 rcParams（rcParams 在多环境下不可靠）。
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, fontManager

# ── 中文字体路径（按优先级搜索） ──
_FONT_CANDIDATES = [
    '/system/fonts/MiSansVF.ttf',          # 小米设备
    '/system/fonts/MiSansTCVF.ttf',        # 小米设备繁体
    '/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',
    '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf',
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'NotoSansSC-Regular.ttf'),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'MiSansVF.ttf'),
]

_CN_FONT_PATH = None
_CN_FONT_PROP = None


def _find_chinese_font():
    """搜索可用的中文字体文件"""
    global _CN_FONT_PATH, _CN_FONT_PROP
    if _CN_FONT_PATH is not None:
        return _CN_FONT_PATH

    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                prop = FontProperties(fname=path)
                _CN_FONT_PATH = path
                _CN_FONT_PROP = prop
                # 注册到 matplotlib 字体管理器
                fontManager.addfont(path)
                return path
            except Exception:
                continue
    return None


def get_cn_font_prop():
    """获取中文字体 FontProperties（供 ax.text() 等直接使用）"""
    _find_chinese_font()
    return _CN_FONT_PROP


def setup_matplotlib_chinese():
    """
    配置 matplotlib 中文显示。

    双重保险:
      1. 注册字体 + 设置 rcParams（对标题/轴标签生效）
      2. 缓存 FontProperties（对 ax.text() 等直接调用生效）
    """
    font_path = _find_chinese_font()

    if font_path is not None:
        # 注册字体（已在 _find_chinese_font 中完成）
        # 设置 rcParams
        font_name = _CN_FONT_PROP.get_name() if _CN_FONT_PROP else None
        if font_name:
            matplotlib.rcParams['font.sans-serif'] = [font_name, 'DejaVu Sans']
            matplotlib.rcParams['font.family'] = 'sans-serif'
    else:
        matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']

    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['figure.dpi'] = 150

    return font_path is not None


def cn_text(ax, x, y, s, **kwargs):
    """
    在 axes 上绘制中文文本（自动附带中文字体）。

    用法:
        cn_text(ax, 0.5, 0.5, '测试中文', fontsize=14)
    """
    prop = get_cn_font_prop()
    if prop is not None:
        # 合并用户传入的 fontproperties
        user_prop = kwargs.pop('fontproperties', None)
        if user_prop is not None:
            # 用户自定义优先
            kwargs['fontproperties'] = user_prop
        else:
            kwargs['fontproperties'] = prop
    ax.text(x, y, s, **kwargs)


def cn_set_title(ax, title, **kwargs):
    """设置带中文的标题"""
    prop = get_cn_font_prop()
    if prop is not None:
        kwargs.setdefault('fontproperties', prop)
    ax.set_title(title, **kwargs)


def cn_set_xlabel(ax, label, **kwargs):
    """设置带中文的X轴标签"""
    prop = get_cn_font_prop()
    if prop is not None:
        kwargs.setdefault('fontproperties', prop)
    ax.set_xlabel(label, **kwargs)


def cn_set_ylabel(ax, label, **kwargs):
    """设置带中文的Y轴标签"""
    prop = get_cn_font_prop()
    if prop is not None:
        kwargs.setdefault('fontproperties', prop)
    ax.set_ylabel(label, **kwargs)


def cn_set_suptitle(fig, title, **kwargs):
    """设置带中文的总标题"""
    prop = get_cn_font_prop()
    if prop is not None:
        kwargs.setdefault('fontproperties', prop)
    fig.suptitle(title, **kwargs)


# ── 模块加载时自动配置 ──
HAS_CN_FONT = setup_matplotlib_chinese()
