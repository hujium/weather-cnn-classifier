#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeatherCNN 模型定义
====================
统一的卷积神经网络模型，供 train.py 和 app.py 共同使用。

网络架构：
  4个卷积块 (32→64→128→256) + 全连接分类器
  支持可配置的激活函数 (ReLU / LeakyReLU / GELU)

输入: 128×128 RGB 图像
输出: 6类天气
"""

import torch
import torch.nn as nn


class WeatherCNN(nn.Module):
    """
    天气图像分类卷积神经网络

    架构:
      Block1: Conv(3→32) + BN + Act ×2 → MaxPool → Dropout2d
      Block2: Conv(32→64) + BN + Act ×2 → MaxPool → Dropout2d
      Block3: Conv(64→128) + BN + Act ×2 → MaxPool → Dropout2d
      Block4: Conv(128→256) + BN + Act ×2 → AdaptiveAvgPool → Dropout2d
      Classifier: Flatten → FC(4096→512) → FC(512→128) → FC(128→num_classes)

    参数:
        num_classes: 输出类别数 (默认6)
        activation: 激活函数类型 ('relu', 'leaky_relu', 'gelu')
    """

    ACTIVATION_MAP = {
        'relu': lambda: nn.ReLU(inplace=True),
        'leaky_relu': lambda: nn.LeakyReLU(0.1, inplace=True),
        'gelu': lambda: nn.GELU(),
    }

    def __init__(self, num_classes=6, activation='relu'):
        super(WeatherCNN, self).__init__()

        act_fn = self.ACTIVATION_MAP[activation]()

        def _make_block(in_ch, out_ch, drop_rate):
            """构建一个卷积块：2×(Conv+BN+Act) + MaxPool + Dropout2d"""
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                act_fn,
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                act_fn,
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(drop_rate),
            )

        # 特征提取: 4个卷积块
        self.features = nn.Sequential(
            _make_block(3, 32, 0.1),       # 128→64
            _make_block(32, 64, 0.15),     # 64→32
            _make_block(64, 128, 0.2),     # 32→16
            _make_block(128, 256, 0.25),   # 16→8, then AdaptiveAvgPool 8→4
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            act_fn,
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            act_fn,
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Kaiming 初始化"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x


def build_model(num_classes=6, activation='relu', device=None):
    """便捷函数：构建并移动模型到指定设备"""
    model = WeatherCNN(num_classes=num_classes, activation=activation)
    if device is not None:
        model = model.to(device)
    return model


def count_parameters(model):
    """统计模型参数量"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for act in ['relu', 'leaky_relu', 'gelu']:
        m = build_model(6, act, device)
        t, tr = count_parameters(m)
        print(f"activation={act:12s}  total={t:>10,}  trainable={tr:>10,}")
