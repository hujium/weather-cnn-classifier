#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气图像分类 — PyTorch CNN 训练脚本
====================================
支持单次训练和实验对比模式，内置实时进度回调。

用法:
    python train.py                        # 使用默认配置训练
    python train.py --config best          # 使用预设最优配置
    python train.py --epochs 50 --lr 0.001 # 自定义参数
    python train.py --json-progress        # JSON 行输出进度 (供程序消费)

输出:
    model/weather_cnn.pth                  # 最佳模型权重
    logs/training_curves.png               # 训练曲线图
    logs/confusion_matrix.png              # 混淆矩阵图
    logs/training_history.json             # 完整训练记录 (供 app.py 使用)
    logs/classification_report.json        # 分类报告

进度回调 (progress_callback):
    每个 epoch 结束时调用 callback(info_dict)，info_dict 包含:
      type         : 'phase' | 'epoch' | 'complete' | 'error'
      phase        : 当前阶段 ('加载数据' | '构建模型' | '配置训练' | '训练中' | '最终评估' | '完成')
      epoch        : 当前 epoch (1-based)
      total_epochs : 总 epoch 数
      train_loss   : 训练损失
      train_acc    : 训练准确率 (%)
      test_loss    : 测试损失
      test_acc     : 测试准确率 (%)
      lr           : 当前学习率
      best_acc     : 当前最佳准确率 (%)
      is_best      : 是否为最佳 epoch
      elapsed      : 本 epoch 耗时 (秒)
      total_elapsed: 总耗时 (秒)
      progress_pct : 总进度百分比 (0-100)
      exp_name     : 实验名称
      message      : 人类可读的消息
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import classification_report, confusion_matrix

# ── 从 model.py 导入统一模型 ──
from model import WeatherCNN, count_parameters

# ── 中文字体配置 ──
import utils as _u  # noqa: E402 — 自动配置 matplotlib 中文
from utils import (cn_set_title, cn_set_xlabel, cn_set_ylabel, cn_set_suptitle, cn_text, get_cn_font_prop)
from utils import (cn_set_title, cn_set_xlabel, cn_set_ylabel, cn_set_suptitle, cn_text, get_cn_font_prop)

# ===================================================================
#  全局常量
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'model')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

CLASS_NAMES_EN = ['cloudy', 'foggy', 'overcast', 'rainy', 'snowy', 'sunny']
CLASS_NAMES_CN = ['多云', '雾天', '阴天', '雨天', '雪天', '晴天']
NUM_CLASSES = 6
IMG_SIZE = 128

# ===================================================================
#  预设实验配置
# ===================================================================
PRESETS = {
    'default': dict(
        optimizer='adam', lr=0.001, weight_decay=1e-4,
        scheduler='plateau', activation='relu',
        epochs=30, batch_size=32,
    ),
    'best': dict(
        optimizer='adamw', lr=0.0008, weight_decay=1e-3,
        scheduler='cosine', activation='leaky_relu',
        epochs=40, batch_size=32,
    ),
    'experiment_adam': dict(
        optimizer='adam', lr=0.001, weight_decay=1e-4,
        scheduler='plateau', activation='relu',
        epochs=30, batch_size=32,
    ),
    'experiment_adamw': dict(
        optimizer='adamw', lr=0.001, weight_decay=1e-4,
        scheduler='plateau', activation='relu',
        epochs=30, batch_size=32,
    ),
    'experiment_sgd': dict(
        optimizer='sgd', lr=0.01, weight_decay=1e-4,
        scheduler='plateau', activation='relu',
        epochs=30, batch_size=32,
    ),
    'experiment_gelu': dict(
        optimizer='adamw', lr=0.0008, weight_decay=1e-3,
        scheduler='cosine', activation='gelu',
        epochs=30, batch_size=32,
    ),
    'experiment_relu': dict(
        optimizer='adamw', lr=0.0008, weight_decay=1e-3,
        scheduler='cosine', activation='relu',
        epochs=30, batch_size=32,
    ),
    'experiment_leaky_relu': dict(
        optimizer='adamw', lr=0.0008, weight_decay=1e-3,
        scheduler='cosine', activation='leaky_relu',
        epochs=30, batch_size=32,
    ),
}


# ===================================================================
#  进度回调工具
# ===================================================================
def _emit(callback, info):
    """
    统一的进度发射函数。
    1. 调用用户 callback（如果提供）
    2. 打印到 stdout
    """
    if callback is not None:
        try:
            callback(info)
        except Exception:
            pass  # 回调不应阻塞训练

    # 终端输出
    t = info.get('type', '')
    if t == 'phase':
        print(f"  [{info['progress_pct']:5.1f}%] {info['message']}")
    elif t == 'epoch':
        marker = ' ★' if info.get('is_best') else ''
        pct = info['progress_pct']
        print(
            f"  [{pct:5.1f}%] Epoch [{info['epoch']:2d}/{info['total_epochs']}] "
            f"train_loss={info['train_loss']:.4f}  test_loss={info['test_loss']:.4f}  "
            f"train_acc={info['train_acc']:.1f}%  test_acc={info['test_acc']:.1f}%  "
            f"lr={info['lr']:.6f}  {info['elapsed']:.1f}s{marker}"
        )
    elif t == 'complete':
        print(f"  [100.0%] {info['message']}")
    elif t == 'error':
        print(f"  [ERROR] {info['message']}")


def _json_emit(callback, info):
    """JSON 行输出模式：只输出 JSON，不打印普通文本"""
    if callback is not None:
        try:
            callback(info)
        except Exception:
            pass
    # 输出 JSON 行到 stdout
    print(json.dumps(info, ensure_ascii=False, default=str), flush=True)


# ===================================================================
#  数据加载
# ===================================================================
def get_transforms():
    """返回训练集和测试集的变换"""
    train_tf = transforms.Compose([
        transforms.RandomCrop(IMG_SIZE, padding=8),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def get_dataloaders(batch_size, num_workers=2):
    """加载训练集和测试集 DataLoader"""
    train_tf, test_tf = get_transforms()
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_tf)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_tf)

    use_pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=use_pin)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=use_pin)

    return train_loader, test_loader, train_ds.class_to_idx


# ===================================================================
#  训练 / 评估
# ===================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / total, 100.0 * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds, all_labels = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return (running_loss / total, 100.0 * correct / total,
            np.array(all_preds), np.array(all_labels))


# ===================================================================
#  可视化
# ===================================================================
def plot_training_curves(history, save_dir, exp_name=''):
    """绘制4合1训练曲线图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    title = f'天气图像分类 — CNN训练曲线'
    if exp_name:
        title += f' ({exp_name})'
    cn_set_suptitle(fig, title, fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)
    fp = get_cn_font_prop()

    axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='训练损失', markersize=3)
    axes[0, 0].plot(epochs, history['test_loss'], 'r-s', label='测试损失', markersize=3)
    cn_set_title(axes[0, 0], '损失曲线 (Loss)')
    cn_set_xlabel(axes[0, 0], 'Epoch')
    cn_set_ylabel(axes[0, 0], 'Loss')
    axes[0, 0].legend(prop=fp)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'g-o', label='训练准确率', markersize=3)
    axes[0, 1].plot(epochs, history['test_acc'], 'm-s', label='测试准确率', markersize=3)
    cn_set_title(axes[0, 1], '准确率曲线 (Accuracy)')
    cn_set_xlabel(axes[0, 1], 'Epoch')
    cn_set_ylabel(axes[0, 1], 'Accuracy (%)')
    axes[0, 1].legend(prop=fp)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['lr'], 'purple', linewidth=2)
    cn_set_title(axes[1, 0], '学习率调度 (Learning Rate)')
    cn_set_xlabel(axes[1, 0], 'Epoch')
    cn_set_ylabel(axes[1, 0], 'LR')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    x = np.array(list(epochs))
    w = 0.35
    axes[1, 1].bar(x - w/2, history['train_acc'], w, label='训练', color='#42A5F5', alpha=0.8)
    axes[1, 1].bar(x + w/2, history['test_acc'], w, label='测试', color='#EF5350', alpha=0.8)
    cn_set_title(axes[1, 1], '训练/测试准确率对比')
    cn_set_xlabel(axes[1, 1], 'Epoch')
    cn_set_ylabel(axes[1, 1], 'Accuracy (%)')
    axes[1, 1].legend(prop=fp)
    axes[1, 1].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    fname = f'training_curves{"_" + exp_name if exp_name else ""}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


def plot_confusion_matrix(all_labels, all_preds, save_dir, exp_name=''):
    """绘制混淆矩阵 (数量 + 百分比)"""
    cm = confusion_matrix(all_labels, all_preds)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    title = '混淆矩阵 (Confusion Matrix)'
    if exp_name:
        title += f' — {exp_name}'
    cn_set_suptitle(fig, title, fontsize=16, fontweight='bold')

    # 混淆矩阵的刻度标签也需要中文字体
    fp = get_cn_font_prop()

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES_CN, yticklabels=CLASS_NAMES_CN, ax=axes[0])
    cn_set_title(axes[0], '数量')
    cn_set_xlabel(axes[0], '预测类别')
    cn_set_ylabel(axes[0], '真实类别')
    if fp:
        for label in axes[0].get_xticklabels() + axes[0].get_yticklabels():
            label.set_fontproperties(fp)

    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='RdYlBu_r',
                xticklabels=CLASS_NAMES_CN, yticklabels=CLASS_NAMES_CN, ax=axes[1])
    cn_set_title(axes[1], '百分比')
    cn_set_xlabel(axes[1], '预测类别')
    cn_set_ylabel(axes[1], '真实类别')
    if fp:
        for label in axes[1].get_xticklabels() + axes[1].get_yticklabels():
            label.set_fontproperties(fp)

    plt.tight_layout()
    fname = f'confusion_matrix{"_" + exp_name if exp_name else ""}.png'
    save_path = os.path.join(save_dir, fname)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return save_path


# ===================================================================
#  构建优化器 / 调度器
# ===================================================================
def build_optimizer(model, name, lr, weight_decay):
    name = name.lower()
    if name == 'adam':
        return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif name == 'sgd':
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"未知优化器: {name}")


def build_scheduler(optimizer, name, **kwargs):
    name = name.lower()
    if name == 'plateau':
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6, verbose=True)
    elif name == 'cosine':
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=kwargs.get('epochs', 30), eta_min=1e-6)
    elif name == 'step':
        return optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        raise ValueError(f"未知调度器: {name}")


# ===================================================================
#  主训练流程
# ===================================================================
def run_training(cfg, save_model=True, progress_callback=None):
    """
    执行一次完整训练，实时回调进度。

    参数:
        cfg: dict — 包含 optimizer, lr, weight_decay, scheduler,
                     activation, epochs, batch_size 等
        save_model: bool — 是否保存模型权重
        progress_callback: callable(info_dict) | None — 每步回调

    返回:
        result: dict — 包含 best_acc, history, report, cm 等
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_name = cfg.get('name', 'run')
    total_epochs = cfg['epochs']
    start_time = time.time()

    def _phase(pct, msg):
        _emit(progress_callback, {
            'type': 'phase', 'phase': msg, 'progress_pct': pct,
            'exp_name': exp_name, 'message': f'{exp_name}: {msg}',
        })

    def _epoch_info(epoch, **kw):
        pct = 20 + 70 * epoch / total_epochs  # 20%-90% 为训练阶段
        base = {
            'type': 'epoch',
            'exp_name': exp_name,
            'epoch': epoch,
            'total_epochs': total_epochs,
            'progress_pct': round(pct, 1),
            'total_elapsed': round(time.time() - start_time, 1),
        }
        base.update(kw)
        _emit(progress_callback, base)

    try:
        # ── 阶段1: 数据 ──
        _phase(5, '加载数据...')
        train_loader, test_loader, class_to_idx = get_dataloaders(cfg['batch_size'])

        # ── 阶段2: 模型 ──
        _phase(10, '构建模型...')
        model = WeatherCNN(num_classes=NUM_CLASSES, activation=cfg['activation']).to(device)
        total_p, train_p = count_parameters(model)

        # ── 阶段3: 配置 ──
        _phase(15, '配置训练...')
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = build_optimizer(model, cfg['optimizer'], cfg['lr'], cfg['weight_decay'])
        scheduler = build_scheduler(optimizer, cfg['scheduler'], epochs=total_epochs)

        # ── 阶段4: 训练 ──
        _phase(20, f'开始训练 ({total_epochs} epochs)...')
        history = {'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []}
        best_acc = 0.0

        for epoch in range(1, total_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)

            current_lr = optimizer.param_groups[0]['lr']
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(test_loss)
            else:
                scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['test_loss'].append(test_loss)
            history['test_acc'].append(test_acc)
            history['lr'].append(current_lr)

            elapsed = time.time() - t0
            is_best = test_acc > best_acc
            if is_best:
                best_acc = test_acc
                if save_model:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'config': cfg,
                        'test_acc': test_acc,
                        'test_loss': test_loss,
                        'class_names_en': CLASS_NAMES_EN,
                        'class_names_cn': CLASS_NAMES_CN,
                        'img_size': IMG_SIZE,
                        'num_classes': NUM_CLASSES,
                    }, os.path.join(MODEL_DIR, 'weather_cnn.pth'))

            _epoch_info(
                epoch,
                train_loss=round(train_loss, 4),
                train_acc=round(train_acc, 2),
                test_loss=round(test_loss, 4),
                test_acc=round(test_acc, 2),
                lr=current_lr,
                best_acc=round(best_acc, 2),
                is_best=is_best,
                elapsed=round(elapsed, 1),
            )

        # ── 阶段5: 最终评估 ──
        _phase(92, '最终评估...')
        best_ckpt = torch.load(os.path.join(MODEL_DIR, 'weather_cnn.pth'), map_location=device, weights_only=False)
        model.load_state_dict(best_ckpt['model_state_dict'])
        _, _, final_preds, final_labels = evaluate(model, test_loader, criterion, device)

        report_dict = classification_report(
            final_labels, final_preds, target_names=CLASS_NAMES_EN, output_dict=True)

        # ── 保存可视化 ──
        _phase(95, '生成可视化图表...')
        curves_path = plot_training_curves(history, LOG_DIR, exp_name)
        cm_path = plot_confusion_matrix(final_labels, final_preds, LOG_DIR, exp_name)

        # ── 保存记录 ──
        _phase(98, '保存训练记录...')
        total_time = time.time() - start_time
        record = {
            'config': {
                'name': exp_name,
                'img_size': IMG_SIZE,
                'batch_size': cfg['batch_size'],
                'num_classes': NUM_CLASSES,
                'epochs': total_epochs,
                'learning_rate': cfg['lr'],
                'weight_decay': cfg['weight_decay'],
                'optimizer': cfg['optimizer'],
                'scheduler': cfg['scheduler'],
                'activation': cfg['activation'],
                'label_smoothing': 0.1,
                'device': str(device),
            },
            'results': {
                'best_test_acc': float(best_acc),
                'total_time_s': round(total_time, 1),
                'final_train_loss': round(history['train_loss'][-1], 4),
                'final_train_acc': round(history['train_acc'][-1], 2),
                'final_test_loss': round(history['test_loss'][-1], 4),
                'final_test_acc': round(history['test_acc'][-1], 2),
            },
            'history': history,
            'classification_report': report_dict,
        }

        with open(os.path.join(LOG_DIR, 'training_history.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(LOG_DIR, f'training_history_{exp_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        with open(os.path.join(LOG_DIR, f'classification_report_{exp_name}.json'), 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)

        # ── 完成 ──
        _emit(progress_callback, {
            'type': 'complete',
            'exp_name': exp_name,
            'progress_pct': 100.0,
            'best_acc': round(best_acc, 2),
            'total_time_s': round(total_time, 1),
            'message': (f'{exp_name}: 训练完成! '
                        f'耗时 {total_time:.1f}s  最佳准确率: {best_acc:.2f}%'),
        })

        return record

    except Exception as e:
        _emit(progress_callback, {
            'type': 'error',
            'exp_name': exp_name,
            'progress_pct': 0,
            'message': f'{exp_name}: 训练失败 — {e}',
            'traceback': traceback.format_exc(),
        })
        raise


# ===================================================================
#  命令行入口
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='天气图像分类 CNN 训练')
    parser.add_argument('--config', type=str, default='best',
                        choices=list(PRESETS.keys()),
                        help='预设配置名 (default/best/experiment_*)')
    parser.add_argument('--epochs', type=int, default=None, help='覆盖训练轮数')
    parser.add_argument('--lr', type=float, default=None, help='覆盖学习率')
    parser.add_argument('--batch-size', type=int, default=None, help='覆盖批大小')
    parser.add_argument('--optimizer', type=str, default=None, help='覆盖优化器')
    parser.add_argument('--activation', type=str, default=None, help='覆盖激活函数')
    parser.add_argument('--scheduler', type=str, default=None, help='覆盖学习率调度器')
    parser.add_argument('--json-progress', action='store_true',
                        help='输出 JSON 行格式的实时进度 (供程序消费)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg = PRESETS[args.config].copy()
    cfg['name'] = args.config

    if args.epochs is not None:
        cfg['epochs'] = args.epochs
    if args.lr is not None:
        cfg['lr'] = args.lr
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.optimizer is not None:
        cfg['optimizer'] = args.optimizer
    if args.activation is not None:
        cfg['activation'] = args.activation
    if args.scheduler is not None:
        cfg['scheduler'] = args.scheduler

    # JSON 进度模式：用 JSON 行替代普通文本
    if args.json_progress:
        def _json_callback(info):
            print(json.dumps(info, ensure_ascii=False, default=str), flush=True)
        run_training(cfg, progress_callback=_json_callback)
    else:
        run_training(cfg)
