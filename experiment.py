#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气图像分类 — 多实验对比脚本
==============================
自动运行多组实验（不同优化器 / 学习率 / 激活函数），生成对比图表。
内置实时进度回调。

用法:
    python experiment.py              # 运行全部对比实验
    python experiment.py --fast       # 快速模式 (15 epochs)
    python experiment.py --only-adam  # 仅对比优化器
    python experiment.py --json-progress  # JSON 行输出进度

输出:
    logs/experiment_comparison.png    # 实验对比总图
    logs/experiment_summary.json      # 实验汇总表

进度回调 (progress_callback):
    每个 epoch / 每个实验阶段都会调用 callback(info_dict)，info_dict 包含:
      type         : 'exp_start' | 'exp_complete' | 'exp_error' | 'all_complete'
      exp_index    : 当前实验序号 (0-based)
      exp_total    : 实验总数
      exp_name     : 实验名称
      exp_best_acc : 该实验的最佳准确率 (仅 exp_complete)
      exp_time_s   : 该实验耗时 (仅 exp_complete)
      progress_pct : 总进度百分比 (0-100)
      best_acc     : 所有实验中的最佳准确率 (仅 all_complete)
      results      : 全部实验结果列表 (仅 all_complete)
      message      : 人类可读的消息
      ── 透传的 epoch 级信息 ──
      epoch, total_epochs, train_loss, train_acc, test_loss, test_acc, lr, ...
"""

import os
import sys
import json
import time
import argparse
import traceback
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from train import run_training, PRESETS, LOG_DIR

# ── 中文字体配置（继承 train.py 已加载的配置） ──
import utils as _u  # noqa: E402
from utils import (cn_set_title, cn_set_xlabel, cn_set_ylabel, cn_set_suptitle, cn_text, get_cn_font_prop)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ===================================================================
#  进度回调工具
# ===================================================================
def _emit(cb, info):
    """统一发射 + 终端打印"""
    if cb is not None:
        try:
            cb(info)
        except Exception:
            pass
    t = info.get('type', '')
    if t == 'exp_start':
        idx = info['exp_index'] + 1
        total = info['exp_total']
        pct = info['progress_pct']
        print(f"\n{'#' * 60}")
        print(f"  [{pct:5.1f}%] ({idx}/{total}) 开始实验: {info['exp_name']}")
        print(f"{'#' * 60}")
    elif t == 'exp_complete':
        idx = info['exp_index'] + 1
        total = info['exp_total']
        pct = info['progress_pct']
        acc = info.get('exp_best_acc', '?')
        t_s = info.get('exp_time_s', '?')
        print(f"  [{pct:5.1f}%] ({idx}/{total}) {info['exp_name']} 完成 — "
              f"最佳准确率 {acc}%  耗时 {t_s}s")
    elif t == 'exp_error':
        pct = info['progress_pct']
        print(f"  [{pct:5.1f}%] ⚠️ {info['exp_name']} 失败: {info['message']}")
    elif t == 'all_complete':
        print(f"\n{'=' * 60}")
        print(f"  [100.0%] {info['message']}")
        print(f"{'=' * 60}")
    elif t == 'epoch':
        # 透传 epoch 级信息，追加实验上下文
        exp_idx = info.get('exp_index', '?')
        exp_total = info.get('exp_total', '?')
        exp_name = info.get('exp_name', '')
        marker = ' ★' if info.get('is_best') else ''
        pct = info.get('progress_pct', 0)
        print(
            f"  [{pct:5.1f}%] [{exp_idx+1}/{exp_total}] {exp_name} | "
            f"Epoch [{info['epoch']:2d}/{info['total_epochs']}] "
            f"train_loss={info['train_loss']:.4f}  test_loss={info['test_loss']:.4f}  "
            f"train_acc={info['train_acc']:.1f}%  test_acc={info['test_acc']:.1f}%  "
            f"lr={info['lr']:.6f}  {info.get('elapsed', 0):.1f}s{marker}"
        )


# ===================================================================
#  实验运行
# ===================================================================
def run_experiment_group(experiment_names, override_epochs=None,
                         progress_callback=None):
    """
    依次运行一组实验，返回结果列表。

    参数:
        experiment_names: list[str] — PRESETS 中的配置名
        override_epochs: int | None — 覆盖所有实验的轮数
        progress_callback: callable(info_dict) | None — 实时进度回调

    返回:
        results: list[dict] — 每个实验的完整结果
    """
    results = []
    total_exps = len(experiment_names)

    for idx, name in enumerate(experiment_names):
        cfg = PRESETS[name].copy()
        cfg['name'] = name
        if override_epochs is not None:
            cfg['epochs'] = override_epochs

        # 计算实验级总进度
        # 0%-90% 分配给各个实验（每个实验内部再细分 epoch）
        exp_pct_base = 90 * idx / total_exps
        exp_pct_end = 90 * (idx + 1) / total_exps

        # 通知: 实验开始
        _emit(progress_callback, {
            'type': 'exp_start',
            'exp_index': idx,
            'exp_total': total_exps,
            'exp_name': name,
            'progress_pct': round(exp_pct_base, 1),
            'message': f'({idx+1}/{total_exps}) 开始实验: {name}',
        })

        # 创建透传回调：将 epoch 级进度映射到总进度
        def _forward_epoch(epoch_info):
            """将 train.py 的 epoch 回调附加实验上下文并映射到总进度"""
            epoch_pct = epoch_info.get('progress_pct', 50)
            # train.py 内部进度 20%-90% → 映射到当前实验的进度区间
            mapped = exp_pct_base + (exp_pct_end - exp_pct_base) * (epoch_pct - 20) / 70
            epoch_info['progress_pct'] = round(max(exp_pct_base, min(mapped, exp_pct_end)), 1)
            epoch_info['exp_index'] = idx
            epoch_info['exp_total'] = total_exps
            _emit(progress_callback, epoch_info)

        try:
            exp_start = time.time()
            record = run_training(cfg, save_model=False,
                                  progress_callback=_forward_epoch)
            exp_time = round(time.time() - exp_start, 1)
            best_acc = record['results']['best_test_acc']
            results.append(record)

            # 通知: 实验完成
            _emit(progress_callback, {
                'type': 'exp_complete',
                'exp_index': idx,
                'exp_total': total_exps,
                'exp_name': name,
                'exp_best_acc': best_acc,
                'exp_time_s': exp_time,
                'progress_pct': round(exp_pct_end, 1),
                'message': f'({idx+1}/{total_exps}) {name} 完成 — 最佳 {best_acc}% {exp_time}s',
            })

        except Exception as e:
            _emit(progress_callback, {
                'type': 'exp_error',
                'exp_index': idx,
                'exp_total': total_exps,
                'exp_name': name,
                'progress_pct': round(exp_pct_end, 1),
                'message': str(e),
            })

    return results


# ===================================================================
#  可视化
# ===================================================================
def plot_comparison(results, save_dir):
    """绘制实验对比图"""
    if not results:
        print('无实验结果，跳过对比图')
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fp = get_cn_font_prop()
    cn_set_suptitle(fig, '实验对比分析', fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, max(len(results), 1)))

    names = [r['config']['name'].replace('experiment_', '') for r in results]
    best_accs = [r['results']['best_test_acc'] for r in results]
    final_accs = [r['results']['final_test_acc'] for r in results]

    x = np.arange(len(names))
    w = 0.35
    bars1 = axes[0].bar(x - w/2, best_accs, w, label='最佳准确率', color='#42A5F5')
    bars2 = axes[0].bar(x + w/2, final_accs, w, label='最终准确率', color='#EF5350')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha='right')
    cn_set_ylabel(axes[0], '准确率 (%)')
    cn_set_title(axes[0], '各实验最终准确率对比')
    axes[0].legend(prop=fp)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars1, best_accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

    for i, r in enumerate(results):
        label = r['config']['name'].replace('experiment_', '')
        axes[1].plot(r['history']['test_loss'], label=label, color=colors[i], linewidth=2)
    axes[1].set_xlabel('Epoch')
    cn_set_ylabel(axes[1], '测试损失')
    cn_set_title(axes[1], '测试损失曲线对比')
    axes[1].legend(fontsize=8, prop=fp)
    axes[1].grid(True, alpha=0.3)

    for i, r in enumerate(results):
        label = r['config']['name'].replace('experiment_', '')
        axes[2].plot(r['history']['test_acc'], label=label, color=colors[i], linewidth=2)
    axes[2].set_xlabel('Epoch')
    cn_set_ylabel(axes[2], '测试准确率 (%)')
    cn_set_title(axes[2], '测试准确率曲线对比')
    axes[2].legend(fontsize=8, prop=fp)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'experiment_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'\n📊 对比图已保存: {save_path}')
    return save_path


def generate_summary_table(results, save_dir):
    """生成实验汇总 JSON"""
    summary = []
    for r in results:
        summary.append({
            'name': r['config']['name'],
            'optimizer': r['config']['optimizer'],
            'lr': r['config']['learning_rate'],
            'activation': r['config']['activation'],
            'scheduler': r['config']['scheduler'],
            'epochs': r['config']['epochs'],
            'best_test_acc': r['results']['best_test_acc'],
            'final_test_acc': r['results']['final_test_acc'],
            'final_test_loss': r['results']['final_test_loss'],
            'total_time_s': r['results']['total_time_s'],
        })

    summary.sort(key=lambda x: x['best_test_acc'], reverse=True)

    save_path = os.path.join(save_dir, 'experiment_summary.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f'\n{"=" * 90}')
    print(f'{"实验名":<25} {"优化器":<8} {"激活函数":<12} {"调度器":<10} {"最佳准确率":>10} {"最终准确率":>10}')
    print(f'{"=" * 90}')
    for s in summary:
        print(f'{s["name"]:<25} {s["optimizer"]:<8} {s["activation"]:<12} '
              f'{s["scheduler"]:<10} {s["best_test_acc"]:>9.2f}% {s["final_test_acc"]:>9.2f}%')
    print(f'{"=" * 90}')

    return summary


# ===================================================================
#  主入口
# ===================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='天气图像分类 — 多实验对比')
    parser.add_argument('--fast', action='store_true', help='快速模式 (15 epochs)')
    parser.add_argument('--only-optimizer', action='store_true', help='仅对比优化器')
    parser.add_argument('--only-activation', action='store_true', help='仅对比激活函数')
    parser.add_argument('--json-progress', action='store_true',
                        help='输出 JSON 行格式的实时进度 (供程序消费)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(LOG_DIR, exist_ok=True)

    epochs_override = 15 if args.fast else None

    if args.only_optimizer:
        experiments = ['experiment_adam', 'experiment_adamw', 'experiment_sgd']
        print('🔬 对比实验: 优化器 (Adam vs AdamW vs SGD)')
    elif args.only_activation:
        experiments = ['experiment_relu', 'experiment_leaky_relu', 'experiment_gelu']
        print('🔬 对比实验: 激活函数 (ReLU vs LeakyReLU vs GELU)')
    else:
        experiments = [
            'experiment_adam', 'experiment_adamw', 'experiment_sgd',
            'experiment_relu', 'experiment_leaky_relu', 'experiment_gelu',
        ]
        print('🔬 全部对比实验: 优化器 + 激活函数')

    # JSON 进度模式
    cb = None
    if args.json_progress:
        def _json_cb(info):
            print(json.dumps(info, ensure_ascii=False, default=str), flush=True)
        cb = _json_cb

    start = time.time()
    results = run_experiment_group(experiments, override_epochs=epochs_override,
                                   progress_callback=cb)
    elapsed = time.time() - start

    if results:
        plot_comparison(results, LOG_DIR)
        summary = generate_summary_table(results, LOG_DIR)
        best = max(summary, key=lambda x: x['best_test_acc'])

        # 最终完成通知
        _emit(cb, {
            'type': 'all_complete',
            'progress_pct': 100.0,
            'best_name': best['name'],
            'best_acc': best['best_test_acc'],
            'total_exps': len(results),
            'total_time_s': round(elapsed, 1),
            'results': summary,
            'message': (f'全部实验完成! 总耗时 {elapsed:.1f}s ({elapsed/60:.1f} min)  '
                        f'最优: {best["name"]} ({best["best_test_acc"]:.2f}%)'),
        })
    else:
        _emit(cb, {
            'type': 'all_complete',
            'progress_pct': 100.0,
            'total_exps': 0,
            'results': [],
            'message': '没有成功的实验结果',
        })
