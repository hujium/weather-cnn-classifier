#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
天气图像分类系统 — Streamlit Web 应用
======================================
基于 PyTorch CNN 的天气场景智能识别系统。

功能:
  Tab 1 — 图片识别: 上传图片，AI 实时识别天气类型
  Tab 2 — 训练过程: 展示训练曲线、混淆矩阵、实验对比
  Tab 3 — 模型配置: 网络结构、超参数、实验配置详情
  Tab 4 — 类别百科: 6类天气的视觉特征说明

启动:
    streamlit run app.py --server.port 8501
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import glob
import json

from model import WeatherCNN, count_parameters
import utils as _u
from utils import cn_set_title, cn_set_xlabel, cn_set_ylabel, get_cn_font_prop

# ===================================================================
#  全局配置
# ===================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'weather_cnn.pth')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
IMG_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CLASS_NAMES = {
    0: ('cloudy', '多云'),
    1: ('foggy', '雾天'),
    2: ('overcast', '阴天'),
    3: ('rainy', '雨天'),
    4: ('snowy', '雪天'),
    5: ('sunny', '晴天'),
}

WEATHER_INFO = {
    'cloudy': {
        'icon': '多云', 'cn': '多云',
        'features': [
            '天空中分布着大量云朵，云量占天空的50%~85%',
            '光线较柔和，阴影不明显',
            '色温偏冷灰色调',
            '可能伴有局部蓝天露出',
            '云层形态多样，有层积云、高积云等',
        ],
    },
    'foggy': {
        'icon': '雾天', 'cn': '雾天',
        'features': [
            '能见度显著降低（通常<1km）',
            '画面整体偏白/灰，对比度极低',
            '远处物体模糊或不可见',
            '色调偏冷灰，饱和度低',
            '湿度高，空气中有明显水汽感',
        ],
    },
    'overcast': {
        'icon': '阴天', 'cn': '阴天',
        'features': [
            '天空被云层完全覆盖，几乎看不到蓝天',
            '光线均匀散射，无明显方向性',
            '画面灰暗，色温偏冷',
            '阴影极淡或消失',
            '云层较厚，呈均匀灰色',
        ],
    },
    'rainy': {
        'icon': '雨天', 'cn': '雨天',
        'features': [
            '可见雨滴或雨丝，路面有积水反光',
            '色调偏暗偏冷，饱和度降低',
            '天空阴沉，常伴有乌云',
            '地面湿滑，有明显水渍',
            '可能伴有闪电或水雾',
        ],
    },
    'snowy': {
        'icon': '雪天', 'cn': '雪天',
        'features': [
            '地面、物体表面覆盖白色积雪',
            '画面整体偏白，高亮区域多',
            '色温偏冷蓝或纯白',
            '可能伴有飘落的雪花',
            '天空通常阴沉或灰白',
        ],
    },
    'sunny': {
        'icon': '晴天', 'cn': '晴天',
        'features': [
            '天空晴朗，蓝天白云',
            '阳光强烈，阴影清晰锐利',
            '色温暖黄，色彩饱和度高',
            '光线充足，画面明亮',
            '可能有明显的镜头光晕',
        ],
    },
}


# ===================================================================
#  工具函数
# ===================================================================
@st.cache_resource
def load_model():
    model = WeatherCNN(num_classes=6, activation='relu')
    if os.path.exists(MODEL_PATH):
        ckpt = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
        if isinstance(ckpt, dict) and 'config' in ckpt:
            act = ckpt['config'].get('activation', 'relu')
            model = WeatherCNN(num_classes=6, activation=act)
        model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
        model.to(DEVICE)
        model.eval()
        return model
    return None


def get_transform():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def predict(model, image: Image.Image):
    tf = get_transform()
    tensor = tf(image.convert('RGB')).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, predicted = torch.max(probs, 1)
    return predicted.item(), confidence.item(), probs[0].cpu().numpy()


def load_training_history():
    path = os.path.join(LOGS_DIR, 'training_history.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def load_experiment_summary():
    path = os.path.join(LOGS_DIR, 'experiment_summary.json')
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


def find_images(directory, keywords=None):
    if not os.path.isdir(directory):
        return []
    imgs = []
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        imgs.extend(glob.glob(os.path.join(directory, ext)))
    if keywords:
        imgs = [p for p in imgs if any(kw in os.path.basename(p).lower() for kw in keywords)]
    return sorted(imgs)


# ===================================================================
#  页面配置
# ===================================================================
st.set_page_config(
    page_title='天气图像分类系统',
    page_icon='🌤',
    layout='wide',
    initial_sidebar_state='expanded',
)


# ===================================================================
#  侧边栏
# ===================================================================
with st.sidebar:
    st.header('系统信息')
    st.divider()

    st.subheader('网络结构')
    st.code("""WeatherCNN
├─ Block1: Conv(3→32) + BN + Act ×2 → MaxPool
├─ Block2: Conv(32→64) + BN + Act ×2 → MaxPool
├─ Block3: Conv(64→128) + BN + Act ×2 → MaxPool
├─ Block4: Conv(128→256) + BN + Act ×2 → AdaptiveAvgPool
└─ Classifier
   ├─ Linear(4096→512) + BN + Act + Dropout(0.5)
   ├─ Linear(512→128) + Act + Dropout(0.3)
   └─ Linear(128→6)""", language=None)

    st.subheader('当前状态')
    model = load_model()
    if model is not None:
        total_p, _ = count_parameters(model)
        st.write(f'模型: 已加载')
        st.write(f'参数量: {total_p:,}')
        st.write(f'设备: {DEVICE}')
    else:
        st.write('模型: 未找到')

    st.divider()
    st.subheader('运行命令')
    st.code("""python train.py          # 训练
python experiment.py     # 实验对比
streamlit run app.py     # Web应用""", language=None)


# ===================================================================
#  首页
# ===================================================================
st.title('天气图像分类系统')
st.caption('基于 PyTorch CNN 的天气场景智能识别')

history = load_training_history()
best_acc = history['results']['best_test_acc'] if history else '--'
total_p, _ = count_parameters(model) if model else (0, 0)

c1, c2, c3, c4 = st.columns(4)
c1.metric('模型', 'WeatherCNN', f'{total_p:,} 参数')
c2.metric('输入', f'{IMG_SIZE}x{IMG_SIZE}', 'RGB')
c3.metric('类别', '6', '天气类型')
c4.metric('最佳准确率', f'{best_acc}%' if isinstance(best_acc, float) else best_acc, '测试集')

st.divider()


# ===================================================================
#  Tab 页面
# ===================================================================
tab1, tab2, tab3, tab4 = st.tabs(['图片识别', '训练过程', '模型配置', '类别百科'])

# ── Tab 1: 图片识别 ──
with tab1:
    st.subheader('上传图片进行天气识别')

    if model is None:
        st.error(f'无法加载模型: {MODEL_PATH}\n\n请先运行 python train.py 训练模型。')
    else:
        col_upload, col_result = st.columns([1, 1])

        with col_upload:
            uploaded = st.file_uploader(
                '选择一张天气图片',
                type=['jpg', 'jpeg', 'png', 'bmp', 'webp'],
                help='支持 JPG、PNG、BMP、WebP 格式',
            )
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption='上传的图片', use_container_width=True)

        with col_result:
            if uploaded:
                with st.spinner('推理中...'):
                    pred_class, confidence, probs = predict(model, img)

                cn_name = CLASS_NAMES[pred_class][1]
                en_name = CLASS_NAMES[pred_class][0]

                # 结果展示
                st.success(f'识别结果: **{cn_name}** ({en_name})')
                st.metric('置信度', f'{confidence*100:.2f}%')

                st.markdown('**各类别概率分布**')
                fig_bar, ax = plt.subplots(figsize=(8, 3.5))
                labels = [CLASS_NAMES[i][1] for i in range(6)]
                bar_colors = ['#42A5F5', '#B0BEC5', '#78909C', '#546E7A', '#E3F2FD', '#FFC107']
                bars = ax.barh(labels, probs * 100, color=bar_colors, edgecolor='white')
                bars[pred_class].set_edgecolor('#1565C0')
                bars[pred_class].set_linewidth(2)
                for i, (bar, prob) in enumerate(zip(bars, probs)):
                    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                            f'{prob*100:.1f}%', va='center', fontsize=10)
                cn_set_xlabel(ax, '概率 (%)')
                ax.set_xlim(0, 105)
                ax.invert_yaxis()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_bar)
                plt.close()

                st.markdown('**各类别详情**')
                for i in range(6):
                    en, cn = CLASS_NAMES[i]
                    prob = probs[i]
                    marker = '**[预测]** ' if i == pred_class else '  '
                    st.progress(float(prob), text=f'{marker}{cn} ({en}): {prob*100:.2f}%')
            else:
                st.info('请在左侧上传图片进行识别')


# ── Tab 2: 训练过程 ──
with tab2:
    st.subheader('训练过程可视化')

    history = load_training_history()
    experiment_summary = load_experiment_summary()

    if history:
        h = history.get('history', history)
        epochs = list(range(1, len(h.get('train_loss', [])) + 1))

        col_loss, col_acc = st.columns(2)

        with col_loss:
            st.markdown('**损失曲线**')
            fig_loss, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, h.get('train_loss', []), 'b-o', label='训练损失', markersize=3)
            ax.plot(epochs, h.get('test_loss', []), 'r-s', label='测试损失', markersize=3)
            cn_set_xlabel(ax, 'Epoch')
            cn_set_ylabel(ax, 'Loss')
            cn_set_title(ax, '训练/测试损失')
            fp = get_cn_font_prop()
            ax.legend(prop=fp)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_loss)
            plt.close()

        with col_acc:
            st.markdown('**准确率曲线**')
            fig_acc, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epochs, h.get('train_acc', []), 'g-o', label='训练准确率', markersize=3)
            ax.plot(epochs, h.get('test_acc', []), 'm-s', label='测试准确率', markersize=3)
            cn_set_xlabel(ax, 'Epoch')
            cn_set_ylabel(ax, 'Accuracy (%)')
            cn_set_title(ax, '训练/测试准确率')
            ax.legend(prop=fp)
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_acc)
            plt.close()

        st.markdown('**学习率调度**')
        fig_lr, ax = plt.subplots(figsize=(8, 3))
        ax.plot(epochs, h.get('lr', []), 'purple', linewidth=2)
        cn_set_xlabel(ax, 'Epoch')
        cn_set_ylabel(ax, 'Learning Rate')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig_lr)
        plt.close()
    else:
        st.warning('未找到训练历史。请先运行 python train.py。')

    st.divider()

    st.markdown('**混淆矩阵**')
    cm_images = find_images(LOGS_DIR, ['confusion'])
    if cm_images:
        for img_path in cm_images[:2]:
            st.image(img_path, caption=os.path.basename(img_path), use_container_width=True)
    else:
        st.info('运行训练后自动生成混淆矩阵图片。')

    st.divider()

    st.markdown('**实验对比**')
    if experiment_summary:
        st.dataframe(experiment_summary, use_container_width=True, hide_index=True)
        comparison_img = find_images(LOGS_DIR, ['experiment_comparison'])
        if comparison_img:
            st.image(comparison_img[0], caption='实验对比图', use_container_width=True)
    else:
        st.info('运行 python experiment.py 生成实验对比数据。')


# ── Tab 3: 模型配置 ──
with tab3:
    st.subheader('模型配置详情')

    st.markdown('**网络结构**')
    st.code("""WeatherCNN(num_classes=6, activation="leaky_relu")

Block 1: Conv(3→32, 3x3) + BN + LeakyReLU x2 → MaxPool(2) → Dropout2d(0.1)
Block 2: Conv(32→64, 3x3) + BN + LeakyReLU x2 → MaxPool(2) → Dropout2d(0.15)
Block 3: Conv(64→128, 3x3) + BN + LeakyReLU x2 → MaxPool(2) → Dropout2d(0.2)
Block 4: Conv(128→256, 3x3) + BN + LeakyReLU x2 → AdaptiveAvgPool(4,4) → Dropout2d(0.25)

Classifier:
  Flatten → Linear(4096→512) + BN + LeakyReLU + Dropout(0.5)
  → Linear(512→128) + LeakyReLU + Dropout(0.3)
  → Linear(128→6)""", language=None)

    st.markdown('**超参数配置**')
    if history:
        cfg = history.get('config', {})
        params = {
            '输入尺寸': f'{IMG_SIZE}x{IMG_SIZE} RGB',
            '输出类别': '6 类天气',
            '优化器': cfg.get('optimizer', 'adamw'),
            '学习率': cfg.get('learning_rate', 0.0008),
            '权重衰减': cfg.get('weight_decay', 1e-3),
            '调度器': cfg.get('scheduler', 'cosine'),
            '激活函数': cfg.get('activation', 'leaky_relu'),
            '标签平滑': cfg.get('label_smoothing', 0.1),
            'Batch Size': cfg.get('batch_size', 32),
            'Epochs': cfg.get('epochs', 40),
            '设备': cfg.get('device', str(DEVICE)),
        }
    else:
        params = {
            '输入尺寸': f'{IMG_SIZE}x{IMG_SIZE} RGB',
            '输出类别': '6 类天气',
            '优化器': 'AdamW',
            '学习率': 0.0008,
            '调度器': 'CosineAnnealingLR',
            '激活函数': 'LeakyReLU',
            'Batch Size': 32,
            'Epochs': 40,
        }

    col_a, col_b = st.columns(2)
    items = list(params.items())
    mid = (len(items) + 1) // 2
    for k, v in items[:mid]:
        col_a.write(f'**{k}:** {v}')
    for k, v in items[mid:]:
        col_b.write(f'**{k}:** {v}')

    st.markdown('**模型参数量**')
    if model:
        total_p, train_p = count_parameters(model)
        col_m1, col_m2 = st.columns(2)
        col_m1.metric('总参数量', f'{total_p:,}')
        col_m2.metric('可训练参数', f'{train_p:,}')

    st.markdown('**数据增强策略**')
    st.dataframe([
        {'方法': 'RandomCrop', '参数': 'padding=8', '说明': '随机裁剪'},
        {'方法': 'RandomHorizontalFlip', '参数': 'p=0.5', '说明': '水平翻转'},
        {'方法': 'RandomRotation', '参数': '15度', '说明': '随机旋转'},
        {'方法': 'ColorJitter', '参数': 'brightness=0.3, contrast=0.3', '说明': '颜色抖动'},
        {'方法': 'RandomGrayscale', '参数': 'p=0.2', '说明': '随机灰度化'},
        {'方法': 'RandomErasing', '参数': 'p=0.2', '说明': '随机擦除'},
        {'方法': 'Label Smoothing', '参数': '0.1', '说明': '标签平滑'},
        {'方法': 'ImageNet Normalize', '参数': 'mean/std', '说明': '标准化'},
    ], use_container_width=True, hide_index=True)


# ── Tab 4: 类别百科 ──
with tab4:
    st.subheader('天气类别说明')

    for en_name, info in WEATHER_INFO.items():
        with st.expander(f'{info["cn"]} ({en_name})', expanded=False):
            st.write(f'**类别标识:** {en_name} -> {info["cn"]}')
            st.write('**视觉特征:**')
            for feat in info['features']:
                st.write(f'- {feat}')

    st.divider()
    st.subheader('类别对比')
    specs = {
        'cloudy': ('中等', '中等', '偏冷灰', '中等'),
        'foggy': ('偏高', '极低', '冷灰白', '极低'),
        'overcast': ('偏低', '低', '冷灰', '低'),
        'rainy': ('偏低', '中等', '冷暗', '低'),
        'snowy': ('极高', '高', '冷蓝白', '低'),
        'sunny': ('高', '高', '暖黄', '高'),
    }
    comparison = []
    for en, info in WEATHER_INFO.items():
        b, c, t, s = specs[en]
        comparison.append({
            '类别': info['cn'],
            '英文名': en,
            '亮度': b, '对比度': c, '色温': t, '饱和度': s,
        })
    st.dataframe(comparison, use_container_width=True, hide_index=True)


# ===================================================================
#  页脚
# ===================================================================
st.divider()
st.caption('天气图像分类系统 | Powered by PyTorch & Streamlit')
