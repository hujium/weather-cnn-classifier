#!/usr/bin/env python3
"""
天气图像分类 - PyTorch CNN模型
6类天气: sunny, cloudy, overcast, rainy, snowy, foggy
基于云朵图形、灰度、色温等特征进行识别
"""

import os
import sys
import json
import time
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# ============ 配置 ============
DATA_DIR = "/storage/emulated/0/Documents/ClawOutPut/weather-cnn/dataset"
MODEL_DIR = "/storage/emulated/0/Documents/ClawOutPut/weather-cnn/model"
LOG_DIR = "/storage/emulated/0/Documents/ClawOutPut/weather-cnn/logs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# 超参数
BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 0.001
IMG_SIZE = 128
NUM_CLASSES = 6
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ['cloudy', 'foggy', 'overcast', 'rainy', 'snowy', 'sunny']
CLASS_NAMES_CN = ['多云', '雾天', '阴天', '雨天', '雪天', '晴天']

print(f"🖥️  使用设备: {DEVICE}")
print(f"📁 数据目录: {DATA_DIR}")
print(f"📊 类别数: {NUM_CLASSES}")


# ============ 数据加载与增强 ============
def get_data_loaders():
    """加载训练集和测试集，含数据增强"""
    
    # 训练集增强：模拟不同天气条件下的图像变化
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),  # 灰度特征增强
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    
    # 测试集：仅Resize和Normalize
    test_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=train_transform)
    test_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'test'), transform=test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print(f"\n📊 数据集统计:")
    print(f"  训练集: {len(train_dataset)} 张")
    print(f"  测试集: {len(test_dataset)} 张")
    print(f"  类别映射: {train_dataset.class_to_idx}")
    
    return train_loader, test_loader, train_dataset.class_to_idx


# ============ CNN模型定义 ============
class WeatherCNN(nn.Module):
    """
    天气分类CNN模型
    特征提取：
    - 低层: 边缘、纹理（云朵形状、雨丝、雪花）
    - 中层: 色温特征（冷暖色调）
    - 高层: 灰度特征、整体场景理解
    """
    def __init__(self, num_classes=6):
        super(WeatherCNN, self).__init__()
        
        # 特征提取器 - 多尺度卷积
        self.features = nn.Sequential(
            # Block1: 低层特征 - 边缘、纹理
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),
            
            # Block2: 中层特征 - 色温、色彩分布
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Block3: 高层特征 - 云朵形态、场景
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),
            
            # Block4: 抽象特征
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Dropout2d(0.3),
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ============ 训练函数 ============
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs):
    """训练模型并返回最佳模型"""
    
    train_losses = []
    test_losses = []
    train_accs = []
    test_accs = []
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"\n🚀 开始训练 ({num_epochs} epochs)")
    print(f"{'='*70}")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # ---- 训练阶段 ----
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # ---- 测试阶段 ----
        model.eval()
        test_running_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                test_running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_loss = test_running_loss / len(test_loader.dataset)
        test_acc = 100.0 * test_correct / test_total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        # 学习率调度
        scheduler.step(test_loss)
        
        elapsed = time.time() - start_time
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            star = " ⭐ 最佳"
        else:
            star = ""
        
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1:2d}/{num_epochs}] "
              f"训练损失:{train_loss:.4f} 准确率:{train_acc:.1f}% | "
              f"测试损失:{test_loss:.4f} 准确率:{test_acc:.1f}% | "
              f"LR:{lr:.6f} | {elapsed:.1f}s{star}")
    
    print(f"{'='*70}")
    print(f"🏆 最佳测试准确率: {best_acc:.1f}%")
    
    # 加载最佳模型
    model.load_state_dict(best_model_wts)
    
    history = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_acc': best_acc,
    }
    
    return model, history


# ============ 可视化 ============
def plot_training_history(history):
    """绘制训练曲线"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 损失曲线
    axes[0].plot(history['train_losses'], 'b-o', label='训练损失', markersize=3)
    axes[0].plot(history['test_losses'], 'r-o', label='测试损失', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('损失值变化曲线')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 准确率曲线
    axes[1].plot(history['train_accs'], 'b-o', label='训练准确率', markersize=3)
    axes[1].plot(history['test_accs'], 'r-o', label='测试准确率', markersize=3)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('准确率变化曲线')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(LOG_DIR, 'training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📈 训练曲线已保存: {path}")


def plot_confusion_matrix(model, test_loader, class_names):
    """绘制混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title('混淆矩阵', fontsize=14)
    plt.colorbar(im)
    
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(class_names)
    
    # 添加数值标注
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')
    plt.tight_layout()
    
    path = os.path.join(LOG_DIR, 'confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"📊 混淆矩阵已保存: {path}")
    
    return all_labels, all_preds


# ============ 主函数 ============
def main():
    # 1. 加载数据
    print("\n📂 [1/4] 加载数据集...")
    train_loader, test_loader, class_to_idx = get_data_loaders()
    
    # 2. 创建模型
    print("\n🏗️  [2/4] 构建CNN模型...")
    model = WeatherCNN(num_classes=NUM_CLASSES).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  模型参数: {total_params:,} (可训练: {trainable_params:,})")
    
    # 3. 训练
    print("\n🚀 [3/4] 训练模型...")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)
    
    model, history = train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, NUM_EPOCHS)
    
    # 4. 评估
    print("\n📊 [4/4] 模型评估...")
    
    # 保存模型
    model_path = os.path.join(MODEL_DIR, 'weather_cnn.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'num_classes': NUM_CLASSES,
        'img_size': IMG_SIZE,
        'model_arch': 'WeatherCNN',
    }, model_path)
    print(f"  💾 模型已保存: {model_path}")
    
    # 保存训练历史
    history_path = os.path.join(LOG_DIR, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"  📈 历史已保存: {history_path}")
    
    # 绘图
    plot_training_history(history)
    all_labels, all_preds = plot_confusion_matrix(model, test_loader, CLASS_NAMES)
    
    # 分类报告
    print("\n📋 分类报告:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES, digits=3))
    
    print("\n" + "=" * 60)
    print(f"✅ 训练完成！最佳准确率: {history['best_acc']:.1f}%")
    print(f"  模型文件: {model_path}")
    print(f"  训练曲线: {os.path.join(LOG_DIR, 'training_curves.png')}")
    print(f"  混淆矩阵: {os.path.join(LOG_DIR, 'confusion_matrix.png')}")
    print("=" * 60)


if __name__ == "__main__":
    main()
