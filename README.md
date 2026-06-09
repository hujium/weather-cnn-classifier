# 🌤️ 天气图像分类系统

[![GitHub](https://img.shields.io/badge/GitHub-hujium%2Fweather--CNN-blue?logo=github)](https://github.com/hujium/weather-cnn)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://github.com/hujium/weather-cnn/blob/main/LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-yellow.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

**GitHub 项目地址:** https://github.com/hujium/weather-cnn

基于 PyTorch 卷积神经网络（CNN）的天气图像自动分类系统，支持识别6种天气类型。

---

## 📋 项目简介

本项目使用深度学习技术，通过卷积神经网络（CNN）对天气图像进行自动分类。系统能够识别以下6种天气类型：

| 类别 | 英文名 | 图标 |
|:---:|:---:|:---:|
| 多云 | cloudy | ⛅ |
| 雾天 | foggy | 🌫️ |
| 阴天 | overcast | ☁️ |
| 雨天 | rainy | 🌧️ |
| 雪天 | snowy | ❄️ |
| 晴天 | sunny | ☀️ |

用户可以通过 Streamlit Web 界面上传天气图片，系统会自动识别并给出分类结果。

---

## 📁 目录结构

```
weather-cnn/
├── launch.bat              # 一键启动脚本（自动检测环境、安装依赖、训练模型、启动网页）
├── webui.bat               # 仅启动 Web 界面（环境已就绪时使用）
├── model.py                # CNN模型定义（统一模型，供训练和推理共用）
├── train.py                # 模型训练脚本（支持预设配置和命令行覆盖）
├── experiment.py           # 多实验对比脚本（对比优化器、激活函数等）
├── app.py                  # Streamlit Web应用（识别、训练可视化、配置展示）
├── utils.py                # 工具函数（中文字体配置等）
├── requirements.txt        # 项目依赖
├── README.md               # 项目说明文档
├── model/                  # 模型权重目录
│   └── weather_cnn.pth     # 训练好的模型权重
├── logs/                   # 训练日志与可视化
├── dataset/                # 数据集
│   ├── train/              # 训练集 (420张)
│   ├── test/               # 测试集 (180张)
│   └── all/                # 全量数据 (备用)
└── fonts/                  # 字体文件目录
```

---

## 🚀 部署流程

### 流程一：一键启动（推荐）

1. 双击运行 `launch.bat`
2. 脚本将自动完成以下所有步骤：
   - 检测 Python3 环境（如未安装会弹窗询问并自动安装）
   - 检测并安装所有必要的 Python 依赖包
   - 首次运行时自动训练模型
   - 运行快速测试验证模型
   - 启动 Streamlit Web 应用并打开浏览器
3. 等待浏览器自动打开 `http://localhost:8501`，即可使用

> **注意：** 首次运行 `launch.bat` 时，训练模型需要一定时间（取决于硬件配置），请耐心等待。

#### launch.bat 交互说明

脚本运行时会依次弹出交互菜单：

**流程三 — 选择训练配置：**
```
  1. default   — 默认配置（快速验证，Adam + ReLU）
  2. best      — 最优配置（推荐，AdamW + LeakyReLU）★
  3. 自定义    — 手动输入所有参数
  4. 跳过训练  — 直接使用已有模型
```
选择 `3`（自定义）后，会依次询问：优化器、学习率、激活函数、调度器、训练轮数、批大小、权重衰减，每项都有默认值，直接回车即可使用。

**流程四 — 选择实验模式：**
```
  1. 快速对比（优化器，15轮）
  2. 快速对比（激活函数，15轮）
  3. 完整对比（优化器 + 激活函数，30轮）
  4. 自定义实验参数
  5. 跳过实验 — 直接启动 Web 应用
```
选择 `4`（自定义）后，会询问实验模式和训练轮数。

> 所有菜单直接按回车即可使用默认值，无需逐项输入。

### 流程二：仅启动 Web 界面

如果模型已训练完成，只需启动 Web 界面：

1. 双击运行 `webui.bat`
2. 浏览器将自动打开 `http://localhost:8501`

### 流程三：手动部署（高级用户）

#### 1. 安装 Python

确保已安装 Python 3.8+，下载地址：https://www.python.org/downloads/

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

#### 3. 训练模型

```bash
# 使用最优配置训练（推荐）
python train.py

# 使用特定预设配置
python train.py --config default
python train.py --config best

# 自定义参数
python train.py --epochs 50 --lr 0.001 --optimizer adam --activation relu
```

**可用的预设配置：**

| 配置名 | 优化器 | 学习率 | 激活函数 | 调度器 | Epochs |
|:---:|:---:|:---:|:---:|:---:|:---:|
| `default` | Adam | 0.001 | ReLU | Plateau | 30 |
| `best` | AdamW | 0.0008 | LeakyReLU | Cosine | 40 |

#### 参数通俗解释

| 参数 | 含义 | 通俗解释 |
|:---|:---|:---|
| **optimizer（优化器）** | 模型"怎么学" | 每次算出错误后用什么策略调整参数。Adam=通用稳健，AdamW=Adam升级版（学得更扎实），SGD=传统方法（慢但后期好） |
| **lr（学习率）** | 每步学多大 | 每次调整参数的幅度。太大容易学过头（震荡），太小学太慢。0.001是常用起点，0.0008更保守稳定 |
| **weight_decay（权重衰减）** | 防止死记硬背 | 正则化惩罚，防止模型把训练数据背下来而不会举一反三。值越大约束越强 |
| **scheduler（调度器）** | 什么时候该学慢点 | 训练初期大步快走，后期小步微调。Plateau=验证集不进步就减半学习率，Cosine=平滑地从大到小 |
| **activation（激活函数）** | 神经元怎么反应 | 给网络加入非线性能力。ReLU=经典简单，LeakyReLU=ReLU改进（不会"死掉"），GELU=更平滑（计算稍慢） |
| **epochs（训练轮数）** | 看几遍数据 | 每个epoch把全部训练数据看一遍。太少学不够，太多会过拟合（死记硬背） |
| **batch_size（批大小）** | 一次看几张图 | 每次喂给模型的图片数量。太小噪声大，太大显存不够。32是常用平衡值 |

#### 两个主要预设的区别

| 对比项 | `default`（默认） | `best`（最优）⭐ |
|:---|:---|:---|
| 定位 | 快速验证，适合测试环境 | 正式训练，追求最佳效果 |
| 优化器 | Adam（通用） | AdamW（更强正则化） |
| 学习率 | 0.001（标准值） | 0.0008（更保守，训练更稳） |
| 激活函数 | ReLU（经典） | LeakyReLU（不会神经元死亡） |
| 调度器 | Plateau（看验证集降） | Cosine（平滑衰减） |
| 训练轮数 | 30轮 | 40轮（更充分收敛） |
| 预计耗时 | 较短 | 稍长，但效果更好 |

#### 命令行覆盖参数说明

除了使用预设配置，还可以通过命令行参数覆盖任意项：

| 参数 | 说明 | 示例 |
|:---|:---|:---|
| `--config` | 选择预设配置名 | `--config best` |
| `--epochs` | 覆盖训练轮数 | `--epochs 50`（训练50轮） |
| `--lr` | 覆盖学习率 | `--lr 0.0005`（更小的学习率） |
| `--optimizer` | 覆盖优化器 | `--optimizer adamw` |
| `--activation` | 覆盖激活函数 | `--activation gelu` |
| `--scheduler` | 覆盖学习率调度器 | `--scheduler cosine` |
| `--batch-size` | 覆盖批大小 | `--batch-size 16` |
| `--json-progress` | 输出JSON格式进度 | `--json-progress`（供程序消费） |

示例：使用 best 预设但只训练20轮：
```bash
python train.py --config best --epochs 20
```

#### 4. 多实验对比

```bash
# 运行全部对比实验
python experiment.py

# 快速模式（15 epochs）
python experiment.py --fast

# 仅对比优化器 / 激活函数
python experiment.py --only-optimizer
python experiment.py --only-activation
```

#### 5. 启动 Web 应用

```bash
streamlit run app.py --server.port 8501
```

浏览器访问 `http://localhost:8501`，即可使用以下功能：
- 📸 **图片识别** — 上传天气图片，AI 实时分类
- 📈 **训练过程** — 查看训练曲线、混淆矩阵、实验对比
- 🏗️ **模型配置** — 查看网络结构、超参数详情
- 📝 **类别百科** — 6类天气的视觉特征说明

---

## 🧠 模型说明

### CNN 网络架构

```
WeatherCNN
├─ Features
│  ├─ Block1: Conv(3→32, 3×3) + BN + Act ×2 → MaxPool(2) → Dropout2d(0.1)
│  ├─ Block2: Conv(32→64, 3×3) + BN + Act ×2 → MaxPool(2) → Dropout2d(0.15)
│  ├─ Block3: Conv(64→128, 3×3) + BN + Act ×2 → MaxPool(2) → Dropout2d(0.2)
│  └─ Block4: Conv(128→256, 3×3) + BN + Act ×2 → AdaptiveAvgPool(4,4) → Dropout2d(0.25)
└─ Classifier
   ├─ Flatten
   ├─ Linear(4096→512) + BN1d + Act + Dropout(0.5)
   ├─ Linear(512→128) + Act + Dropout(0.3)
   └─ Linear(128→6)
```

**支持的激活函数：**
- `relu` — ReLU (默认)
- `leaky_relu` — LeakyReLU(0.1) ⭐ 推荐
- `gelu` — GELU

### 训练策略

| 项目 | 配置 |
|:---|:---|
| 损失函数 | CrossEntropyLoss + Label Smoothing (0.1) |
| 优化器 | AdamW (lr=0.0008, weight_decay=1e-3) ⭐ 推荐 |
| 学习率调度 | CosineAnnealingLR (eta_min=1e-6) |
| 权重初始化 | Kaiming Normal |
| 输入尺寸 | 128×128 RGB |

### 数据增强

| 方法 | 参数 | 说明 |
|:---|:---|:---|
| RandomCrop | padding=8 | 随机裁剪 |
| RandomHorizontalFlip | p=0.5 | 水平翻转 |
| RandomRotation | ±15° | 随机旋转 |
| ColorJitter | brightness=0.3, contrast=0.3 | 颜色抖动 |
| RandomGrayscale | p=0.2 | 随机灰度化 |
| RandomErasing | p=0.2 | 随机擦除 |
| ImageNet Normalize | mean/std | 标准化 |

---

## 📊 数据集说明

- **总图片数量：** 600张
- **天气类别数：** 6类
- **每类图片数：** 100张
- **训练集比例：** 70%（每类70张，共420张）
- **测试集比例：** 30%（每类30张，共180张）

---

## 📝 注意事项

1. `launch.bat` 仅适用于 Windows 系统
2. 首次运行 `launch.bat` 会自动训练模型，请耐心等待
3. 如需重新训练，删除 `model/weather_cnn.pth` 后重新运行
4. 训练时推荐使用 GPU（CPU 也可以运行，但较慢）
5. 如 `launch.bat` 启动失败，可运行 `webui.bat` 仅启动 Web 界面
6. 如需重新划分数据集，可使用 `dataset/all/` 目录中的全量数据

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源协议。

详情请查看 [LICENSE](LICENSE) 文件。
