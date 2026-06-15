# 👁️ Spatial-Tracker: 基于空间计算与特征增强的裸手交互追踪系统

![Python](https://img.shields.io/badge/python-3.8%2B-brightgreen)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10%2B-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-red)

> **面向边缘设备的轻量化视觉交互解决方案** > 本系统针对复杂动态环境下，端侧设备算力受限、高频镜头抖动易跟丢等痛点，进行了底层的管线重构。通过级联检测与零样本（Zero-shot）追踪技术，在保证极低静态功耗的同时，实现了对较大实体目标（如水杯、人脸等）的高帧率稳健交互追踪。

---

## ⚠️ 关于模型权重的特别说明 (Weights Disclaimer)

**本项目核心代码已全面开源。** 因 GitHub 网页端对单文件存在 25MB 的上传限制，本项目中提及的预训练权重文件（如 `hand_yolov8n.pt` 及重构训练的 `resnet_50-size-256-loss-0.0642.pth`，单文件超 90MB）未包含在当前仓库中。
* 本仓库旨在展示**系统底层管线架构、状态机控制逻辑及 1 Euro 动态滤波核心算法**。

---

## 💡 核心工程创新与代码导读

### 1. 纯 O(1) 的自适应防抖 (1 Euro Filter)
* **核心逻辑详见:** `1Euro_filter.py`
针对姿态网络输出的高频底噪，独立引入并重写了适用于 HCI 的 1 Euro 动态滤波器。通过“速度-截止频率”的自适应映射，在保持纯 $O(1)$ 极低算力开销的前提下，有效平衡了“低速抗抖”与“高速跟手”的工程博弈。

### 2. 状态机驱动的延迟计算 (Lazy Evaluation)
在底层逻辑中构建了完整的交互状态机。系统仅在双指界定 ROI 区域后，才触发深度计算与 CLAHE 亮度通道增强，通过 Lazy Evaluation 策略按需调度计算资源，显著控制了边缘侧设备的静态功耗。

### 3. 零样本特征追踪 (Zero-Shot Tracking)
剥离了开源框架单一的闭集分类逻辑。在锁定目标瞬间，系统动态初始化 CSRT 相关滤波追踪器，提取目标 HOG 梯度特征与色彩空间指纹。无需对未知目标进行网络重训练，即可实现物理轮廓的平滑追踪。

---

## 📁 核心代码目录结构 (Repository Structure)

```text
Bare-hand-tracking/
├── 1Euro_filter.py         # 核心创新：O(1)复杂度的动态防抖滤波算法实现
├── models/
│   └── resnet.py           # 姿态估计网络架构定义与重构
├── hand_data_iter/
│   ├── datasets.py         # 自定义数据集加载与动态批处理管线
│   └── data_agu.py         # 空间增广与预处理脚本
└── README.md               # 项目工程白皮书
