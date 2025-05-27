# 实验记录：resnet50_bs32_lr0.001_e10 (branch)

## 基本信息
- **框架**：TensorFlow（TF） & PyTorch
- **模型**：ResNet50（冻结全部卷积层）+102多头分类器
- **分类任务**：Oxford Flowers 102
- **输入尺寸**：224x224
- **类别数**：102
- **训练轮数**：10
- **批次大小**：32
- **学习率**：0.001

## 模型结构
```text
ResNet50 (预训练权重, include_top=False)
→ GlobalAveragePooling2D
→ Dense(128, relu)
→ Dropout(0.5)
→ Dense(102, softmax)
