# MolE 深度神经网络模型架构详解

## 概述

MolE（Molecular Embeddings）是Recursion公司开发的化学基础模型，创新性地结合了图神经网络和Transformer架构，专门用于化学分子的性质预测和嵌入生成。该模型采用编码器-解码器架构，通过两阶段预训练策略实现对分子结构的深度理解。

## 1. 整体架构概览

### 1.1 核心设计理念
- **混合架构**：结合图神经网络的局部结构感知能力和Transformer的长程依赖建模能力
- **分子特化**：专门针对分子结构特点设计的编码和注意力机制
- **多任务支持**：统一框架支持分类、回归和嵌入生成任务

### 1.2 架构流程图

```
输入层              编码器层                    解码器层
┌─────────────┐    ┌──────────────────────────┐    ┌─────────────────┐
│ SMILES      │    │  原子环境编码器            │    │  任务预测头      │
│ 分子字符串   │ -> │  (AtomEnvEmbeddings)     │ -> │ (TaskPredictionHead)│
│             │    │                          │    │                 │
│             │    │ ┌─────────────────────┐  │    │ ┌─────────────┐ │
│             │    │ │  BertEmbeddings     │  │    │ │  池化层     │ │
│             │    │ │  词嵌入+位置嵌入    │  │    │ │  分类器     │ │
│             │    │ └─────────────────────┘  │    │ │  损失函数   │ │
│             │    │          ↓               │    │ └─────────────┘ │
│             │    │ ┌─────────────────────┐  │    │                 │
│             │    │ │  BertEncoder        │  │    │                 │
│             │    │ │  多层Transformer    │  │    │                 │
│             │    │ │  解耦注意力机制     │  │    │                 │
│             │    │ └─────────────────────┘  │    │                 │
└─────────────┘    └──────────────────────────┘    └─────────────────┘
```

## 2. 核心组件详细架构

### 2.1 数据预处理层 (MolDataset)

**功能**: 将SMILES字符串转换为模型可处理的图结构数据

**关键组件**:
- **分子解析**: 使用RDKit将SMILES转换为分子对象
- **原子环境编码**: 基于Morgan指纹算法计算原子环境特征
- **图结构构建**: 生成节点-边-距离矩阵的图表示
- **特殊标记处理**: 支持CLS、UNK、MASK、PAD等特殊token

**数据转换流程**:
```
SMILES字符串 → RDKit分子对象 → 原子环境计算 → Token序列 → 图数据结构
     "CCO"        分子对象        [1,15,7]      [CLS,1,15,7]    PyG.Data
```

### 2.2 嵌入层 (BertEmbeddings)

**架构设计**:
```python
class BertEmbeddings:
    word_embeddings: nn.Embedding(vocab_size=207, hidden_size=768)
    position_embeddings: nn.Embedding(max_pos=512, hidden_size=768)  
    token_type_embeddings: nn.Embedding(type_vocab_size=2, hidden_size=768)
    LayerNorm: LayerNorm(hidden_size=768)
    dropout: StableDropout(p=0.1)
```

**功能**:
- **词嵌入**: 将原子环境ID映射为768维向量
- **位置嵌入**: 编码原子在分子序列中的位置信息
- **类型嵌入**: 区分不同类型的token（如普通原子vs特殊标记）
- **层归一化**: 稳定训练过程，加速收敛

### 2.3 Transformer编码器 (BertEncoder)

**整体架构**:
```
BertEncoder:
├── 12 × BertLayer
│   ├── DisentangledSelfAttention  # 解耦自注意力
│   ├── BertSelfOutput            # 输出投影+残差连接
│   ├── BertIntermediate          # 前馈网络第一层
│   └── BertOutput               # 前馈网络第二层+残差连接
├── 相对位置嵌入 (RelativePositionEmbedding)
└── 卷积层 (ConvLayer, 可选)
```

**每层结构 (BertLayer)**:
```
输入 → 解耦自注意力 → 残差连接+层归一化 → 前馈网络 → 残差连接+层归一化 → 输出
     ↗              ↘                    ↗            ↘
   Query/Key/Value  注意力权重          GELU激活      残差连接
```

### 2.4 解耦注意力机制 (DisentangledSelfAttention)

**核心创新**: 这是MolE的关键技术，继承自DeBERTa架构

**三种注意力类型**:
1. **内容到内容** (C2C): 传统的self-attention机制
2. **内容到位置** (C2P): 内容向量对相对位置的注意力
3. **位置到内容** (P2C): 相对位置对内容向量的注意力

**数学公式**:
```
AttentionScore = (Q×K^T + Q×K_rel^T + Q_rel×K^T) / √(d_k × scale_factor)
Attention = softmax(AttentionScore) × V
```

**架构实现**:
```python
class DisentangledSelfAttention:
    query_proj: nn.Linear(768, 768)     # Q投影
    key_proj: nn.Linear(768, 768)       # K投影  
    value_proj: nn.Linear(768, 768)     # V投影
    pos_query_proj: nn.Linear(768, 768) # 位置Q投影 (P2C)
    pos_key_proj: nn.Linear(768, 768)   # 位置K投影 (C2P)
    dropout: StableDropout(0.1)
```

### 2.5 任务预测头 (TaskPredictionHead)

**结构组成**:
```python
class TaskPredictionHead:
    dense: nn.Linear(768, 768)              # 特征变换层
    classifier: nn.Linear(768, num_classes)  # 分类/回归层
    dropout: StableDropout(0.1)             # 正则化
    loss_fn: BCEWithLogitsLoss/MSELoss      # 损失函数
```

**处理流程**:
```
CLS token表示 → Dropout → Dense层 → GELU激活 → Dropout → 分类器 → 预测结果
   [768]        [768]     [768]      [768]      [768]     [num_classes]
```

## 3. 数据流转详细过程

### 3.1 完整数据流
```
1. SMILES输入: "CCO" (乙醇分子)
   ↓
2. RDKit解析: 生成包含3个原子的分子对象
   ↓ 
3. 原子环境计算: 使用Morgan指纹
   - 原子0 (C): 环境ID = 1  
   - 原子1 (C): 环境ID = 15
   - 原子2 (O): 环境ID = 7
   ↓
4. Token化: [CLS, 1, 15, 7] (添加CLS标记)
   ↓
5. 图结构构建:
   - 节点特征: [CLS, 1, 15, 7]
   - 边信息: 原子间连接关系  
   - 距离矩阵: 原子间空间距离
   ↓
6. 批处理: to_dense_batch → 统一序列长度
   ↓
7. 嵌入层: Token → 768维向量
   ↓
8. Transformer编码: 12层注意力机制
   ↓
9. CLS提取: 取第一个位置的输出作为分子表示
   ↓
10. 预测头: 生成最终预测结果
```

### 3.2 注意力权重计算
```
对于分子"CCO":
- 内容注意力: 每个原子关注其他原子的化学环境
- 位置注意力: 考虑原子间的空间距离关系  
- 相对位置: 通过距离矩阵编码三维结构信息
```

## 4. 模型配置与超参数

### 4.1 核心架构参数
```yaml
Transformer配置:
  num_hidden_layers: 12        # Transformer层数
  hidden_size: 768            # 隐藏层维度
  num_attention_heads: 12     # 注意力头数
  attention_head_size: 64     # 每个注意力头维度 (768/12)
  intermediate_size: 3072     # 前馈网络中间层维度 (4×768)
  hidden_act: "gelu"          # 激活函数
  hidden_dropout_prob: 0.1    # Dropout概率
  attention_probs_dropout_prob: 0.1
  max_position_embeddings: 512 # 最大序列长度
  layer_norm_eps: 1e-12       # 层归一化epsilon
  
词汇表配置:
  vocab_size: 207             # 原子环境词汇表大小
  padding_idx: 0              # 填充token ID
  
相对位置编码:
  relative_attention: true
  max_relative_positions: 64  # 最大相对位置数
  position_buckets: -1        # 位置分桶数量
```

### 4.2 训练超参数
```yaml
优化器配置:
  optimizer: Adam
  learning_rate: 1e-4
  betas: [0.9, 0.999] 
  eps: 1e-6
  weight_decay: 0.0
  
调度器配置:
  scheduler: constant_with_warmup
  warmup_steps: 1000
  
训练配置:
  batch_size: 32
  gradient_clip_val: 1.0
  precision: 16               # 混合精度训练
  accumulate_grad_batches: 1
```

## 5. 关键技术特性

### 5.1 解耦注意力的优势

**传统注意力局限**:
- 只考虑内容信息，忽略位置关系
- 对于分子结构，原子间的空间关系至关重要

**解耦注意力改进**:
- **C2P**: 原子内容关注相对位置 → 「这个碳原子关注距离2个键长的氧原子」
- **P2C**: 相对位置关注原子内容 → 「距离1个键长的位置适合什么类型的原子」  
- **P2P**: 位置间的相互关系 → 「不同位置间的空间几何约束」

### 5.2 分子图处理创新

**原子环境编码**:
```python
# Morgan指纹计算示例
分子: H3C-CH2-OH
原子环境 (radius=0):
- C(甲基): 环境特征基于直接连接的H和C
- C(亚甲基): 环境特征基于连接的C、C、H、H  
- O(羟基): 环境特征基于连接的H和C
```

**距离矩阵编码**:
```
对于3原子分子CCO:
距离矩阵 = [[0, 1, 2],    # C-C距离1, C-O距离2
           [1, 0, 1],    # C-C距离1, C-O距离1  
           [2, 1, 0]]    # O-C距离1, O-C距离2
```

### 5.3 多任务学习架构

**支持任务类型**:
1. **分类任务**: 分子毒性、活性预测等
   - 损失函数: BCEWithLogitsLoss
   - 输出: sigmoid概率

2. **回归任务**: 溶解度、logP值预测等  
   - 损失函数: MSELoss
   - 输出: 连续数值

3. **嵌入生成**: 分子指纹、相似性搜索
   - 输出: CLS token的768维表示

**损失函数组合**:
```python
total_loss = main_loss + λ × aux_loss
# main_loss: 主任务损失
# aux_loss: 辅助任务损失 (可选)
# λ: 辅助损失权重系数
```

## 6. 架构优势与创新点

### 6.1 相比传统图神经网络(GNN)

**优势**:
- **长程依赖**: Transformer擅长建模远距离原子间相互作用
- **全局信息**: 通过self-attention机制获得全分子视野
- **并行计算**: 相比GNN的逐层传播，Transformer支持更好的并行化

**保留优势**:
- **局部结构**: 通过原子环境编码保留化学键信息
- **空间关系**: 通过相对位置编码保留三维几何信息

### 6.2 相比纯Transformer模型

**化学特化改进**:
- **原子环境Tokenization**: 比简单字符编码更适合分子结构
- **距离矩阵编码**: 捕捉分子的三维空间信息
- **解耦注意力**: 分离内容和位置，更好建模化学键关系

### 6.3 模型创新亮点

1. **混合架构设计**: 图网络 + Transformer的有机结合
2. **化学知识融入**: Morgan指纹等化学信息学方法的集成
3. **解耦注意力**: 从DeBERTa借鉴并适配分子结构特点
4. **两阶段预训练**: 自监督预训练 + 多任务微调
5. **工程化实现**: 基于PyTorch Lightning，易扩展易部署

## 7. 模型规模与性能

### 7.1 参数规模
```
总参数量: ~110M
├── 嵌入层: ~25M
│   ├── 词嵌入: 207×768 ≈ 159K
│   ├── 位置嵌入: 512×768 ≈ 393K  
│   └── 层归一化等: 少量参数
├── Transformer层: ~85M 
│   ├── 每层参数: ~7M (注意力+前馈网络)
│   └── 12层总计: ~84M
└── 预测头: 少量参数 (~1M)
```

### 7.2 计算复杂度
- **时间复杂度**: O(n²d) (n: 序列长度, d: 隐藏维度)
- **空间复杂度**: O(n²) (主要是注意力矩阵)
- **实际性能**: 支持GPU加速，混合精度训练

## 8. 使用场景与扩展性

### 8.1 适用场景
- **药物发现**: ADMET性质预测、活性预测
- **材料科学**: 材料性质预测、分子设计
- **化学信息学**: 分子相似性搜索、虚拟筛选

### 8.2 扩展可能性
- **多模态融合**: 结合光谱、图像等其他模态信息
- **更大规模**: 扩展到更大的Transformer架构
- **领域适应**: 针对特定化学领域进行定制化改进

---

## 总结

MolE通过创新性地结合图神经网络和Transformer架构，实现了对分子结构的深度建模。其解耦注意力机制、原子环境编码、相对位置编码等技术创新，使得模型能够同时捕捉分子的局部化学环境和全局结构特征，为化学领域的AI应用提供了强大而灵活的基础工具。

该架构的模块化设计和基于PyTorch Lightning的实现，确保了良好的可扩展性和工程实用性，为后续的模型改进和应用拓展奠定了坚实基础。