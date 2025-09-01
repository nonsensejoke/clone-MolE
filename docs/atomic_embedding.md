# MolE 原子Embedding提取指南

## 概述

本文档详细介绍如何从训练好的MolE模型中获取单个原子的embedding表示。MolE采用Transformer架构对分子进行编码，每个原子在经过多层注意力机制处理后，会得到一个包含丰富上下文信息的768维向量表示。

## 1. 基本原理

### 1.1 Transformer编码输出结构

MolE的Transformer编码器输出一个字典，包含以下关键信息：

```python
encoder_output = {
    "hidden_states": all_encoder_layers,    # 主要输出 - List[torch.Tensor]
    "attention_matrices": att_matrices      # 注意力权重矩阵 - List[torch.Tensor]
}
```

### 1.2 hidden_states详细结构

- **数据类型**: `List[torch.Tensor]`
- **列表长度**: 12（对应12层Transformer）
- **每个张量形状**: `[batch_size, sequence_length, hidden_size]`
- **示例维度**: `[32, 128, 768]` - 32个样本，最大序列长度128，隐藏维度768

```python
hidden_states[0]   # 第1层的输出 [batch_size, seq_len, 768]
hidden_states[1]   # 第2层的输出 [batch_size, seq_len, 768] 
# ...
hidden_states[11]  # 第12层的输出 [batch_size, seq_len, 768] ← 通常使用这一层
```

### 1.3 原子位置映射

对于分子"CCO"（乙醇）的例子：
```python
# 输入序列: [CLS, C_token, C_token, O_token]
# 位置映射:
# 位置 0: CLS token - 整个分子的全局表示
# 位置 1: 第1个C原子的表示
# 位置 2: 第2个C原子的表示  
# 位置 3: O原子的表示
```

## 2. 提取方法详解

### 2.1 方法一：直接从模型输出提取

```python
import torch
from torch_geometric.utils import to_dense_batch, to_dense_adj

def get_atom_embeddings(model, input_data):
    """
    从训练好的模型中获取所有原子的embedding
    
    Args:
        model: 训练好的MolE模型 (MolE类实例)
        input_data: 包含input_ids, input_mask等的数据字典
    
    Returns:
        torch.Tensor: 原子embeddings，形状为[batch_size, seq_len, 768]
    """
    model.eval()
    with torch.no_grad():
        # 通过模型的MolE编码器获取输出
        encoder_output = model.model.MolE(
            input_ids=input_data['input_ids'],
            input_mask=input_data['input_mask'],
            relative_pos=input_data.get('relative_pos', None),
            output_all_encoded_layers=True
        )
        
        # 获取最后一层的hidden states
        hidden_states = encoder_output["hidden_states"]
        final_layer_output = hidden_states[-1]  # [batch_size, seq_len, 768]
        
        return final_layer_output

# 使用示例
atom_embeddings = get_atom_embeddings(trained_model, batch_data)

# 获取特定原子的embedding（跳过CLS token）
first_atom_embedding = atom_embeddings[0, 1, :]  # 第一个原子
second_atom_embedding = atom_embeddings[0, 2, :] # 第二个原子
```

### 2.2 方法二：使用Encoder类（推荐）

```python
from mole.training.models.encoder import encoder

def get_atom_embeddings_with_encoder(encoder_model, input_data):
    """
    使用专门的Encoder类获取原子embeddings
    
    Args:
        encoder_model: encoder类实例
        input_data: 输入数据字典
        
    Returns:
        torch.Tensor: 原子embeddings
    """
    encoder_model.eval()
    with torch.no_grad():
        # 获取完整的hidden states
        encoder_output = encoder_model.MolE(
            input_ids=input_data['input_ids'],
            input_mask=input_data['input_mask'],
            relative_pos=input_data.get('relative_pos', None),
            output_all_encoded_layers=True
        )
        
        hidden_states = encoder_output["hidden_states"]
        final_embeddings = hidden_states[-1]  # 最后一层输出
        
        return final_embeddings
```

### 2.3 方法三：从PyG Data对象直接提取

```python
def get_molecular_atom_embeddings(trained_model, mol_data):
    """
    从PyG Data对象直接提取分子的原子embeddings
    
    Args:
        trained_model: 训练好的MolE模型
        mol_data: PyG Data对象，包含分子的图结构数据
    
    Returns:
        dict: 包含CLS和各原子embedding的字典
    """
    trained_model.eval()
    
    with torch.no_grad():
        # 准备输入数据（参考MolE的training_step）
        input_ids, input_mask = to_dense_batch(mol_data.x, mol_data.batch, fill_value=0)
        relative_pos = to_dense_adj(mol_data.edge_index, mol_data.batch, mol_data.edge_attr)
        
        # 通过模型获取embeddings
        encoder_output = trained_model.model.MolE(
            input_ids=input_ids,
            input_mask=input_mask,
            relative_pos=relative_pos,
            output_all_encoded_layers=True
        )
        
        # 提取最终层的输出
        hidden_states = encoder_output["hidden_states"] 
        final_embeddings = hidden_states[-1]  # [batch_size, seq_len, 768]
        
        # 构建结果字典
        results = {}
        batch_size, seq_len, hidden_dim = final_embeddings.shape
        
        for batch_idx in range(batch_size):
            batch_results = {}
            
            # CLS token (分子整体表示)
            batch_results['cls'] = final_embeddings[batch_idx, 0, :].cpu().numpy()
            
            # 各个原子的embedding
            for atom_pos in range(1, seq_len):
                if input_mask[batch_idx, atom_pos].item() > 0:  # 确保不是padding
                    batch_results[f'atom_{atom_pos-1}'] = final_embeddings[batch_idx, atom_pos, :].cpu().numpy()
            
            results[f'molecule_{batch_idx}'] = batch_results
        
        return results

# 使用示例
embeddings_dict = get_molecular_atom_embeddings(your_trained_model, your_mol_data)

# 访问特定原子的embedding
first_molecule_first_atom = embeddings_dict['molecule_0']['atom_0']  # shape: (768,)
first_molecule_cls = embeddings_dict['molecule_0']['cls']  # shape: (768,)
```

## 3. 实用工具函数

### 3.1 原子Embedding提取器

```python
def extract_individual_atoms(embeddings, exclude_cls=True):
    """
    从batch embeddings中提取单个原子的embeddings
    
    Args:
        embeddings: [batch_size, seq_len, hidden_size] 张量
        exclude_cls: 是否排除CLS token
    
    Returns:
        Dict[str, torch.Tensor]: 原子标识 -> embedding的映射
    """
    batch_size, seq_len, hidden_size = embeddings.shape
    atom_embeddings = {}
    
    start_idx = 1 if exclude_cls else 0  # 跳过CLS token
    
    for batch_idx in range(batch_size):
        for atom_idx in range(start_idx, seq_len):
            atom_key = f"batch_{batch_idx}_atom_{atom_idx - start_idx}"
            atom_embeddings[atom_key] = embeddings[batch_idx, atom_idx, :].clone()
    
    return atom_embeddings
```

### 3.2 Embedding可视化工具

```python
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_atom_embeddings(atom_embeddings_dict, method='tsne'):
    """
    可视化原子embeddings的分布
    
    Args:
        atom_embeddings_dict: 原子embeddings字典
        method: 降维方法 ('tsne', 'pca')
    """
    # 收集所有原子embeddings
    embeddings = []
    labels = []
    
    for mol_id, mol_data in atom_embeddings_dict.items():
        for atom_id, embedding in mol_data.items():
            if atom_id != 'cls':  # 跳过CLS token
                embeddings.append(embedding)
                labels.append(f"{mol_id}_{atom_id}")
    
    embeddings = np.array(embeddings)
    
    # 降维
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=2)
        embeddings_2d = reducer.fit_transform(embeddings)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
    
    # 添加标签（可选，数据点较少时）
    if len(labels) < 50:
        for i, label in enumerate(labels):
            plt.annotate(label, (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                        fontsize=8, alpha=0.7)
    
    plt.title(f'原子Embeddings分布 ({method.upper()})')
    plt.xlabel('维度1')
    plt.ylabel('维度2')
    plt.grid(True, alpha=0.3)
    plt.show()
```

## 4. 代码位置参考

### 4.1 核心文件位置

- **模型定义**: `mole/training/models/mole.py`
  - `AtomEnvEmbeddings` 类：原子环境编码器
  - `Supervised` 类：监督学习模型
  
- **Transformer组件**: `mole/training/nn/bert.py`
  - `BertEncoder` 类：Transformer编码器
  - `BertEmbeddings` 类：嵌入层
  
- **编码器模块**: `mole/training/models/encoder.py`
  - `encoder` 类：专门用于编码的简化模型

### 4.2 关键代码片段

```python
# 在Supervised.forward()中的关键部分
encoder_output = self.MolE(...)
hidden_states = encoder_output["hidden_states"]
ctx_layer = hidden_states[-1]  # 选择最后一层编码器输出
context_token = ctx_layer[:, 0]  # 选择CLS token
# ctx_layer[:, 1:] 就是各个原子的embeddings
```

## 5. 重要注意事项

### 5.1 数据处理要点

1. **层级选择**: 通常使用最后一层（`hidden_states[-1]`）的输出，包含最丰富的上下文信息
2. **CLS token处理**: 位置0是CLS token，代表整个分子；原子从位置1开始
3. **Padding处理**: 注意处理padding位置（`input_mask`为0的位置），这些位置的embedding无意义
4. **批处理**: 支持批量处理多个分子
5. **设备管理**: 确保所有张量在同一设备（CPU/GPU）上

### 5.2 Embedding特性

1. **上下文化表示**: 每个原子的embedding都包含了与分子中其他原子的交互信息
2. **维度固定**: 每个原子的embedding都是768维的向量
3. **动态表示**: 相同类型的原子在不同分子环境中会有不同的embedding
4. **解耦信息**: 包含内容信息（C2C）、位置信息（C2P、P2C）的综合表示

### 5.3 性能优化

```python
# 推理时的性能优化
@torch.no_grad()  # 禁用梯度计算
def optimized_embedding_extraction(model, data_loader):
    """
    批量优化的embedding提取
    """
    model.eval()  # 设置为评估模式
    
    all_embeddings = []
    
    for batch in data_loader:
        # 确保数据在正确的设备上
        batch = batch.to(model.device)
        
        # 获取embeddings
        embeddings = extract_batch_embeddings(model, batch)
        all_embeddings.append(embeddings.cpu())  # 移至CPU以节省GPU内存
    
    return torch.cat(all_embeddings, dim=0)
```

## 6. 应用场景

### 6.1 分子相似性分析

```python
def compute_atom_similarity(embedding1, embedding2, metric='cosine'):
    """
    计算两个原子embedding之间的相似性
    """
    if metric == 'cosine':
        from torch.nn.functional import cosine_similarity
        return cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()
    elif metric == 'euclidean':
        return torch.dist(embedding1, embedding2, p=2).item()
```

### 6.2 原子聚类分析

```python
from sklearn.cluster import KMeans

def cluster_atoms_by_embedding(atom_embeddings, n_clusters=5):
    """
    基于embedding对原子进行聚类
    """
    embeddings_array = np.array(list(atom_embeddings.values()))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings_array)
    
    # 返回聚类结果
    result = {}
    for i, (atom_id, embedding) in enumerate(atom_embeddings.items()):
        result[atom_id] = {
            'embedding': embedding,
            'cluster': cluster_labels[i]
        }
    
    return result
```

### 6.3 原子环境分析

```python
def analyze_atom_environment(model, molecules_list):
    """
    分析不同分子中相似原子的环境差异
    """
    carbon_embeddings = []
    oxygen_embeddings = []
    nitrogen_embeddings = []
    
    for mol_data in molecules_list:
        embeddings = get_molecular_atom_embeddings(model, mol_data)
        
        # 根据原子类型收集embeddings
        # 这里需要结合原子类型信息进行分类
        # 具体实现取决于数据格式
    
    # 分析各类原子的embedding分布特征
    return {
        'carbon_stats': analyze_embedding_statistics(carbon_embeddings),
        'oxygen_stats': analyze_embedding_statistics(oxygen_embeddings),
        'nitrogen_stats': analyze_embedding_statistics(nitrogen_embeddings)
    }
```

## 7. 故障排除

### 7.1 常见问题

1. **维度不匹配**: 确保模型配置与预训练权重一致
2. **设备错误**: 确保所有张量在同一设备上
3. **内存不足**: 处理大批量数据时考虑分批处理
4. **Padding处理**: 使用`input_mask`正确处理填充位置

### 7.2 调试技巧

```python
def debug_embedding_extraction(model, sample_data):
    """
    调试embedding提取过程
    """
    print(f"输入数据形状:")
    print(f"  input_ids: {sample_data['input_ids'].shape}")
    print(f"  input_mask: {sample_data['input_mask'].shape}")
    
    with torch.no_grad():
        encoder_output = model.model.MolE(
            input_ids=sample_data['input_ids'],
            input_mask=sample_data['input_mask'],
            output_all_encoded_layers=True
        )
        
        hidden_states = encoder_output["hidden_states"]
        print(f"Hidden states层数: {len(hidden_states)}")
        print(f"最终层形状: {hidden_states[-1].shape}")
        
        # 检查非padding位置
        valid_positions = sample_data['input_mask'].sum(dim=1)
        print(f"每个样本的有效位置数: {valid_positions}")
        
        return hidden_states[-1]
```

---

## 总结

通过以上方法，您可以成功从训练好的MolE模型中提取出每个原子的深度学习表示。这些768维的embedding向量包含了丰富的化学和结构信息，可用于各种下游的分子分析任务，如相似性搜索、聚类分析、原子环境研究等。

记住关键要点：
- 使用最后一层Transformer的输出（`hidden_states[-1]`）
- 跳过CLS token（位置0），从位置1开始提取原子embedding
- 注意处理padding和批处理
- 每个原子的embedding都是上下文化的，包含与整个分子的交互信息