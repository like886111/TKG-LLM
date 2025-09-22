import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.data import Data, Batch

class TimeSeriesKnowledgeGraph:
    """
    时间序列知识图谱构建类
    
    该类用于从时间序列数据构建知识图谱，包括：
    1. 时间点节点：表示不同的时间点
    2. 特征节点：表示不同的特征
    3. 关系边：表示时间点之间的时序关系、特征之间的相关性等
    """
    def __init__(self, seq_len, num_features, window_size=5):
        """
        初始化知识图谱构建器
        
        参数:
            seq_len (int): 序列长度
            num_features (int): 特征数量
            window_size (int): 构建时序关系的窗口大小
        """
        self.seq_len = seq_len
        self.num_features = num_features
        self.window_size = window_size
        
    def build_graph(self, data):
        """
        从时间序列数据构建知识图谱
        
        参数:
            data (torch.Tensor): 形状为 [batch_size, seq_len, num_features] 的时间序列数据
            
        返回:
            torch_geometric.data.Data: 构建的知识图谱
        """
        batch_size = data.shape[0]
        device = data.device  # 获取输入数据的设备
        graphs = []
        
        for b in range(batch_size):
            # 1. 创建节点
            num_nodes = self.seq_len * self.num_features
            x = data[b].reshape(-1, 1)  # [seq_len * num_features, 1]
            
            # 2. 创建边索引和边属性
            edge_index = []
            edge_attr = []
            
            # 2.1 添加时序关系边
            for t in range(self.seq_len - 1):
                for f in range(self.num_features):
                    # 当前节点索引
                    curr_idx = t * self.num_features + f
                    # 下一个时间点的相同特征节点索引
                    next_idx = (t + 1) * self.num_features + f
                    # 添加有向边
                    edge_index.append([curr_idx, next_idx])
                    edge_attr.append(1.0)  # 时序关系权重
            
            # 2.2 添加特征相关性边
            for t in range(self.seq_len):
                for f1 in range(self.num_features):
                    for f2 in range(f1 + 1, self.num_features):
                        # 计算特征相关性
                        idx1 = t * self.num_features + f1
                        idx2 = t * self.num_features + f2
                        corr = torch.corrcoef(torch.stack([x[idx1], x[idx2]]))[0, 1]
                        if not torch.isnan(corr):
                            # 添加无向边
                            edge_index.append([idx1, idx2])
                            edge_index.append([idx2, idx1])
                            edge_attr.extend([corr, corr])
            
            # 3. 转换为PyTorch Geometric格式
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().to(device)  # 确保在GPU上
            edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1).to(device)  # 确保在GPU上
            
            # 4. 创建图数据对象
            graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            graphs.append(graph)
        
        # 5. 将多个图批处理成一个批次
        return Batch.from_data_list(graphs)

class GraphEncoder(nn.Module):
    """
    图神经网络编码器
    
    使用图卷积网络对知识图谱进行编码，生成节点嵌入向量
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        """
        初始化图编码器
        
        参数:
            input_dim (int): 输入特征维度
            hidden_dim (int): 隐藏层维度
            output_dim (int): 输出嵌入维度
            num_layers (int): 图卷积层数
        """
        super().__init__()
        
        self.layers = nn.ModuleList()
        # 第一层
        self.layers.append(GCNConv(input_dim, hidden_dim))
        # 中间层
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        # 最后一层
        self.layers.append(GCNConv(hidden_dim, output_dim))
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, data):
        """
        前向传播
        
        参数:
            data (torch_geometric.data.Data): 输入图数据
            
        返回:
            torch.Tensor: 节点嵌入向量
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # 确保所有张量都在GPU上
        device = next(self.parameters()).device
        x = x.to(device)
        edge_index = edge_index.to(device)
        edge_attr = edge_attr.to(device)
        
        # 应用图卷积层
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)
            x = F.dropout(x, p=0.2, training=self.training)
        
        # 层归一化
        x = self.layer_norm(x)
        
        return x

class ConcatFusion(nn.Module):
    """
    简单拼接融合层
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x_embed, node_embeddings):
        combined_embeddings = torch.cat([x_embed, node_embeddings], dim=-1)
        return self.fusion_layer(combined_embeddings)

class AttentionFusion(nn.Module):
    """
    注意力融合层
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = 8
        self.head_dim = embed_dim // self.num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads=self.num_heads, batch_first=True)
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x_embed, node_embeddings):
        # 打印形状以便调试
        
        # 确保输入形状兼容
        if x_embed.shape[0] != node_embeddings.shape[0]:
            # 如果批次大小不同，将node_embeddings调整为与x_embed相同的批次大小
            if node_embeddings.shape[0] < x_embed.shape[0]:
                # 如果node_embeddings的批次大小小于x_embed，则重复node_embeddings
                repeat_times = x_embed.shape[0] // node_embeddings.shape[0] + 1
                node_embeddings = node_embeddings.repeat(repeat_times, 1, 1)
                node_embeddings = node_embeddings[:x_embed.shape[0]]
            else:
                # 如果node_embeddings的批次大小大于x_embed，则截取node_embeddings
                node_embeddings = node_embeddings[:x_embed.shape[0]]
        
        # 确保序列长度相同
        if x_embed.shape[1] != node_embeddings.shape[1]:
            # 如果序列长度不同，将node_embeddings调整为与x_embed相同的序列长度
            if node_embeddings.shape[1] > x_embed.shape[1]:
                node_embeddings = node_embeddings[:, :x_embed.shape[1], :]
            else:
                # 使用重复来扩展序列长度
                repeat_times = x_embed.shape[1] // node_embeddings.shape[1] + 1
                node_embeddings = node_embeddings.repeat(1, repeat_times, 1)
                node_embeddings = node_embeddings[:, :x_embed.shape[1], :]
        
        # 确保嵌入维度相同
        if x_embed.shape[-1] != node_embeddings.shape[-1]:
            # 如果嵌入维度不同，使用线性层调整node_embeddings的维度
            linear = nn.Linear(node_embeddings.shape[-1], x_embed.shape[-1]).to(node_embeddings.device)
            node_embeddings = linear(node_embeddings)
        
        # 使用简单的加权平均
        attn_output = x_embed + 0.5 * node_embeddings
        
        return self.fusion_layer(attn_output)

class GateFusion(nn.Module):
    """
    门控融合层
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Sigmoid()
        )
        self.fusion_layer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x_embed, node_embeddings):
        # 确保输入形状兼容
        if x_embed.shape[0] != node_embeddings.shape[0]:
            node_embeddings = node_embeddings[:x_embed.shape[0]]
        
        if x_embed.shape[1] != node_embeddings.shape[1]:
            if node_embeddings.shape[1] > x_embed.shape[1]:
                node_embeddings = node_embeddings[:, :x_embed.shape[1], :]
            else:
                repeat_times = x_embed.shape[1] // node_embeddings.shape[1] + 1
                node_embeddings = node_embeddings.repeat(1, repeat_times, 1)
                node_embeddings = node_embeddings[:, :x_embed.shape[1], :]
        
        # 确保嵌入维度相同
        if x_embed.shape[-1] != node_embeddings.shape[-1]:
            # 如果嵌入维度不同，使用线性层调整node_embeddings的维度
            linear = nn.Linear(node_embeddings.shape[-1], x_embed.shape[-1]).to(node_embeddings.device)
            node_embeddings = linear(node_embeddings)
        
        # 计算门控值
        gate_values = self.gate(torch.cat([x_embed, node_embeddings], dim=-1))
        # 门控融合
        fused = gate_values * x_embed + (1 - gate_values) * node_embeddings
        return self.fusion_layer(fused)

class KnowledgeGraphEnhancedPrompt(nn.Module):
    """
    知识图谱增强的提示学习模块
    
    将图神经网络生成的节点嵌入与时间序列嵌入结合，生成增强型提示向量
    """
    def __init__(self, seq_len, num_features, embed_dim=768, hidden_dim=256, fusion_type='concat'):
        """
        初始化知识图谱增强的提示学习模块
        
        参数:
            seq_len (int): 序列长度
            num_features (int): 特征数量
            embed_dim (int): 嵌入维度
            hidden_dim (int): 图神经网络隐藏层维度
            fusion_type (str): 特征融合方式，可选：concat, attention, gate
        """
        super().__init__()
        
        # 知识图谱构建器
        self.kg_builder = TimeSeriesKnowledgeGraph(seq_len, num_features)
        
        # 图编码器
        self.graph_encoder = GraphEncoder(
            input_dim=1,  # 时间序列数据是1维的
            hidden_dim=hidden_dim,
            output_dim=embed_dim
        )
        
        # 特征融合层
        if fusion_type == 'concat':
            self.fusion = ConcatFusion(embed_dim)
        elif fusion_type == 'attention':
            self.fusion = AttentionFusion(embed_dim)
        elif fusion_type == 'gate':
            self.fusion = GateFusion(embed_dim)
        else:
            raise ValueError(f"不支持的融合方式: {fusion_type}")
        
    def forward(self, x_embed, data):
        """
        前向传播
        
        参数:
            x_embed (torch.Tensor): 时间序列嵌入 [batch_size, seq_len, embed_dim]
            data (torch.Tensor): 原始时间序列数据 [batch_size, seq_len, num_features]
            
        返回:
            torch.Tensor: 增强型提示向量
        """
        # 确保数据在GPU上
        device = next(self.parameters()).device
        data = data.to(device)
        x_embed = x_embed.to(device)
        
        # 1. 构建知识图谱
        graph_data = self.kg_builder.build_graph(data)
        
        # 2. 使用图编码器生成节点嵌入
        node_embeddings = self.graph_encoder(graph_data)
        
        # 3. 重塑节点嵌入以匹配时间序列嵌入的形状
        batch_size = data.shape[0]
        # 计算每个批次中的节点数量
        nodes_per_batch = node_embeddings.shape[0] // batch_size
        
        # 重塑节点嵌入，确保与x_embed的形状兼容
        node_embeddings = node_embeddings.view(batch_size, nodes_per_batch, node_embeddings.size(-1))
        
        # 4. 融合时间序列嵌入和图节点嵌入
        enhanced_embeddings = self.fusion(x_embed, node_embeddings)
        
        return enhanced_embeddings 