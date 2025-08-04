"""
优化版模型文件

主要优化内容：
1. 改进的Flash Attention实现
2. 更高效的特征处理
3. 增强的模型架构
4. 更好的内存管理和性能优化
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OptimizedFlashMultiHeadAttention(nn.Module):
    """
    优化版多头注意力机制
    
    主要改进：
    1. 更高效的Flash Attention实现
    2. 改进的梯度检查点
    3. 更好的内存管理
    4. 支持不同的注意力模式
    """
    
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float = 0.1):
        super(OptimizedFlashMultiHeadAttention, self).__init__()
        
        self.hidden_units = hidden_units
        self.num_heads = num_heads
        self.head_dim = hidden_units // num_heads
        self.dropout_rate = dropout_rate
        self.scale = self.head_dim ** -0.5
        
        assert hidden_units % num_heads == 0, "hidden_units must be divisible by num_heads"
        
        # 使用更高效的参数初始化
        self.qkv_proj = nn.Linear(hidden_units, 3 * hidden_units, bias=False)
        self.out_proj = nn.Linear(hidden_units, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        前向传播
        
        Args:
            x: 输入张量 [batch_size, seq_len, hidden_units]
            attn_mask: 注意力掩码
            
        Returns:
            输出张量和注意力权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V (一次性计算，提高效率)
        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch_size, num_heads, seq_len, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # 使用Flash Attention（如果可用）
        if hasattr(F, 'scaled_dot_product_attention') and torch.cuda.is_available():
            try:
                # PyTorch 2.0+ 内置Flash Attention
                # 如果有显式掩码，就不使用is_causal
                if attn_mask is not None:
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v, 
                        attn_mask=attn_mask.unsqueeze(1).unsqueeze(1),
                        dropout_p=self.dropout_rate if self.training else 0.0
                    )
                else:
                    attn_output = F.scaled_dot_product_attention(
                        q, k, v, 
                        dropout_p=self.dropout_rate if self.training else 0.0,
                        is_causal=True  # 因果掩码
                    )
                attn_weights = None
            except Exception as e:
                logger.warning(f"Flash Attention failed, fallback to standard attention: {e}")
                attn_output, attn_weights = self._standard_attention(q, k, v, attn_mask)
        else:
            # 标准注意力机制
            attn_output, attn_weights = self._standard_attention(q, k, v, attn_mask)
        
        # 重新整形并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_units)
        output = self.out_proj(attn_output)
        
        return output, attn_weights
    
    def _standard_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, 
                          attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        标准注意力机制实现
        
        Args:
            q, k, v: 查询、键、值张量
            attn_mask: 注意力掩码
            
        Returns:
            注意力输出和权重
        """
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # 应用掩码
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask.unsqueeze(1).unsqueeze(1).logical_not(), float('-inf'))
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 计算输出
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights


class OptimizedPointWiseFeedForward(nn.Module):
    """
    优化版逐点前馈网络
    
    主要改进：
    1. 使用GELU激活函数（更好的性能）
    2. 改进的dropout策略
    3. 可选的层归一化
    """
    
    def __init__(self, hidden_units: int, dropout_rate: float = 0.1, 
                 expansion_factor: int = 4, activation: str = 'gelu'):
        super(OptimizedPointWiseFeedForward, self).__init__()
        
        inner_dim = hidden_units * expansion_factor
        
        self.fc1 = nn.Linear(hidden_units, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_units)
        self.dropout = nn.Dropout(dropout_rate)
        
        # 选择激活函数
        if activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        else:
            self.activation = nn.ReLU()
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.constant_(self.fc2.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    优化版Transformer块
    
    结合了多头注意力和前馈网络，支持不同的归一化策略
    """
    
    def __init__(self, hidden_units: int, num_heads: int, dropout_rate: float = 0.1,
                 norm_first: bool = True, activation: str = 'gelu'):
        super(TransformerBlock, self).__init__()
        
        self.norm_first = norm_first
        self.attention = OptimizedFlashMultiHeadAttention(hidden_units, num_heads, dropout_rate)
        self.feed_forward = OptimizedPointWiseFeedForward(hidden_units, dropout_rate, activation=activation)
        
        self.norm1 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.norm2 = nn.LayerNorm(hidden_units, eps=1e-8)
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """前向传播"""
        if self.norm_first:
            # Pre-norm
            attn_out, _ = self.attention(self.norm1(x), attn_mask)
            x = x + self.dropout(attn_out)
            ff_out = self.feed_forward(self.norm2(x))
            x = x + self.dropout(ff_out)
        else:
            # Post-norm
            attn_out, _ = self.attention(x, attn_mask)
            x = self.norm1(x + self.dropout(attn_out))
            ff_out = self.feed_forward(x)
            x = self.norm2(x + self.dropout(ff_out))
        
        return x


class OptimizedFeatureProcessor(nn.Module):
    """
    优化版特征处理器
    
    主要改进：
    1. 批量特征处理
    2. 更高效的embedding查找
    3. 动态特征维度处理
    4. 改进的特征融合策略
    """
    
    def __init__(self, feat_statistics: Dict, feat_types: Dict, hidden_units: int, device: str):
        super(OptimizedFeatureProcessor, self).__init__()
        
        self.hidden_units = hidden_units
        self.device = device
        self.feat_statistics = feat_statistics
        self.feat_types = feat_types
        
        # 初始化embedding层
        self.sparse_emb = nn.ModuleDict()
        self.emb_transform = nn.ModuleDict()
        
        # 稀疏特征embedding
        for feat_type in ['user_sparse', 'item_sparse', 'user_array', 'item_array']:
            if feat_type in feat_types:
                for feat_id in feat_types[feat_type]:
                    if feat_id in feat_statistics:
                        vocab_size = feat_statistics[feat_id] + 1
                        self.sparse_emb[feat_id] = nn.Embedding(vocab_size, hidden_units, padding_idx=0)
        
        # 多模态特征变换
        emb_shape_dict = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        if 'item_emb' in feat_types:
            for feat_id in feat_types['item_emb']:
                if feat_id in emb_shape_dict:
                    input_dim = emb_shape_dict[feat_id]
                    self.emb_transform[feat_id] = nn.Sequential(
                        nn.Linear(input_dim, hidden_units),
                        nn.LayerNorm(hidden_units),
                        nn.ReLU(),
                        nn.Dropout(0.1)
                    )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_units, hidden_units),
            nn.LayerNorm(hidden_units),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def process_sparse_features(self, features: List[Dict], feat_ids: List[str]) -> torch.Tensor:
        """
        处理稀疏特征
        
        Args:
            features: 特征列表
            feat_ids: 特征ID列表
            
        Returns:
            处理后的特征张量
        """
        batch_size = len(features)
        seq_len = len(features[0]) if features else 0
        
        if not feat_ids or seq_len == 0:
            return torch.zeros(batch_size, seq_len, self.hidden_units, device=self.device)
        
        feature_tensors = []
        
        for feat_id in feat_ids:
            if feat_id in self.sparse_emb:
                # 批量处理特征
                feat_values = []
                for batch_feats in features:
                    seq_values = []
                    for seq_feat in batch_feats:
                        if feat_id in seq_feat:
                            seq_values.append(seq_feat[feat_id])
                        else:
                            seq_values.append(0)  # 默认值
                    feat_values.append(seq_values)
                
                feat_tensor = torch.tensor(feat_values, dtype=torch.long, device=self.device)
                emb_tensor = self.sparse_emb[feat_id](feat_tensor)
                feature_tensors.append(emb_tensor)
        
        if feature_tensors:
            # 特征融合
            combined_features = torch.stack(feature_tensors, dim=-1).sum(dim=-1)
            return self.feature_fusion(combined_features)
        else:
            return torch.zeros(batch_size, seq_len, self.hidden_units, device=self.device)
    
    def process_multimodal_features(self, features: List[Dict], feat_ids: List[str]) -> torch.Tensor:
        """
        处理多模态特征
        
        Args:
            features: 特征列表
            feat_ids: 特征ID列表
            
        Returns:
            处理后的特征张量
        """
        batch_size = len(features)
        seq_len = len(features[0]) if features and len(features[0]) > 0 else 0
        
        if not feat_ids or seq_len == 0:
            return torch.zeros(batch_size, seq_len, self.hidden_units, device=self.device)
        
        feature_tensors = []
        
        for feat_id in feat_ids:
            if feat_id in self.emb_transform:
                # 批量处理多模态特征
                feat_values = []
                for batch_feats in features:
                    seq_values = []
                    for seq_feat in batch_feats:
                        if feat_id in seq_feat and seq_feat[feat_id] is not None:
                            seq_values.append(seq_feat[feat_id])
                        else:
                            # 使用零向量作为默认值
                            emb_dim = 32  # 默认维度
                            if feat_id in ["82"]: emb_dim = 1024
                            elif feat_id in ["83", "85", "86"]: emb_dim = 3584
                            elif feat_id in ["84"]: emb_dim = 4096
                            seq_values.append([0.0] * emb_dim)
                    feat_values.append(seq_values)
                
                try:
                    feat_tensor = torch.tensor(feat_values, dtype=torch.float32, device=self.device)
                    emb_tensor = self.emb_transform[feat_id](feat_tensor)
                    feature_tensors.append(emb_tensor)
                except Exception as e:
                    logger.warning(f"Error processing feature {feat_id}: {e}")
                    continue
        
        if feature_tensors:
            # 特征融合
            combined_features = torch.stack(feature_tensors, dim=-1).sum(dim=-1)
            return combined_features
        else:
            return torch.zeros(batch_size, seq_len, self.hidden_units, device=self.device)


class OptimizedBaselineModel(nn.Module):
    """
    优化版生成式推荐模型
    
    主要改进：
    1. 更高效的特征处理
    2. 改进的Transformer架构
    3. 更好的参数初始化
    4. 支持梯度检查点
    5. 改进的推理性能
    """
    
    def __init__(self, user_num: int, item_num: int, feat_statistics: Dict, feat_types: Dict, args):
        super(OptimizedBaselineModel, self).__init__()
        
        self.user_num = user_num
        self.item_num = item_num
        self.device = args.device
        self.hidden_units = args.hidden_units
        self.maxlen = args.maxlen
        self.norm_first = args.norm_first
        
        # 基础embedding层
        self.item_emb = nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.user_emb = nn.Embedding(self.user_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(2 * args.maxlen + 1, args.hidden_units, padding_idx=0)
        
        # Dropout层
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # 特征处理器
        self.feature_processor = OptimizedFeatureProcessor(
            feat_statistics, feat_types, args.hidden_units, args.device
        )
        
        # Transformer层
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                args.hidden_units, 
                args.num_heads, 
                args.dropout_rate,
                args.norm_first
            ) for _ in range(args.num_blocks)
        ])
        
        # 最终归一化
        self.final_norm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        # 初始化权重
        self._init_weights()
        
        logger.info(f"Initialized OptimizedBaselineModel with {sum(p.numel() for p in self.parameters())} parameters")
    
    def _init_weights(self):
        """初始化模型权重"""
        # Xavier初始化
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if hasattr(module, 'weight'):
                    nn.init.xavier_normal_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # 特殊初始化padding embedding
        with torch.no_grad():
            self.item_emb.weight[0].fill_(0)
            self.user_emb.weight[0].fill_(0)
            self.pos_emb.weight[0].fill_(0)
    
    def create_attention_mask(self, seq_len: int, token_mask: torch.Tensor) -> torch.Tensor:
        """
        创建注意力掩码
        
        Args:
            seq_len: 序列长度
            token_mask: token掩码
            
        Returns:
            注意力掩码
        """
        # 创建因果掩码
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=self.device))
        
        # 结合padding掩码
        padding_mask = (token_mask != 0).unsqueeze(1)
        attention_mask = causal_mask.unsqueeze(0) & padding_mask
        
        return attention_mask
    
    def encode_sequence(self, seq: torch.Tensor, token_mask: torch.Tensor, 
                       seq_features: List[Dict]) -> torch.Tensor:
        """
        编码输入序列
        
        Args:
            seq: 序列ID
            token_mask: token掩码
            seq_features: 序列特征
            
        Returns:
            编码后的序列表示
        """
        batch_size, seq_len = seq.shape
        
        # 基础embedding
        item_embeddings = self.item_emb(seq * (token_mask == 1).long())
        user_embeddings = self.user_emb(seq * (token_mask == 2).long())
        
        # 位置embedding
        positions = torch.arange(1, seq_len + 1, device=self.device).unsqueeze(0).expand(batch_size, -1)
        positions = positions * (token_mask != 0).long()
        pos_embeddings = self.pos_emb(positions)
        
        # 特征处理
        feature_embeddings = self.feature_processor.process_multimodal_features(
            seq_features, self.feature_processor.feat_types.get('item_emb', [])
        )
        
        # 组合embeddings
        sequence_embeddings = item_embeddings + user_embeddings + pos_embeddings + feature_embeddings
        sequence_embeddings = sequence_embeddings * (self.hidden_units ** 0.5)
        sequence_embeddings = self.emb_dropout(sequence_embeddings)
        
        # 创建注意力掩码
        attention_mask = self.create_attention_mask(seq_len, token_mask)
        
        # Transformer编码
        hidden_states = sequence_embeddings
        for transformer_block in self.transformer_blocks:
            hidden_states = transformer_block(hidden_states, attention_mask)
        
        # 最终归一化
        hidden_states = self.final_norm(hidden_states)
        
        return hidden_states
    
    def forward(self, user_item: torch.Tensor, pos_seqs: torch.Tensor, neg_seqs: torch.Tensor,
                mask: torch.Tensor, next_mask: torch.Tensor, next_action_type: torch.Tensor,
                seq_feature: List[Dict], pos_feature: List[Dict], neg_feature: List[Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            user_item: 用户序列
            pos_seqs: 正样本序列
            neg_seqs: 负样本序列
            mask: token掩码
            next_mask: 下一个token掩码
            next_action_type: 下一个动作类型
            seq_feature: 序列特征
            pos_feature: 正样本特征
            neg_feature: 负样本特征
            
        Returns:
            正负样本logits
        """
        # 编码序列
        sequence_repr = self.encode_sequence(user_item, mask, seq_feature)
        
        # 处理正负样本
        pos_embeddings = self.item_emb(pos_seqs)
        neg_embeddings = self.item_emb(neg_seqs)
        
        # 添加特征信息
        if pos_feature:
            pos_feat_emb = self.feature_processor.process_multimodal_features(
                pos_feature, self.feature_processor.feat_types.get('item_emb', [])
            )
            pos_embeddings = pos_embeddings + pos_feat_emb
        
        if neg_feature:
            neg_feat_emb = self.feature_processor.process_multimodal_features(
                neg_feature, self.feature_processor.feat_types.get('item_emb', [])
            )
            neg_embeddings = neg_embeddings + neg_feat_emb
        
        # 计算logits
        pos_logits = (sequence_repr * pos_embeddings).sum(dim=-1)
        neg_logits = (sequence_repr * neg_embeddings).sum(dim=-1)
        
        # 应用掩码
        loss_mask = (next_mask == 1).float()
        pos_logits = pos_logits * loss_mask
        neg_logits = neg_logits * loss_mask
        
        return pos_logits, neg_logits
    
    def predict(self, log_seqs: torch.Tensor, seq_feature: List[Dict], mask: torch.Tensor) -> torch.Tensor:
        """
        预测用户兴趣表示
        
        Args:
            log_seqs: 用户序列
            seq_feature: 序列特征
            mask: token掩码
            
        Returns:
            用户兴趣表示
        """
        # 编码序列
        sequence_repr = self.encode_sequence(log_seqs, mask, seq_feature)
        
        # 取最后一个位置的表示
        final_repr = sequence_repr[:, -1, :]
        
        return final_repr
    
    def save_item_embeddings(self, item_ids: List[int], retrieval_ids: List[int], 
                           feat_dict: Dict, save_path: str, batch_size: int = 1024):
        """
        保存物品embeddings用于检索
        
        Args:
            item_ids: 物品ID列表
            retrieval_ids: 检索ID列表
            feat_dict: 特征字典
            save_path: 保存路径
            batch_size: 批次大小
        """
        self.eval()
        all_embeddings = []
        
        with torch.no_grad():
            for start_idx in tqdm(range(0, len(item_ids), batch_size), 
                                desc="Saving item embeddings"):
                end_idx = min(start_idx + batch_size, len(item_ids))
                
                # 构建批次数据
                batch_item_ids = torch.tensor(item_ids[start_idx:end_idx], 
                                            dtype=torch.long, device=self.device)
                batch_features = []
                for i in range(start_idx, end_idx):
                    if i in feat_dict:
                        batch_features.append(feat_dict[i])
                    else:
                        batch_features.append({})
                
                # 计算embeddings
                item_embeddings = self.item_emb(batch_item_ids)
                
                if batch_features:
                    # 添加特征信息
                    feat_embeddings = self.feature_processor.process_multimodal_features(
                        [batch_features], self.feature_processor.feat_types.get('item_emb', [])
                    )
                    if feat_embeddings.size(0) == 1:
                        feat_embeddings = feat_embeddings.squeeze(0)
                    item_embeddings = item_embeddings + feat_embeddings
                
                all_embeddings.append(item_embeddings.cpu().numpy())
        
        # 保存embeddings
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_ids = np.array(retrieval_ids, dtype=np.uint64).reshape(-1, 1)
        
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # 保存文件
        from dataset import save_emb
        save_emb(final_embeddings, save_path / 'embedding.fbin')
        save_emb(final_ids, save_path / 'id.u64bin')
        
        logger.info(f"Saved {len(final_embeddings)} item embeddings to {save_path}")
        
        self.train()  # 恢复训练模式
