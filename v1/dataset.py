"""
优化版数据集处理模块

主要优化内容：
1. 改进数据加载和预处理效率
2. 优化内存使用和缓存策略
3. 增强错误处理和数据验证
4. 改进多模态特征处理
"""

import json
import pickle
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

import numpy as np
import torch
from tqdm import tqdm


# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def save_emb(arr: np.ndarray, path: Path) -> None:
    """
    保存embedding数组到二进制文件
    
    Args:
        arr: 要保存的numpy数组
        path: 保存路径
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'wb') as f:
            # 写入数组维度信息
            f.write(struct.pack('I', len(arr.shape)))
            for dim in arr.shape:
                f.write(struct.pack('I', dim))
            # 写入数据类型信息
            dtype_str = str(arr.dtype).encode('utf-8')
            f.write(struct.pack('I', len(dtype_str)))
            f.write(dtype_str)
            # 写入数组数据
            arr.tobytes()
            f.write(arr.tobytes())
        logger.info(f"Successfully saved embedding to {path}")
    except Exception as e:
        logger.error(f"Error saving embedding to {path}: {e}")
        raise


def load_mm_emb(data_dir: Path, mm_ids: List[str]) -> Dict[str, np.ndarray]:
    """
    加载多模态特征embedding
    
    Args:
        data_dir: 数据目录
        mm_ids: 多模态特征ID列表
        
    Returns:
        多模态特征字典
    """
    mm_emb_dict = {}
    
    for mm_id in mm_ids:
        emb_path = data_dir / f"{mm_id}.npy"
        try:
            if emb_path.exists():
                mm_emb_dict[mm_id] = np.load(emb_path, allow_pickle=True).item()
                logger.info(f"Loaded multimodal embedding {mm_id} from {emb_path}")
            else:
                logger.warning(f"Multimodal embedding file not found: {emb_path}")
                mm_emb_dict[mm_id] = {}
        except Exception as e:
            logger.error(f"Error loading multimodal embedding {mm_id}: {e}")
            mm_emb_dict[mm_id] = {}
    
    return mm_emb_dict


class OptimizedDataset(torch.utils.data.Dataset):
    """
    优化版用户序列数据集
    
    主要优化：
    1. 改进数据加载缓存策略
    2. 优化特征处理流程
    3. 增强错误处理和数据验证
    4. 支持更灵活的特征配置
    """

    def __init__(self, data_dir: str, args):
        """初始化数据集"""
        super().__init__()
        self.data_dir = Path(data_dir)
        self.maxlen = args.maxlen
        self.mm_emb_ids = args.mm_emb_id
        self.device = args.device
        
        # 验证数据目录
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")
        
        self._load_data_and_offsets()
        self._load_metadata()
        self._init_feature_info()
        
        logger.info(f"Dataset initialized with {len(self.seq_offsets)} users, "
                   f"{self.itemnum} items, {self.usernum} users")

    def _load_data_and_offsets(self) -> None:
        """加载用户序列数据和偏移量"""
        seq_file = self.data_dir / "seq.jsonl"
        offset_file = self.data_dir / "seq_offsets.pkl"
        
        if not seq_file.exists():
            raise FileNotFoundError(f"Sequence file not found: {seq_file}")
        if not offset_file.exists():
            raise FileNotFoundError(f"Offset file not found: {offset_file}")
        
        try:
            self.data_file = open(seq_file, 'rb')
            with open(offset_file, 'rb') as f:
                self.seq_offsets = pickle.load(f)
            logger.info(f"Loaded {len(self.seq_offsets)} user sequences")
        except Exception as e:
            logger.error(f"Error loading data files: {e}")
            raise

    def _load_metadata(self) -> None:
        """加载元数据和特征信息"""
        # 加载物品特征字典
        item_feat_file = self.data_dir / "item_feat_dict.json"
        if item_feat_file.exists():
            try:
                with open(item_feat_file, 'r') as f:
                    self.item_feat_dict = json.load(f)
                logger.info(f"Loaded item features for {len(self.item_feat_dict)} items")
            except Exception as e:
                logger.error(f"Error loading item features: {e}")
                self.item_feat_dict = {}
        else:
            logger.warning(f"Item feature file not found: {item_feat_file}")
            self.item_feat_dict = {}
        
        # 加载多模态特征
        mm_emb_dir = self.data_dir / "creative_emb"
        if mm_emb_dir.exists():
            self.mm_emb_dict = load_mm_emb(mm_emb_dir, self.mm_emb_ids)
        else:
            logger.warning(f"Multimodal embedding directory not found: {mm_emb_dir}")
            self.mm_emb_dict = {mm_id: {} for mm_id in self.mm_emb_ids}
        
        # 加载索引器
        indexer_file = self.data_dir / 'indexer.pkl'
        if indexer_file.exists():
            try:
                with open(indexer_file, 'rb') as f:
                    indexer = pickle.load(f)
                    self.itemnum = len(indexer['i'])
                    self.usernum = len(indexer['u'])
                    self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
                    self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
                    self.indexer = indexer
                logger.info(f"Loaded indexer: {self.itemnum} items, {self.usernum} users")
            except Exception as e:
                logger.error(f"Error loading indexer: {e}")
                raise
        else:
            raise FileNotFoundError(f"Indexer file not found: {indexer_file}")

    def _init_feature_info(self) -> None:
        """初始化特征信息"""
        # 默认特征配置
        default_features = {
            'user_sparse': [],
            'user_continual': [],
            'user_array': [],
            'item_sparse': [],
            'item_continual': [],
            'item_array': [],
            'item_emb': self.mm_emb_ids
        }
        
        # 特征默认值配置
        self.feature_default_value = {
            'sparse': 0,
            'continual': 0.0,
            'array': [],
            'emb': None
        }
        
        # 特征类型配置
        self.feature_types = default_features
        
        # 特征统计信息
        self.feat_statistics = {}
        
        # 为多模态特征设置维度
        emb_shape_dict = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
        for mm_id in self.mm_emb_ids:
            if mm_id in emb_shape_dict:
                self.feat_statistics[mm_id] = emb_shape_dict[mm_id]

    def _load_user_data(self, uid: int) -> List:
        """
        加载单个用户的数据
        
        Args:
            uid: 用户ID (reid)
            
        Returns:
            用户序列数据
        """
        try:
            self.data_file.seek(self.seq_offsets[uid])
            line = self.data_file.readline()
            data = json.loads(line)
            return data
        except Exception as e:
            logger.error(f"Error loading user data for uid {uid}: {e}")
            return []

    def _random_neq(self, l: int, r: int, s: set) -> int:
        """
        生成不在集合s中的随机数
        
        Args:
            l: 范围下界
            r: 范围上界
            s: 排除的数字集合
            
        Returns:
            随机数
        """
        t = np.random.randint(l, r)
        while t in s:
            t = np.random.randint(l, r)
        return t

    def _pad_sequence(self, seq: List, maxlen: int, pad_value: int = 0) -> List:
        """
        序列填充到指定长度
        
        Args:
            seq: 输入序列
            maxlen: 最大长度
            pad_value: 填充值
            
        Returns:
            填充后的序列
        """
        if len(seq) >= maxlen:
            return seq[-maxlen:]
        else:
            return [pad_value] * (maxlen - len(seq)) + seq

    def _process_features(self, seq_data: List, feature_types: List[str]) -> Tuple[List, List]:
        """
        处理序列特征
        
        Args:
            seq_data: 序列数据
            feature_types: 特征类型列表
            
        Returns:
            处理后的特征数据
        """
        seq_features = []
        item_ids = []
        
        for item_data in seq_data:
            if len(item_data) >= 6:  # 确保数据格式正确
                user_id, item_id, user_feat, item_feat, action_type, timestamp = item_data
                
                # 构建特征字典
                feat_dict = {}
                
                # 添加基础特征
                if isinstance(item_feat, dict):
                    feat_dict.update(item_feat)
                
                # 添加多模态特征
                for mm_id in self.mm_emb_ids:
                    if mm_id in self.mm_emb_dict and str(item_id) in self.mm_emb_dict[mm_id]:
                        feat_dict[mm_id] = self.mm_emb_dict[mm_id][str(item_id)]
                
                seq_features.append(feat_dict)
                item_ids.append(item_id)
        
        return seq_features, item_ids

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.seq_offsets)

    def __getitem__(self, idx: int) -> Tuple:
        """
        获取单个数据样本
        
        Args:
            idx: 样本索引
            
        Returns:
            数据样本元组
        """
        uid = list(self.seq_offsets.keys())[idx]
        user_data = self._load_user_data(uid)
        
        if not user_data:
            # 返回空样本
            empty_seq = [0] * self.maxlen
            empty_features = [{}] * self.maxlen
            return (torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    empty_features, empty_features, empty_features)
        
        # 处理序列数据
        seq_features, item_ids = self._process_features(user_data, self.feature_types['item_emb'])
        
        # 构建训练样本
        seq = self._pad_sequence(item_ids, self.maxlen)
        pos_seq = seq[1:] + [0]  # 正样本：下一个item
        
        # 负采样
        neg_seq = []
        for _ in range(len(seq)):
            neg_item = self._random_neq(1, self.itemnum + 1, set(item_ids))
            neg_seq.append(neg_item)
        
        # 构建token类型掩码（1表示item，2表示user）
        token_types = [1] * self.maxlen  # 这里简化为都是item
        next_token_types = [1] * self.maxlen
        next_action_types = [1] * self.maxlen  # 1表示点击
        
        # 填充特征
        padded_features = seq_features + [{}] * (self.maxlen - len(seq_features))
        pos_features = padded_features[1:] + [{}]
        neg_features = [{}] * self.maxlen  # 负样本特征为空
        
        return (torch.tensor(seq, dtype=torch.long),
                torch.tensor(pos_seq, dtype=torch.long),
                torch.tensor(neg_seq, dtype=torch.long),
                torch.tensor(token_types, dtype=torch.long),
                torch.tensor(next_token_types, dtype=torch.long),
                torch.tensor(next_action_types, dtype=torch.long),
                padded_features, pos_features, neg_features)

    def collate_fn(self, batch: List) -> Tuple:
        """
        批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            批处理后的数据
        """
        if not batch:
            return tuple()
        
        # 分离各个组件
        seqs, pos_seqs, neg_seqs, token_types, next_token_types, next_action_types, seq_feats, pos_feats, neg_feats = zip(*batch)
        
        # 堆叠张量
        batch_seqs = torch.stack(seqs)
        batch_pos_seqs = torch.stack(pos_seqs)
        batch_neg_seqs = torch.stack(neg_seqs)
        batch_token_types = torch.stack(token_types)
        batch_next_token_types = torch.stack(next_token_types)
        batch_next_action_types = torch.stack(next_action_types)
        
        return (batch_seqs, batch_pos_seqs, batch_neg_seqs, 
                batch_token_types, batch_next_token_types, batch_next_action_types,
                list(seq_feats), list(pos_feats), list(neg_feats))

    def __del__(self):
        """析构函数，关闭文件"""
        if hasattr(self, 'data_file') and self.data_file:
            try:
                self.data_file.close()
            except:
                pass


class OptimizedTestDataset(OptimizedDataset):
    """
    优化版测试数据集
    
    继承自OptimizedDataset，专门用于测试和推理
    """
    
    def __init__(self, data_dir: str, args, test_mode: bool = True):
        """
        初始化测试数据集
        
        Args:
            data_dir: 数据目录
            args: 参数对象
            test_mode: 是否为测试模式
        """
        super().__init__(data_dir, args)
        self.test_mode = test_mode
        
        if test_mode:
            # 测试模式下的特殊处理
            logger.info("Initialized test dataset in test mode")
    
    def __getitem__(self, idx: int) -> Tuple:
        """
        获取测试样本（不包含负采样）
        
        Args:
            idx: 样本索引
            
        Returns:
            测试样本元组
        """
        uid = list(self.seq_offsets.keys())[idx]
        user_data = self._load_user_data(uid)
        
        if not user_data:
            # 返回空样本
            empty_seq = [0] * self.maxlen
            empty_features = [{}] * self.maxlen
            return (torch.zeros(self.maxlen, dtype=torch.long),
                    torch.zeros(self.maxlen, dtype=torch.long),
                    empty_features)
        
        # 处理序列数据
        seq_features, item_ids = self._process_features(user_data, self.feature_types['item_emb'])
        
        # 构建测试样本
        seq = self._pad_sequence(item_ids, self.maxlen)
        token_types = [1] * self.maxlen  # 简化为都是item
        
        # 填充特征
        padded_features = seq_features + [{}] * (self.maxlen - len(seq_features))
        
        return (torch.tensor(seq, dtype=torch.long),
                torch.tensor(token_types, dtype=torch.long),
                padded_features)
    
    def collate_fn(self, batch: List) -> Tuple:
        """
        测试批处理函数
        
        Args:
            batch: 批次数据
            
        Returns:
            批处理后的数据
        """
        if not batch:
            return tuple()
        
        # 分离各个组件
        seqs, token_types, seq_feats = zip(*batch)
        
        # 堆叠张量
        batch_seqs = torch.stack(seqs)
        batch_token_types = torch.stack(token_types)
        
        return (batch_seqs, batch_token_types, list(seq_feats))
