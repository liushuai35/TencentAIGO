"""
该文件定义了用于加载和处理推荐系统所需的数据集。

主要包含三个部分：
1.  `MyDataset` 类：用于训练时加载用户序列数据、物品特征和多模态特征，并进行负采样和padding。
2.  `MyTestDataset` 类：继承自 `MyDataset`，用于测试时加载数据，并处理冷启动问题。
3.  辅助函数 `save_emb` 和 `load_mm_emb`：分别用于保存和加载多模态特征的Embedding。
"""

import json
import pickle
import struct
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm


class MyDataset(torch.utils.data.Dataset):
    """
    用户序列数据集

    Args:
        data_dir: 数据文件目录
        args: 全局参数

    Attributes:
        data_dir: 数据文件目录
        maxlen: 最大长度
        item_feat_dict: 物品特征字典
        mm_emb_ids: 激活的mm_emb特征ID
        mm_emb_dict: 多模态特征字典
        itemnum: 物品数量
        usernum: 用户数量
        indexer_i_rev: 物品索引字典 (reid -> item_id)
        indexer_u_rev: 用户索引字典 (reid -> user_id)
        indexer: 索引字典
        feature_default_value: 特征缺省值
        feature_types: 特征类型，分为user和item的sparse, array, emb, continual类型
        feat_statistics: 特征统计信息，包括user和item的特征数量
    """

    def __init__(self, data_dir, args):
        """
        初始化数据集
        """
        super().__init__()  # 调用父类torch.utils.data.Dataset的初始化方法
        self.data_dir = Path(data_dir)  # 将字符串路径转换为Path对象，方便路径操作
        self._load_data_and_offsets()  # 调用内部方法加载用户序列数据和文件偏移量
        self.maxlen = args.maxlen  # 从参数中获取序列的最大长度
        self.mm_emb_ids = args.mm_emb_id  # 从参数中获取需要加载的多模态特征ID列表

        # 加载物品特征字典，该文件存储了每个物品的特征信息
        self.item_feat_dict = json.load(open(Path(data_dir, "item_feat_dict.json"), 'r'))
        # 调用辅助函数加载指定ID的多模态特征Embedding
        self.mm_emb_dict = load_mm_emb(Path(data_dir, "creative_emb"), self.mm_emb_ids)
        # 使用'rb'模式（二进制读取）打开索引文件
        with open(self.data_dir / 'indexer.pkl', 'rb') as ff:
            indexer = pickle.load(ff)  # 从pickle文件中加载索引器对象
            self.itemnum = len(indexer['i'])  # 获取物品（item）的总数
            self.usernum = len(indexer['u'])  # 获取用户（user）的总数
        # 创建一个从reid到原始物品ID的反向映射字典
        self.indexer_i_rev = {v: k for k, v in indexer['i'].items()}
        # 创建一个从reid到原始用户ID的反向映射字典
        self.indexer_u_rev = {v: k for k, v in indexer['u'].items()}
        self.indexer = indexer  # 保存整个索引器对象

        # 调用内部方法初始化特征的默认值、类型和统计信息
        self.feature_default_value, self.feature_types, self.feat_statistics = self._init_feat_info()

    def _load_data_and_offsets(self):
        """
        加载用户序列数据和每一行的文件偏移量(预处理好的), 用于快速随机访问数据并I/O
        """
        # 以二进制读取模式打开存储用户序列的jsonl文件
        self.data_file = open(self.data_dir / "seq.jsonl", 'rb')
        # 以二进制读取模式打开存储偏移量的pickle文件
        with open(Path(self.data_dir, 'seq_offsets.pkl'), 'rb') as f:
            # 加载偏移量数据，self.seq_offsets是一个列表或字典，存储每个用户数据在文件中的起始位置
            self.seq_offsets = pickle.load(f)

    def _load_user_data(self, uid):
        """
        从数据文件中加载单个用户的数据

        Args:
            uid: 用户ID(reid)

        Returns:
            data: 用户序列数据，格式为[(user_id, item_id, user_feat, item_feat, action_type, timestamp)]
        """
        self.data_file.seek(self.seq_offsets[uid])  # 将文件读取指针移动到指定用户数据行的起始位置
        line = self.data_file.readline()  # 读取一行数据
        data = json.loads(line)  # 将读取到的JSON字符串解析为Python对象
        return data  # 返回解析后的用户数据

    def _random_neq(self, l, r, s):
        """
        生成一个不在序列s中的随机整数, 用于训练时的负采样

        Args:
            l: 随机整数的最小值
            r: 随机整数的最大值
            s: 序列

        Returns:
            t: 不在序列s中的随机整数
        """
        t = np.random.randint(l, r)  # 在[l, r)范围内生成一个随机整数
        # 循环直到生成的随机数t既不在集合s中，也不在item_feat_dict的键中（确保item有特征）
        while t in s or str(t) not in self.item_feat_dict:
            t = np.random.randint(l, r)  # 重新生成随机数
        return t  # 返回满足条件的随机数

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户ID(reid)

        Returns:
            seq: 用户序列ID
            pos: 正样本ID（即下一个真实访问的item）
            neg: 负样本ID
            token_type: 用户序列类型，1表示item，2表示user
            next_token_type: 下一个token类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            pos_feat: 正样本特征，每个元素为字典，key为特征ID，value为特征值
            neg_feat: 负样本特征，每个元素为字典，key为特征ID，value为特征值
        """
        user_sequence = self._load_user_data(uid)  # 动态加载指定用户的序列数据

        ext_user_sequence = []  # 初始化一个扩展用户序列列表，用于统一处理用户和物品事件
        for record_tuple in user_sequence:  # 遍历原始用户序列中的每个记录
            u, i, user_feat, item_feat, action_type, _ = record_tuple  # 解包记录元组
            if u and user_feat:  # 如果存在用户ID和用户特征
                # 将用户事件（ID，特征，类型2，行为类型）插入到扩展序列的开头
                ext_user_sequence.insert(0, (u, user_feat, 2, action_type))
            if i and item_feat:  # 如果存在物品ID和物品特征
                # 将物品事件（ID，特征，类型1，行为类型）追加到扩展序列的末尾
                ext_user_sequence.append((i, item_feat, 1, action_type))

        # 初始化用于存储模型输入的numpy数组，长度为maxlen+1
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)  # 序列ID数组
        pos = np.zeros([self.maxlen + 1], dtype=np.int32)  # 正样本ID数组
        neg = np.zeros([self.maxlen + 1], dtype=np.int32)  # 负样本ID数组
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 序列中每个token的类型（user/item）
        next_token_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 下一个token的类型
        next_action_type = np.zeros([self.maxlen + 1], dtype=np.int32)  # 下一个token的行为类型

        # 初始化用于存储特征的numpy数组，数据类型为object以容纳字典
        seq_feat = np.empty([self.maxlen + 1], dtype=object)  # 序列特征数组
        pos_feat = np.empty([self.maxlen + 1], dtype=object)  # 正样本特征数组
        neg_feat = np.empty([self.maxlen + 1], dtype=object)  # 负样本特征数组

        nxt = ext_user_sequence[-1]  # 将扩展序列的最后一个事件作为“下一个”事件的初始值
        idx = self.maxlen  # 初始化填充索引为最大长度

        ts = set()  # 创建一个集合，用于存储序列中出现过的所有物品ID，方便负采样
        for record_tuple in ext_user_sequence:  # 遍历扩展序列
            if record_tuple[2] == 1 and record_tuple[0]:  # 如果是物品事件且ID不为0
                ts.add(record_tuple[0])  # 将物品ID添加到集合中

        # left-padding, 从后往前遍历，将用户序列填充到maxlen+1的长度
        for record_tuple in reversed(ext_user_sequence[:-1]):  # 反向遍历除最后一个事件外的所有事件
            i, feat, type_, act_type = record_tuple  # 当前事件的ID，特征，类型，行为类型
            next_i, next_feat, next_type, next_act_type = nxt  # 下一个事件的ID，特征，类型，行为类型
            feat = self.fill_missing_feat(feat, i)  # 为当前事件的特征填充缺失值
            next_feat = self.fill_missing_feat(next_feat, next_i)  # 为下一个事件的特征填充缺失值
            seq[idx] = i  # 在当前索引位置填充序列ID
            token_type[idx] = type_  # 填充token类型
            next_token_type[idx] = next_type  # 填充下一个token的类型
            if next_act_type is not None:  # 如果下一个行为类型存在
                next_action_type[idx] = next_act_type  # 填充下一个行为类型
            seq_feat[idx] = feat  # 填充序列特征
            if next_type == 1 and next_i != 0:  # 如果下一个事件是物品事件且ID不为0
                pos[idx] = next_i  # 将其作为正样本
                pos_feat[idx] = next_feat  # 填充正样本特征
                neg_id = self._random_neq(1, self.itemnum + 1, ts)  # 进行负采样，生成一个负样本ID
                neg[idx] = neg_id  # 填充负样本ID
                # 获取并填充负样本的特征
                neg_feat[idx] = self.fill_missing_feat(self.item_feat_dict[str(neg_id)], neg_id)
            nxt = record_tuple  # 更新“下一个”事件为当前事件
            idx -= 1  # 索引向前移动一位
            if idx == -1:  # 如果索引超出范围，则停止填充
                break

        # 使用默认值填充特征数组中可能存在的None值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)
        pos_feat = np.where(pos_feat == None, self.feature_default_value, pos_feat)
        neg_feat = np.where(neg_feat == None, self.feature_default_value, neg_feat)

        # 返回所有处理好的模型输入数据
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat

    def __len__(self):
        """
        返回数据集长度，即用户数量

        Returns:
            usernum: 用户数量
        """
        return len(self.seq_offsets)  # 返回用户数量，即偏移量列表的长度

    def _init_feat_info(self):
        """
        初始化特征信息, 包括特征缺省值和特征类型

        Returns:
            feat_default_value: 特征缺省值，每个元素为字典，key为特征ID，value为特征缺省值
            feat_types: 特征类型，key为特征类型名称，value为包含的特征ID列表
        """
        feat_default_value = {}  # 初始化特征默认值字典
        feat_statistics = {}  # 初始化特征统计信息字典
        feat_types = {}  # 初始化特征类型字典
        # 定义用户稀疏特征的ID列表
        feat_types['user_sparse'] = ['103', '104', '105', '109']
        # 定义物品稀疏特征的ID列表
        feat_types['item_sparse'] = [
            '100', '117', '111', '118', '101', '102', '119', '120',
            '114', '112', '121', '115', '122', '116',
        ]
        feat_types['item_array'] = []  # 定义物品数组特征的ID列表（当前为空）
        feat_types['user_array'] = ['106', '107', '108', '110']  # 定义用户数组特征的ID列表
        feat_types['item_emb'] = self.mm_emb_ids  # 定义物品多模态Embedding特征的ID列表
        feat_types['user_continual'] = []  # 定义用户连续特征的ID列表（当前为空）
        feat_types['item_continual'] = []  # 定义物品连续特征的ID列表（当前为空）

        for feat_id in feat_types['user_sparse']:  # 遍历用户稀疏特征
            feat_default_value[feat_id] = 0  # 设置默认值为0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])  # 统计该特征的取值数量（词表大小）
        for feat_id in feat_types['item_sparse']:  # 遍历物品稀疏特征
            feat_default_value[feat_id] = 0  # 设置默认值为0
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])  # 统计词表大小
        for feat_id in feat_types['item_array']:  # 遍历物品数组特征
            feat_default_value[feat_id] = [0]  # 设置默认值为[0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])  # 统计词表大小
        for feat_id in feat_types['user_array']:  # 遍历用户数组特征
            feat_default_value[feat_id] = [0]  # 设置默认值为[0]
            feat_statistics[feat_id] = len(self.indexer['f'][feat_id])  # 统计词表大小
        for feat_id in feat_types['user_continual']:  # 遍历用户连续特征
            feat_default_value[feat_id] = 0  # 设置默认值为0
        for feat_id in feat_types['item_continual']:  # 遍历物品连续特征
            feat_default_value[feat_id] = 0  # 设置默认值为0
        for feat_id in feat_types['item_emb']:  # 遍历物品多模态Embedding特征
            # 设置默认值为一个全零的numpy数组，其维度与该特征的Embedding维度一致
            feat_default_value[feat_id] = np.zeros(
                list(self.mm_emb_dict[feat_id].values())[0].shape[0], dtype=np.float32
            )

        return feat_default_value, feat_types, feat_statistics  # 返回初始化好的三个字典

    def fill_missing_feat(self, feat, item_id):
        """
        对于原始数据中缺失的特征进行填充缺省值

        Args:
            feat: 特征字典
            item_id: 物品ID

        Returns:
            filled_feat: 填充后的特征字典
        """
        if feat is None:  # 如果输入的特征字典为None
            feat = {}  # 将其初始化为空字典
        filled_feat = {}  # 初始化用于存储填充后特征的字典
        for k in feat.keys():  # 遍历输入特征字典的所有键
            filled_feat[k] = feat[k]  # 将已有的特征复制到新字典中

        all_feat_ids = []  # 初始化一个包含所有特征ID的列表
        for feat_type in self.feature_types.values():  # 遍历所有特征类型
            all_feat_ids.extend(feat_type)  # 将该类型下的所有特征ID添加到列表中
        # 计算出当前特征字典中缺失的特征ID集合
        missing_fields = set(all_feat_ids) - set(feat.keys())
        for feat_id in missing_fields:  # 遍历所有缺失的特征ID
            filled_feat[feat_id] = self.feature_default_value[feat_id]  # 使用预设的默认值进行填充
        for feat_id in self.feature_types['item_emb']:  # 专门处理物品多模态Embedding特征
            # 如果物品ID有效，并且在多模态Embedding字典中存在该物品的原始ID
            if item_id != 0 and self.indexer_i_rev[item_id] in self.mm_emb_dict[feat_id]:
                # 并且其Embedding是numpy数组类型
                if isinstance(self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]], np.ndarray):
                    # 则将该Embedding填充到特征字典中
                    filled_feat[feat_id] = self.mm_emb_dict[feat_id][self.indexer_i_rev[item_id]]

        return filled_feat  # 返回填充完毕的特征字典

    @staticmethod
    def collate_fn(batch):
        """
        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            pos: 正样本ID, torch.Tensor形式
            neg: 负样本ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            next_token_type: 下一个token类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            pos_feat: 正样本特征, list形式
            neg_feat: 负样本特征, list形式
        """
        # 使用zip(*)将batch中的数据解包成独立的元组
        seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = zip(*batch)
        # 将numpy数组转换为torch.Tensor
        seq = torch.from_numpy(np.array(seq))
        pos = torch.from_numpy(np.array(pos))
        neg = torch.from_numpy(np.array(neg))
        token_type = torch.from_numpy(np.array(token_type))
        next_token_type = torch.from_numpy(np.array(next_token_type))
        next_action_type = torch.from_numpy(np.array(next_action_type))
        # 将特征数据转换为列表
        seq_feat = list(seq_feat)
        pos_feat = list(pos_feat)
        neg_feat = list(neg_feat)
        # 返回拼接好的batch数据
        return seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat


class MyTestDataset(MyDataset):
    """
    测试数据集
    """

    def __init__(self, data_dir, args):
        super().__init__(data_dir, args)  # 调用父类的初始化方法

    def _load_data_and_offsets(self):
        # 打开用于预测的用户序列文件
        self.data_file = open(self.data_dir / "predict_seq.jsonl", 'rb')
        # 加载预测序列对应的文件偏移量
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            self.seq_offsets = pickle.load(f)

    def _process_cold_start_feat(self, feat):
        """
        处理冷启动特征。训练集未出现过的特征value为字符串，默认转换为0.可设计替换为更好的方法。
        """
        processed_feat = {}  # 初始化处理后的特征字典
        for feat_id, feat_value in feat.items():  # 遍历输入的特征字典
            if isinstance(feat_value, list):  # 如果特征值是列表
                value_list = []  # 初始化一个新的列表
                for v in feat_value:  # 遍历列表中的每个值
                    if isinstance(v, str):  # 如果值是字符串（冷启动值）
                        value_list.append(0)  # 用0替换
                    else:
                        value_list.append(v)  # 否则保留原值
                processed_feat[feat_id] = value_list  # 更新处理后的特征值
            elif isinstance(feat_value, str):  # 如果特征值是字符串
                processed_feat[feat_id] = 0  # 用0替换
            else:
                processed_feat[feat_id] = feat_value  # 否则保留原值
        return processed_feat  # 返回处理后的特征字典

    def __getitem__(self, uid):
        """
        获取单个用户的数据，并进行padding处理，生成模型需要的数据格式

        Args:
            uid: 用户在self.data_file中储存的行号
        Returns:
            seq: 用户序列ID
            token_type: 用户序列类型，1表示item，2表示user
            seq_feat: 用户序列特征，每个元素为字典，key为特征ID，value为特征值
            user_id: user_id eg. user_xxxxxx ,便于后面对照答案
        """
        user_sequence = self._load_user_data(uid)  # 动态加载用户数据

        ext_user_sequence = []  # 初始化扩展用户序列
        for record_tuple in user_sequence:  # 遍历原始序列
            u, i, user_feat, item_feat, _, _ = record_tuple  # 解包记录
            if u:  # 如果存在用户ID
                if isinstance(u, str):  # 如果是字符串，说明是原始user_id
                    user_id = u
                else:  # 如果是int，说明是re_id
                    user_id = self.indexer_u_rev[u]  # 通过反向索引找到原始user_id
            if u and user_feat:  # 如果存在用户ID和特征
                if isinstance(u, str):  # 如果用户ID是字符串（冷启动）
                    u = 0  # 转换为0
                if user_feat:  # 如果存在用户特征
                    user_feat = self._process_cold_start_feat(user_feat)  # 处理冷启动特征
                ext_user_sequence.insert(0, (u, user_feat, 2))  # 插入用户事件

            if i and item_feat:  # 如果存在物品ID和特征
                # 序列对于训练时没见过的item，不会直接赋0，而是保留creative_id，creative_id远大于训练时的itemnum
                if i > self.itemnum:  # 如果物品ID大于训练集中的最大ID（冷启动）
                    i = 0  # 转换为0
                if item_feat:  # 如果存在物品特征
                    item_feat = self._process_cold_start_feat(item_feat)  # 处理冷启动特征
                ext_user_sequence.append((i, item_feat, 1))  # 追加物品事件

        # 初始化用于存储模型输入的numpy数组
        seq = np.zeros([self.maxlen + 1], dtype=np.int32)
        token_type = np.zeros([self.maxlen + 1], dtype=np.int32)
        seq_feat = np.empty([self.maxlen + 1], dtype=object)

        idx = self.maxlen  # 初始化填充索引

        ts = set()  # 初始化一个集合用于存储序列中的物品ID
        for record_tuple in ext_user_sequence:
            if record_tuple[2] == 1 and record_tuple[0]:
                ts.add(record_tuple[0])

        for record_tuple in reversed(ext_user_sequence[:-1]):  # 反向遍历除最后一个事件外的序列
            i, feat, type_ = record_tuple  # 解包事件
            feat = self.fill_missing_feat(feat, i)  # 填充缺失特征
            seq[idx] = i  # 填充序列ID
            token_type[idx] = type_  # 填充token类型
            seq_feat[idx] = feat  # 填充序列特征
            idx -= 1  # 索引前移
            if idx == -1:  # 如果索引越界则停止
                break

        # 使用默认值填充特征数组中可能存在的None值
        seq_feat = np.where(seq_feat == None, self.feature_default_value, seq_feat)

        return seq, token_type, seq_feat, user_id  # 返回处理好的数据

    def __len__(self):
        """
        Returns:
            len(self.seq_offsets): 用户数量
        """
        # 重新加载一次偏移量文件以获取最新长度（可能不是最优实现，但能确保正确性）
        with open(Path(self.data_dir, 'predict_seq_offsets.pkl'), 'rb') as f:
            temp = pickle.load(f)
        return len(temp)  # 返回预测集中的用户数量

    @staticmethod
    def collate_fn(batch):
        """
        将多个__getitem__返回的数据拼接成一个batch

        Args:
            batch: 多个__getitem__返回的数据

        Returns:
            seq: 用户序列ID, torch.Tensor形式
            token_type: 用户序列类型, torch.Tensor形式
            seq_feat: 用户序列特征, list形式
            user_id: user_id, str
        """
        seq, token_type, seq_feat, user_id = zip(*batch)  # 解包batch数据
        seq = torch.from_numpy(np.array(seq))  # 转换为Tensor
        token_type = torch.from_numpy(np.array(token_type))  # 转换为Tensor
        seq_feat = list(seq_feat)  # 转换为list

        return seq, token_type, seq_feat, user_id  # 返回拼接好的batch


def save_emb(emb, save_path):
    """
    将Embedding保存为二进制文件

    Args:
        emb: 要保存的Embedding，形状为 [num_points, num_dimensions]
        save_path: 保存路径
    """
    num_points = emb.shape[0]  # 数据点数量
    num_dimensions = emb.shape[1]  # 向量的维度
    print(f'saving {save_path}')  # 打印保存信息
    with open(Path(save_path), 'wb') as f:  # 以二进制写入模式打开文件
        # 将数据点数量和维度打包成二进制格式并写入文件
        f.write(struct.pack('II', num_points, num_dimensions))
        emb.tofile(f)  # 将numpy数组的内容写入文件


def load_mm_emb(mm_path, feat_ids):
    """
    加载多模态特征Embedding

    Args:
        mm_path: 多模态特征Embedding路径
        feat_ids: 要加载的多模态特征ID列表

    Returns:
        mm_emb_dict: 多模态特征Embedding字典，key为特征ID，value为特征Embedding字典（key为item ID，value为Embedding）
    """
    # 定义不同特征ID对应的Embedding维度
    SHAPE_DICT = {"81": 32, "82": 1024, "83": 3584, "84": 4096, "85": 3584, "86": 3584}
    mm_emb_dict = {}  # 初始化多模态Embedding字典
    for feat_id in tqdm(feat_ids, desc='Loading mm_emb'):  # 使用tqdm显示加载进度
        shape = SHAPE_DICT[feat_id]  # 获取当前特征ID的维度
        emb_dict = {}  # 初始化当前特征的Embedding字典
        if feat_id != '81':  # 对非'81'的特征ID进行处理
            try:
                # 构建特征Embedding存储的基础路径
                base_path = Path(mm_path, f'emb_{feat_id}_{shape}')
                for json_file in base_path.glob('*.json'):  # 遍历路径下所有的json文件
                    with open(json_file, 'r', encoding='utf-8') as file:  # 打开json文件
                        for line in file:  # 逐行读取
                            data_dict_origin = json.loads(line.strip())  # 解析json行
                            insert_emb = data_dict_origin['emb']  # 获取Embedding
                            if isinstance(insert_emb, list):  # 如果是列表格式
                                # 转换为numpy数组，并指定数据类型
                                insert_emb = np.array(insert_emb, dtype=np.float32)
                            # 构建一个以匿名cid为键，Embedding为值的字典
                            data_dict = {data_dict_origin['anonymous_cid']: insert_emb}
                            emb_dict.update(data_dict)  # 更新到当前特征的Embedding字典中
            except Exception as e:  # 捕获并打印可能的异常
                print(f"transfer error: {e}")
        if feat_id == '81':  # 对特征ID '81' 进行特殊处理
            with open(Path(mm_path, f'emb_{feat_id}_{shape}.pkl'), 'rb') as f:  # 打开对应的pickle文件
                emb_dict = pickle.load(f)  # 加载pickle数据
        mm_emb_dict[feat_id] = emb_dict  # 将当前特征的Embedding字典存入总字典
        print(f'Loaded #{feat_id} mm_emb')  # 打印加载完成信息
    return mm_emb_dict  # 返回加载好的多模态Embedding字典
