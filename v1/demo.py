"""
演示脚本 - 使用模拟数据测试完整的训练流程

这个脚本创建模拟的推荐系统数据，并运行一个小规模的训练来验证整个流程
"""

import os
import json
import pickle
import tempfile
from pathlib import Path
import numpy as np
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mock_data(data_dir: Path, num_users: int = 10, num_items: int = 100, seq_length: int = 20):
    """
    创建模拟的推荐系统数据
    
    Args:
        data_dir: 数据目录
        num_users: 用户数量
        num_items: 物品数量
        seq_length: 序列长度
    """
    logger.info(f"Creating mock data in {data_dir}")
    
    # 创建用户序列数据
    sequences = []
    seq_offsets = {}
    
    for user_id in range(1, num_users + 1):
        user_seq = []
        for i in range(seq_length):
            item_id = np.random.randint(1, num_items + 1)
            user_feat = {"age": np.random.randint(18, 65), "gender": np.random.randint(0, 2)}
            item_feat = {"category": np.random.randint(1, 10), "price": np.random.uniform(10, 1000)}
            action_type = np.random.randint(0, 2)  # 0: view, 1: click
            timestamp = 1600000000 + i * 3600  # Mock timestamp
            
            user_seq.append([user_id, item_id, user_feat, item_feat, action_type, timestamp])
        
        sequences.append(user_seq)
    
    # 保存序列数据
    with open(data_dir / "seq.jsonl", 'w') as f:
        offset = 0
        for i, seq in enumerate(sequences):
            seq_offsets[i] = offset
            line = json.dumps(seq) + '\\n'
            f.write(line)
            offset += len(line.encode('utf-8'))
    
    # 保存偏移量
    with open(data_dir / "seq_offsets.pkl", 'wb') as f:
        pickle.dump(seq_offsets, f)
    
    # 创建索引器
    indexer = {
        'i': {item_id: item_id for item_id in range(1, num_items + 1)},
        'u': {user_id: user_id for user_id in range(1, num_users + 1)}
    }
    with open(data_dir / "indexer.pkl", 'wb') as f:
        pickle.dump(indexer, f)
    
    # 创建物品特征字典
    item_feat_dict = {}
    for item_id in range(1, num_items + 1):
        item_feat_dict[str(item_id)] = {
            "category": np.random.randint(1, 10),
            "price": float(np.random.uniform(10, 1000)),
            "brand": np.random.randint(1, 20)
        }
    
    with open(data_dir / "item_feat_dict.json", 'w') as f:
        json.dump(item_feat_dict, f)
    
    # 创建多模态特征目录（可选）
    mm_dir = data_dir / "creative_emb"
    mm_dir.mkdir(exist_ok=True)
    
    # 创建模拟的多模态特征（这里创建空的，因为测试中我们可能不需要）
    mock_mm_features = {}
    for item_id in range(1, min(num_items + 1, 50)):  # 只为前50个物品创建特征
        mock_mm_features[str(item_id)] = np.random.randn(32).tolist()  # 32维特征
    
    np.save(mm_dir / "81.npy", mock_mm_features)
    
    logger.info(f"Mock data created: {num_users} users, {num_items} items")


def run_demo_training():
    """运行演示训练"""
    logger.info("Starting demo training with mock data")
    
    # 创建临时数据目录
    with tempfile.TemporaryDirectory() as temp_dir:
        data_path = Path(temp_dir)
        
        # 创建模拟数据
        create_mock_data(data_path, num_users=5, num_items=20, seq_length=10)
        
        # 设置环境变量
        os.environ['TRAIN_DATA_PATH'] = str(data_path)
        os.environ['TRAIN_LOG_PATH'] = str(data_path / 'logs')
        os.environ['TRAIN_TF_EVENTS_PATH'] = str(data_path / 'tb_logs')
        
        # 创建模拟参数
        class MockArgs:
            # 数据参数
            batch_size = 2
            maxlen = 8
            num_workers = 0
            
            # 模型参数
            hidden_units = 16
            num_blocks = 1
            num_heads = 1
            dropout_rate = 0.1
            norm_first = True
            
            # 训练参数
            num_epochs = 2
            lr = 0.01
            optimizer = 'adam'
            weight_decay = 0.01
            lr_scheduler = 'none'
            grad_clip = 1.0
            l2_emb = 0.0
            
            # 系统参数
            device = 'cpu'  # 使用CPU避免CUDA问题
            seed = 42
            save_freq = 1
            
            # 其他参数
            state_dict_path = None
            inference_only = False
            mm_emb_id = ['81']
        
        try:
            # 导入训练模块
            from dataset import OptimizedDataset
            from model import OptimizedBaselineModel
            from main import ModelTrainer
            
            args = MockArgs()
            
            # 测试数据集加载
            logger.info("Testing dataset loading...")
            dataset = OptimizedDataset(str(data_path), args)
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # 测试模型创建
            logger.info("Testing model creation...")
            model = OptimizedBaselineModel(
                user_num=dataset.usernum,
                item_num=dataset.itemnum,
                feat_statistics=dataset.feat_statistics,
                feat_types=dataset.feature_types,
                args=args
            )
            logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
            
            # 测试训练器
            logger.info("Testing trainer initialization...")
            trainer = ModelTrainer(args)
            logger.info("Trainer initialized successfully")
            
            # 运行一个短的训练
            logger.info("Running short training demo...")
            trainer.train()
            
            logger.info("✓ Demo training completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Demo training failed: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """主函数"""
    print("=" * 60)
    print("DEMO TRAINING WITH MOCK DATA")
    print("=" * 60)
    
    try:
        success = run_demo_training()
        
        if success:
            print("\\n🎉 Demo training completed successfully!")
            print("\\nThe optimized generative recommendation system is working correctly.")
            print("\\nTo use with real data:")
            print("1. Prepare your data in the required format")
            print("2. Set the TRAIN_DATA_PATH environment variable")
            print("3. Run: python main.py with your desired parameters")
        else:
            print("\\n⚠️ Demo training failed. Please check the logs above.")
            return False
            
    except Exception as e:
        print(f"\\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
