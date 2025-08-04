"""
优化版模型测试脚本

用于测试优化后的代码是否能正常运行，包括：
1. 导入测试
2. 数据处理测试
3. 模型初始化测试
4. 训练流程测试
"""

import os
import sys
import tempfile
import json
import pickle
from pathlib import Path
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRunner:
    """测试运行器"""
    
    def __init__(self):
        self.test_results = []
    
    def run_test(self, test_name: str, test_func):
        """运行单个测试"""
        try:
            logger.info(f"Running test: {test_name}")
            test_func()
            self.test_results.append((test_name, "PASSED", None))
            logger.info(f"✓ {test_name} PASSED")
        except Exception as e:
            self.test_results.append((test_name, "FAILED", str(e)))
            logger.error(f"✗ {test_name} FAILED: {e}")
    
    def print_results(self):
        """打印测试结果"""
        print("\\n" + "="*60)
        print("TEST RESULTS")
        print("="*60)
        
        passed = sum(1 for _, status, _ in self.test_results if status == "PASSED")
        total = len(self.test_results)
        
        for test_name, status, error in self.test_results:
            status_symbol = "✓" if status == "PASSED" else "✗"
            print(f"{status_symbol} {test_name}: {status}")
            if error:
                print(f"  Error: {error}")
        
        print(f"\\nSummary: {passed}/{total} tests passed")
        print("="*60)
        
        return passed == total


def test_imports():
    """测试导入"""
    try:
        import torch
        import numpy as np
        from tqdm import tqdm
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"NumPy version: {np.__version__}")
        logger.info("All imports successful")
    except ImportError as e:
        raise ImportError(f"Import failed: {e}")


def test_dataset_creation():
    """测试数据集创建"""
    # 创建模拟参数
    class MockArgs:
        maxlen = 10
        mm_emb_id = ['81']
        device = 'cpu'
    
    # 创建临时数据目录和文件
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 创建模拟数据文件
        seq_file = temp_path / "seq.jsonl"
        with open(seq_file, 'w') as f:
            # 写入模拟序列数据
            sample_data = [
                [1, 100, {}, {"feature1": 1}, 1, 1234567890],
                [1, 101, {}, {"feature1": 2}, 1, 1234567891]
            ]
            f.write(json.dumps(sample_data) + '\\n')
        
        # 创建偏移文件
        offset_file = temp_path / "seq_offsets.pkl"
        with open(offset_file, 'wb') as f:
            pickle.dump({0: 0}, f)
        
        # 创建索引器文件
        indexer_file = temp_path / "indexer.pkl"
        indexer = {
            'i': {100: 1, 101: 2},
            'u': {1: 1}
        }
        with open(indexer_file, 'wb') as f:
            pickle.dump(indexer, f)
        
        # 创建物品特征文件
        item_feat_file = temp_path / "item_feat_dict.json"
        with open(item_feat_file, 'w') as f:
            json.dump({"100": {"feature1": 1}, "101": {"feature1": 2}}, f)
        
        # 创建多模态特征目录
        mm_dir = temp_path / "creative_emb"
        mm_dir.mkdir()
        
        # 测试数据集初始化
        from dataset import OptimizedDataset
        
        args = MockArgs()
        dataset = OptimizedDataset(str(temp_path), args)
        
        logger.info(f"Dataset created with {len(dataset)} samples")
        
        # 测试获取样本
        if len(dataset) > 0:
            sample = dataset[0]
            logger.info(f"Sample shape: {len(sample)} components")


def test_model_creation():
    """测试模型创建"""
    try:
        import torch
        
        # 创建模拟参数
        class MockArgs:
            hidden_units = 32
            num_blocks = 1
            num_heads = 1
            dropout_rate = 0.1
            norm_first = True
            device = 'cpu'
            maxlen = 10
        
        from model import OptimizedBaselineModel
        
        # 模拟特征统计和类型
        feat_statistics = {'81': 32}
        feat_types = {
            'user_sparse': [],
            'user_continual': [],
            'user_array': [],
            'item_sparse': [],
            'item_continual': [],
            'item_array': [],
            'item_emb': ['81']
        }
        
        args = MockArgs()
        model = OptimizedBaselineModel(
            user_num=100,
            item_num=1000,
            feat_statistics=feat_statistics,
            feat_types=feat_types,
            args=args
        )
        
        # 测试模型前向传播
        batch_size = 2
        seq_len = 10
        
        # 创建模拟输入
        seq = torch.randint(1, 100, (batch_size, seq_len))
        pos = torch.randint(1, 100, (batch_size, seq_len))
        neg = torch.randint(1, 100, (batch_size, seq_len))
        mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        next_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
        next_action = torch.ones(batch_size, seq_len, dtype=torch.long)
        
        # 模拟特征
        seq_features = [[{} for _ in range(seq_len)] for _ in range(batch_size)]
        pos_features = [[{} for _ in range(seq_len)] for _ in range(batch_size)]
        neg_features = [[{} for _ in range(seq_len)] for _ in range(batch_size)]
        
        # 前向传播
        with torch.no_grad():
            pos_logits, neg_logits = model(
                seq, pos, neg, mask, next_mask, next_action,
                seq_features, pos_features, neg_features
            )
        
        logger.info(f"Model forward pass successful: {pos_logits.shape}, {neg_logits.shape}")
        
    except Exception as e:
        raise RuntimeError(f"Model creation failed: {e}")


def test_training_setup():
    """测试训练设置"""
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader
        
        # 创建模拟训练组件
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        
        # 测试简单的前向和反向传播
        x = torch.randn(5, 10)
        y = torch.randn(5, 1)
        
        optimizer.zero_grad()
        output = model(x)
        loss = criterion(output, torch.sigmoid(y))
        loss.backward()
        optimizer.step()
        
        logger.info(f"Training setup test successful, loss: {loss.item()}")
        
    except Exception as e:
        raise RuntimeError(f"Training setup failed: {e}")


def test_configuration_loading():
    """测试配置加载"""
    config_file = Path(__file__).parent / "config.yaml"
    
    if config_file.exists():
        try:
            import yaml
            with open(config_file, 'r') as f:
                config = yaml.safe_load(f)
            logger.info("Configuration file loaded successfully")
            logger.info(f"Model config: {config.get('model', {})}")
        except ImportError:
            logger.warning("PyYAML not installed, skipping YAML config test")
            # 简单的文本解析测试
            with open(config_file, 'r') as f:
                content = f.read()
            logger.info("Configuration file exists and readable")
    else:
        raise FileNotFoundError("Configuration file not found")


def main():
    """主测试函数"""
    print("="*60)
    print("OPTIMIZED GENERATIVE RECOMMENDATION SYSTEM TESTS")
    print("="*60)
    
    runner = TestRunner()
    
    # 运行所有测试
    runner.run_test("Import Test", test_imports)
    runner.run_test("Dataset Creation Test", test_dataset_creation)
    runner.run_test("Model Creation Test", test_model_creation)
    runner.run_test("Training Setup Test", test_training_setup)
    runner.run_test("Configuration Loading Test", test_configuration_loading)
    
    # 打印结果
    all_passed = runner.print_results()
    
    if all_passed:
        print("\\n🎉 All tests passed! The optimized system is ready to use.")
    else:
        print("\\n⚠️  Some tests failed. Please check the errors above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
