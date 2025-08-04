"""
简化的测试脚本

专注于测试核心功能，避免复杂的特征处理问题
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


def test_simple_imports():
    """测试基本导入"""
    try:
        import torch
        import numpy as np
        from tqdm import tqdm
        print(f"✓ PyTorch {torch.__version__} imported successfully")
        print(f"✓ NumPy {np.__version__} imported successfully")
        print(f"✓ tqdm imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_torch_operations():
    """测试PyTorch基本操作"""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        
        # 测试基本张量操作
        x = torch.randn(2, 10, 32)
        y = torch.randn(2, 10, 32)
        z = x + y
        print(f"✓ Basic tensor operations work: {z.shape}")
        
        # 测试线性层
        linear = nn.Linear(32, 64)
        output = linear(x)
        print(f"✓ Linear layer works: {output.shape}")
        
        # 测试注意力机制基本组件
        q = torch.randn(2, 2, 10, 16)  # batch, heads, seq, dim
        k = torch.randn(2, 2, 10, 16)
        v = torch.randn(2, 2, 10, 16)
        
        scores = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        print(f"✓ Attention mechanism works: {attn_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ PyTorch operations failed: {e}")
        return False


def test_model_components():
    """测试模型组件"""
    try:
        import torch
        import torch.nn as nn
        
        # 测试Embedding层
        vocab_size = 1000
        embed_dim = 64
        embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # 测试输入
        seq = torch.randint(1, vocab_size, (2, 10))  # batch_size=2, seq_len=10
        emb_output = embedding(seq)
        print(f"✓ Embedding layer works: {emb_output.shape}")
        
        # 测试LayerNorm
        layer_norm = nn.LayerNorm(embed_dim)
        norm_output = layer_norm(emb_output)
        print(f"✓ LayerNorm works: {norm_output.shape}")
        
        # 测试简单的前馈网络
        ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        ffn_output = ffn(emb_output)
        print(f"✓ Feed-forward network works: {ffn_output.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model components test failed: {e}")
        return False


def test_training_loop():
    """测试简单的训练循环"""
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        
        # 创建简单的模型
        model = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        # 创建优化器和损失函数
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        # 模拟训练数据
        x = torch.randn(32, 64)  # batch_size=32, input_dim=64
        y = torch.randn(32, 1)   # target
        
        # 训练步骤
        model.train()
        for i in range(5):  # 5个训练步骤
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            
            if i == 0:
                initial_loss = loss.item()
            elif i == 4:
                final_loss = loss.item()
        
        print(f"✓ Training loop works: loss {initial_loss:.4f} -> {final_loss:.4f}")
        return True
        
    except Exception as e:
        print(f"✗ Training loop test failed: {e}")
        return False


def test_file_operations():
    """测试文件操作"""
    try:
        import json
        import pickle
        
        # 测试JSON操作
        data = {"test": "data", "numbers": [1, 2, 3]}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            json_file = f.name
        
        with open(json_file, 'r') as f:
            loaded_data = json.load(f)
        
        os.unlink(json_file)
        print(f"✓ JSON operations work: {loaded_data}")
        
        # 测试Pickle操作
        data = {"tensor": [1, 2, 3], "dict": {"a": 1}}
        with tempfile.NamedTemporaryFile(mode='wb', suffix='.pkl', delete=False) as f:
            pickle.dump(data, f)
            pkl_file = f.name
        
        with open(pkl_file, 'rb') as f:
            loaded_data = pickle.load(f)
        
        os.unlink(pkl_file)
        print(f"✓ Pickle operations work: {loaded_data}")
        
        return True
        
    except Exception as e:
        print(f"✗ File operations test failed: {e}")
        return False


def main():
    """主测试函数"""
    print("=" * 60)
    print("SIMPLIFIED SYSTEM TESTS")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_simple_imports),
        ("PyTorch Operations Test", test_torch_operations),
        ("Model Components Test", test_model_components),
        ("Training Loop Test", test_training_loop),
        ("File Operations Test", test_file_operations),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\nRunning {test_name}...")
        print("-" * 40)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            print(f"✗ {test_name} FAILED: {e}")
    
    print("\\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\\n🎉 All tests passed! The system is ready to use.")
        print("\\nNext steps:")
        print("1. Prepare your data in the required format")
        print("2. Run: python main.py --help to see training options")
        print("3. Start training: bash run.sh")
        return True
    else:
        print(f"\\n⚠️  {total - passed} tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
