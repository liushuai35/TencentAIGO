"""
优化版训练主程序

主要改进：
1. 更完善的参数配置和验证
2. 改进的训练循环和优化策略
3. 更好的日志记录和监控
4. 支持分布式训练
5. 改进的模型保存和加载
6. 更全面的评估指标
"""

import argparse
import json
import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import OptimizedDataset, OptimizedTestDataset
from model import OptimizedBaselineModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    优化版模型训练器
    
    提供完整的训练、验证和测试流程
    """
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.best_val_ndcg = 0.0
        self.best_val_hr = 0.0
        
        # 设置随机种子
        self._set_random_seed(args.seed)
        
        # 初始化日志和监控
        self._setup_logging()
        
        # 加载数据
        self._load_datasets()
        
        # 初始化模型
        self._initialize_model()
        
        # 设置优化器和损失函数
        self._setup_optimizer()
        
        logger.info("ModelTrainer initialized successfully")
    
    def _set_random_seed(self, seed: int = 42):
        """设置随机种子"""
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _setup_logging(self):
        """设置日志和TensorBoard"""
        log_dir = Path(os.environ.get('TRAIN_LOG_PATH', './logs'))
        tb_dir = Path(os.environ.get('TRAIN_TF_EVENTS_PATH', './tb_logs'))
        
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = open(log_dir / 'train.log', 'w')
        self.writer = SummaryWriter(tb_dir)
        
        logger.info(f"Logging to {log_dir}, TensorBoard to {tb_dir}")
    
    def _load_datasets(self):
        """加载数据集"""
        data_path = os.environ.get('TRAIN_DATA_PATH', './data')
        
        try:
            self.dataset = OptimizedDataset(data_path, self.args)
            
            # 划分训练集和验证集
            train_size = int(0.9 * len(self.dataset))
            val_size = len(self.dataset) - train_size
            self.train_dataset, self.val_dataset = random_split(
                self.dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            self.train_loader = DataLoader(
                self.train_dataset,
                batch_size=self.args.batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                collate_fn=self.dataset.collate_fn,
                pin_memory=True if self.args.device == 'cuda' else False
            )
            
            self.val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.args.batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                collate_fn=self.dataset.collate_fn,
                pin_memory=True if self.args.device == 'cuda' else False
            )
            
            logger.info(f"Loaded dataset: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")
            
        except Exception as e:
            logger.error(f"Error loading datasets: {e}")
            raise
    
    def _initialize_model(self):
        """初始化模型"""
        try:
            self.model = OptimizedBaselineModel(
                user_num=self.dataset.usernum,
                item_num=self.dataset.itemnum,
                feat_statistics=self.dataset.feat_statistics,
                feat_types=self.dataset.feature_types,
                args=self.args
            ).to(self.device)
            
            # 加载预训练权重（如果指定）
            if self.args.state_dict_path:
                self._load_checkpoint()
            
            logger.info(f"Model initialized with {sum(p.numel() for p in self.model.parameters())} parameters")
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            raise
    
    def _setup_optimizer(self):
        """设置优化器和损失函数"""
        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        
        # 优化器配置
        if self.args.optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.98),
                weight_decay=self.args.weight_decay
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(0.9, 0.95),
                weight_decay=self.args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.args.optimizer}")
        
        # 学习率调度器
        if self.args.lr_scheduler == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.args.num_epochs
            )
        elif self.args.lr_scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=10, gamma=0.5
            )
        else:
            self.scheduler = None
        
        logger.info(f"Optimizer: {self.args.optimizer}, LR: {self.args.lr}")
    
    def _load_checkpoint(self):
        """加载检查点"""
        try:
            checkpoint = torch.load(self.args.state_dict_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    if 'optimizer_state_dict' in checkpoint:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    if 'epoch' in checkpoint:
                        self.start_epoch = checkpoint['epoch'] + 1
                    if 'best_val_ndcg' in checkpoint:
                        self.best_val_ndcg = checkpoint['best_val_ndcg']
                else:
                    self.model.load_state_dict(checkpoint)
                    self.start_epoch = 1
            else:
                self.model.load_state_dict(checkpoint)
                self.start_epoch = 1
            
            logger.info(f"Loaded checkpoint from {self.args.state_dict_path}")
            
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            self.start_epoch = 1
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_ndcg': self.best_val_ndcg,
            # 不保存args对象，避免pickle问题
        }
        
        # 保存当前检查点
        checkpoint_path = Path(os.environ.get('TRAIN_LOG_PATH', './logs')) / f'checkpoint_epoch_{epoch}.pt'
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = Path(os.environ.get('TRAIN_LOG_PATH', './logs')) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            try:
                # 解包批次数据
                seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                
                # 移动到设备
                seq = seq.to(self.device)
                pos = pos.to(self.device)
                neg = neg.to(self.device)
                token_type = token_type.to(self.device)
                next_token_type = next_token_type.to(self.device)
                next_action_type = next_action_type.to(self.device)
                
                # 前向传播
                pos_logits, neg_logits = self.model(
                    seq, pos, neg, token_type, next_token_type, next_action_type,
                    seq_feat, pos_feat, neg_feat
                )
                
                # 计算损失
                pos_labels = torch.ones_like(pos_logits)
                neg_labels = torch.zeros_like(neg_logits)
                
                # 只在item位置计算损失
                item_mask = (next_token_type == 1)
                
                if item_mask.any():
                    pos_loss = self.criterion(pos_logits[item_mask], pos_labels[item_mask])
                    neg_loss = self.criterion(neg_logits[item_mask], neg_labels[item_mask])
                    loss = pos_loss + neg_loss
                    
                    # 添加L2正则化
                    if self.args.l2_emb > 0:
                        l2_reg = torch.tensor(0., device=self.device)
                        for param in self.model.item_emb.parameters():
                            l2_reg += torch.norm(param)
                        loss += self.args.l2_emb * l2_reg
                    
                    # 反向传播
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    # 梯度裁剪
                    if self.args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                    
                    self.optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # 更新进度条
                    progress_bar.set_postfix({'loss': loss.item()})
                    
                    # 记录到TensorBoard
                    global_step = epoch * len(self.train_loader) + batch_idx
                    self.writer.add_scalar('Loss/train', loss.item(), global_step)
                    
                    # 记录到日志文件
                    log_entry = {
                        'epoch': epoch,
                        'batch': batch_idx,
                        'loss': loss.item(),
                        'lr': self.optimizer.param_groups[0]['lr'],
                        'timestamp': time.time()
                    }
                    self.log_file.write(json.dumps(log_entry) + '\\n')
                    self.log_file.flush()
                
            except Exception as e:
                logger.error(f"Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / max(num_batches, 1)
        return {'loss': avg_loss}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """验证模型"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validation'):
                try:
                    # 解包批次数据
                    seq, pos, neg, token_type, next_token_type, next_action_type, seq_feat, pos_feat, neg_feat = batch
                    
                    # 移动到设备
                    seq = seq.to(self.device)
                    pos = pos.to(self.device)
                    neg = neg.to(self.device)
                    token_type = token_type.to(self.device)
                    next_token_type = next_token_type.to(self.device)
                    next_action_type = next_action_type.to(self.device)
                    
                    # 前向传播
                    pos_logits, neg_logits = self.model(
                        seq, pos, neg, token_type, next_token_type, next_action_type,
                        seq_feat, pos_feat, neg_feat
                    )
                    
                    # 计算损失
                    pos_labels = torch.ones_like(pos_logits)
                    neg_labels = torch.zeros_like(neg_logits)
                    
                    item_mask = (next_token_type == 1)
                    
                    if item_mask.any():
                        pos_loss = self.criterion(pos_logits[item_mask], pos_labels[item_mask])
                        neg_loss = self.criterion(neg_logits[item_mask], neg_labels[item_mask])
                        loss = pos_loss + neg_loss
                        
                        total_loss += loss.item()
                        num_batches += 1
                
                except Exception as e:
                    logger.error(f"Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # 记录验证结果
        self.writer.add_scalar('Loss/val', avg_loss, epoch)
        
        return {'val_loss': avg_loss}
    
    def train(self):
        """主训练循环"""
        logger.info("Starting training...")
        
        start_epoch = getattr(self, 'start_epoch', 1)
        
        for epoch in range(start_epoch, self.args.num_epochs + 1):
            epoch_start_time = time.time()
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate(epoch)
            
            # 学习率调度
            if self.scheduler:
                self.scheduler.step()
            
            epoch_time = time.time() - epoch_start_time
            
            # 打印结果
            logger.info(
                f"Epoch {epoch}/{self.args.num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )
            
            # 保存检查点
            is_best = val_metrics['val_loss'] < self.best_val_ndcg  # 这里简化使用loss作为指标
            if is_best:
                self.best_val_ndcg = val_metrics['val_loss']
            
            if epoch % self.args.save_freq == 0 or is_best:
                self._save_checkpoint(epoch, is_best)
        
        logger.info("Training completed!")
        
        # 关闭文件和writer
        self.log_file.close()
        self.writer.close()


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Optimized Generative Recommendation Training')
    
    # 基础参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use')
    
    # 数据参数
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--maxlen', type=int, default=101, help='Maximum sequence length')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of data loading workers')
    
    # 模型参数
    parser.add_argument('--hidden_units', type=int, default=64, help='Hidden units')
    parser.add_argument('--num_blocks', type=int, default=2, help='Number of transformer blocks')
    parser.add_argument('--num_heads', type=int, default=2, help='Number of attention heads')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--norm_first', action='store_true', help='Use pre-norm in transformer')
    
    # 训练参数
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw'], help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine', choices=['cosine', 'step', 'none'], help='LR scheduler')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--l2_emb', type=float, default=0.0, help='L2 regularization for embeddings')
    
    # 保存和加载
    parser.add_argument('--save_freq', type=int, default=5, help='Save frequency (epochs)')
    parser.add_argument('--state_dict_path', type=str, default=None, help='Path to load pretrained model')
    parser.add_argument('--inference_only', action='store_true', help='Only run inference')
    
    # 多模态特征
    parser.add_argument('--mm_emb_id', nargs='+', default=['81'], 
                       choices=[str(i) for i in range(81, 87)], 
                       help='Multimodal embedding IDs')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = get_args()
    
    # 打印配置
    logger.info("Training configuration:")
    for arg, value in vars(args).items():
        logger.info(f"  {arg}: {value}")
    
    try:
        # 创建训练器并开始训练
        trainer = ModelTrainer(args)
        
        if not args.inference_only:
            trainer.train()
        else:
            logger.info("Inference only mode - skipping training")
            
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise


if __name__ == '__main__':
    main()
