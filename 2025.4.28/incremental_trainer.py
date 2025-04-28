import os
import logging
import time
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader, ConcatDataset
import threading
from datetime import datetime, timedelta

from incremental_learner import IncrementalLearner, ModelBuffer, DataProcessor
from model import HybridModel
from data_utils import TimeSeriesDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('incremental_trainer')


class AdaptiveIncrementalTrainer:
    """
    增强版增量训练器，支持自适应学习率和特殊市场事件处理
    """
    
    def __init__(self, 
                 incremental_learner: IncrementalLearner,
                 market_events_db_url: Optional[str] = None,
                 max_epochs_per_update: int = 5,
                 min_epochs_per_update: int = 1,
                 gradient_clip_val: float = 0.5,
                 accumulate_grad_batches: int = 1,
                 performance_mode: str = 'balanced'):
        """
        初始化增强型增量训练器
        
        Args:
            incremental_learner: 基础增量学习器实例
            market_events_db_url: 市场事件数据库URL（可选）
            max_epochs_per_update: 每次更新的最大训练轮数
            min_epochs_per_update: 每次更新的最小训练轮数
            gradient_clip_val: 梯度裁剪值
            accumulate_grad_batches: 梯度累积批次
            performance_mode: 性能模式 ('speed', 'accuracy', 'balanced')
        """
        self.learner = incremental_learner
        self.market_events_db_url = market_events_db_url
        self.max_epochs_per_update = max_epochs_per_update
        self.min_epochs_per_update = min_epochs_per_update
        self.gradient_clip_val = gradient_clip_val
        self.accumulate_grad_batches = accumulate_grad_batches
        self.performance_mode = performance_mode
        
        # 训练状态
        self.is_training = False
        self.current_symbol = None
        self.training_iterations = 0
        self.market_events = {}
        
        # 性能优化配置
        self.setup_performance_config()
        
        logger.info(f"增强型增量训练器已初始化，性能模式: {performance_mode}")
    
    def setup_performance_config(self):
        """根据性能模式设置配置"""
        if self.performance_mode == 'speed':
            # 优化速度模式
            self.precision = 16 if torch.cuda.is_available() else 32
            self.accumulate_grad_batches = 1
            self.min_epochs_per_update = 1
            self.max_epochs_per_update = 3
        elif self.performance_mode == 'accuracy':
            # 优化精度模式
            self.precision = 32
            self.accumulate_grad_batches = 4
            self.min_epochs_per_update = 3
            self.max_epochs_per_update = 10
        else:
            # 平衡模式
            self.precision = 16 if torch.cuda.is_available() else 32
            self.accumulate_grad_batches = 2
            self.min_epochs_per_update = 2
            self.max_epochs_per_update = 5
    
    def fetch_market_events(self, symbol: str, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        获取特定时间段内的市场事件
        
        Args:
            symbol: 交易符号
            start_date: 开始日期
            end_date: 结束日期
            
        Returns:
            市场事件列表
        """
        if not self.market_events_db_url:
            return []
            
        try:
            from sqlalchemy import create_engine, text
            
            engine = create_engine(self.market_events_db_url)
            
            query = text("""
                SELECT event_type, event_time, importance, description
                FROM market_events
                WHERE symbol = :symbol 
                  AND event_time BETWEEN :start_date AND :end_date
                ORDER BY importance DESC, event_time DESC
            """)
            
            with engine.connect() as conn:
                result = conn.execute(query, {
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date
                })
                
                events = [
                    {
                        "event_type": row[0],
                        "event_time": row[1],
                        "importance": row[2],
                        "description": row[3]
                    }
                    for row in result
                ]
                
            return events
        except Exception as e:
            logger.error(f"获取市场事件失败: {str(e)}")
            return []
    
    def adaptive_epochs_calculation(self, symbol: str, data_size: int) -> int:
        """
        基于数据大小和市场事件计算自适应训练轮数
        
        Args:
            symbol: 交易符号
            data_size: 数据大小
            
        Returns:
            训练轮数
        """
        # 基础轮数取决于数据大小
        if data_size < 100:
            base_epochs = self.min_epochs_per_update
        elif data_size < 500:
            base_epochs = int((self.min_epochs_per_update + self.max_epochs_per_update) / 2)
        else:
            base_epochs = self.max_epochs_per_update
            
        # 检查是否存在重要市场事件
        now = datetime.now()
        start_date = now - timedelta(days=2)
        
        if symbol in self.market_events and (now - self.market_events[symbol]["last_check"]).total_seconds() < 3600:
            # 使用缓存的事件
            events = self.market_events[symbol]["events"]
        else:
            # 获取新事件
            events = self.fetch_market_events(symbol, start_date, now)
            self.market_events[symbol] = {
                "events": events,
                "last_check": now
            }
        
        # 根据事件重要性增加训练轮数
        importance_boost = 0
        for event in events:
            if event["importance"] >= 8:  # 高重要性事件
                importance_boost = max(importance_boost, 2)
            elif event["importance"] >= 5:  # 中等重要性事件
                importance_boost = max(importance_boost, 1)
        
        adjusted_epochs = min(base_epochs + importance_boost, self.max_epochs_per_update)
        logger.info(f"符号 {symbol} 的自适应训练轮数: {adjusted_epochs} (基础: {base_epochs}, 事件提升: {importance_boost})")
        
        return adjusted_epochs
    
    def create_trainer(self, epochs: int) -> pl.Trainer:
        """
        创建PyTorch Lightning训练器
        
        Args:
            epochs: 训练轮数
            
        Returns:
            pl.Trainer实例
        """
        # 定义回调
        callbacks = [
            ModelCheckpoint(
                dirpath=self.learner.checkpoint_dir,
                filename='{epoch}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                mode='min'
            ),
            LearningRateMonitor(logging_interval='epoch')
        ]
        
        # 创建训练器
        trainer = pl.Trainer(
            max_epochs=epochs,
            callbacks=callbacks,
            gpus=1 if torch.cuda.is_available() else 0,
            precision=self.precision,
            gradient_clip_val=self.gradient_clip_val,
            accumulate_grad_batches=self.accumulate_grad_batches,
            logger=True
        )
        
        return trainer
    
    def train_on_new_data(self, symbol: str, dataset: TimeSeriesDataset) -> float:
        """
        在新数据上训练影子模型
        
        Args:
            symbol: 交易符号
            dataset: 时间序列数据集
            
        Returns:
            验证损失
        """
        # 跟踪当前训练的符号
        self.current_symbol = symbol
        self.is_training = True
        
        try:
            start_time = time.time()
            
            # 跳过太小的数据集
            if len(dataset) < self.learner.batch_size:
                logger.warning(f"数据集 {symbol} 太小，跳过训练")
                return float('inf')
            
            # 分割训练集/验证集
            train_size = int(len(dataset) * 0.8)
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                dataset, [train_size, val_size]
            )
            
            # 创建数据加载器
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.learner.batch_size,
                shuffle=True,
                num_workers=self.learner.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.learner.batch_size,
                shuffle=False,
                num_workers=self.learner.num_workers,
                pin_memory=True
            )
            
            # 计算自适应轮数
            epochs = self.adaptive_epochs_calculation(symbol, len(dataset))
            
            # 获取影子模型的Lightning模块
            pl_module = self.learner.pl_module
            
            # 创建训练器
            trainer = self.create_trainer(epochs)
            
            # 训练模型
            logger.info(f"在符号 {symbol} 的新数据上训练影子模型...")
            trainer.fit(pl_module, train_loader, val_loader)
            
            # 获取验证损失
            val_loss = pl_module.trainer.callback_metrics['val_loss'].item()
            
            # 更新指标
            self.learner.metrics['shadow_model_loss'] = val_loss
            self.learner.metrics['training_iterations'] += 1
            self.training_iterations += 1
            
            # 记录训练时间
            training_time = time.time() - start_time
            self.learner.training_time = training_time
            
            logger.info(f"训练完成，耗时 {training_time:.2f} 秒。验证损失: {val_loss:.6f}")
            
            return val_loss
            
        except Exception as e:
            logger.error(f"训练出错: {str(e)}")
            return float('inf')
        finally:
            self.is_training = False
            self.current_symbol = None
    
    def start_enhanced_training(self, continuous: bool = True) -> Optional[threading.Thread]:
        """
        启动增强型增量训练过程
        
        Args:
            continuous: 是否连续训练
            
        Returns:
            如果是连续模式，则返回训练线程
        """
        # 如果非连续模式，则只进行一次训练
        if not continuous:
            try:
                # 获取下一批数据
                symbol, dataset = self.learner.data_processor.get_next_batch()
                
                # 训练模型
                val_loss = self.train_on_new_data(symbol, dataset)
                
                # 检查是否应该交换模型
                self.check_and_swap_models(val_loss)
                
                return None
            except Exception as e:
                logger.error(f"非连续训练出错: {str(e)}")
                return None
        
        # 连续训练模式
        def continuous_training():
            while True:
                try:
                    # 获取下一批数据
                    symbol, dataset = self.learner.data_processor.get_next_batch()
                    
                    # 训练模型
                    val_loss = self.train_on_new_data(symbol, dataset)
                    
                    # 检查是否应该交换模型
                    self.check_and_swap_models(val_loss)
                    
                    # 短暂休眠以避免CPU过载
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"连续训练出错: {str(e)}")
                    time.sleep(10)  # 等待一段时间后重试
        
        # 启动连续训练线程
        thread = threading.Thread(target=continuous_training, daemon=True)
        thread.start()
        
        logger.info("增强型连续训练已启动")
        return thread
    
    def check_and_swap_models(self, val_loss: float) -> bool:
        """
        检查是否应该交换模型，如果是则执行交换
        
        Args:
            val_loss: 新模型的验证损失
            
        Returns:
            是否执行了交换
        """
        # 检查影子模型是否已准备好
        if not self.learner.model_buffer.is_shadow_ready():
            self.learner.model_buffer.mark_shadow_ready()
        
        # 计算改进
        time_since_swap = self.learner.model_buffer.time_since_last_swap()
        improvement = (self.learner.metrics['active_model_loss'] - val_loss) / self.learner.metrics['active_model_loss']
        
        if (improvement > self.learner.swap_threshold and 
            time_since_swap > self.learner.min_swap_interval):
            logger.info(f"模型改进: {improvement:.2%}。正在交换模型...")
            
            # 执行交换
            swap_time = self.learner.model_buffer.swap_models()
            self.learner.metrics['swap_times'].append(swap_time)
            self.learner.metrics['active_model_loss'] = val_loss
            
            # 交换后保存模型
            self.learner.save_state()
            
            logger.info(f"模型交换完成，耗时 {swap_time:.2f} ms")
            return True
        
        return False
    
    def get_training_status(self) -> Dict:
        """
        获取训练状态信息
        
        Returns:
            包含训练状态的字典
        """
        return {
            "is_training": self.is_training,
            "current_symbol": self.current_symbol,
            "training_iterations": self.training_iterations,
            "performance_mode": self.performance_mode,
            "precision": self.precision,
            "accumulate_grad_batches": self.accumulate_grad_batches,
            "min_epochs_per_update": self.min_epochs_per_update,
            "max_epochs_per_update": self.max_epochs_per_update,
            **self.learner.get_metrics()
        }


class HybridMemoryManager:
    """
    混合内存管理器，优化GPU和CPU内存使用
    """
    
    def __init__(self, 
                 gpu_memory_fraction: float = 0.8,
                 enable_cpu_offloading: bool = True,
                 prefetch_batches: int = 2):
        """
        初始化混合内存管理器
        
        Args:
            gpu_memory_fraction: GPU内存比例
            enable_cpu_offloading: 是否启用CPU卸载
            prefetch_batches: 预取批次数
        """
        self.gpu_memory_fraction = gpu_memory_fraction
        self.enable_cpu_offloading = enable_cpu_offloading
        self.prefetch_batches = prefetch_batches
        
        # 初始化
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
            
        self.cpu_tensors = {}
        self.pinned_memory = {}
        
        logger.info(f"混合内存管理器已初始化，设备: {self.device}, GPU内存比例: {gpu_memory_fraction}")
    
    def optimize_for_training(self):
        """为训练优化内存"""
        if torch.cuda.is_available():
            # 增加训练时的内存比例
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(min(0.95, self.gpu_memory_fraction + 0.1))
    
    def optimize_for_inference(self):
        """为推理优化内存"""
        if torch.cuda.is_available():
            # 恢复正常内存比例
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
    
    def offload_tensor(self, name: str, tensor: torch.Tensor):
        """
        将张量卸载到CPU
        
        Args:
            name: 张量名称
            tensor: 要卸载的张量
        """
        if not self.enable_cpu_offloading:
            return
            
        # 复制到CPU并保存
        self.cpu_tensors[name] = tensor.detach().cpu()
        
        # 创建固定内存以加速后续传输
        self.pinned_memory[name] = torch.zeros_like(
            self.cpu_tensors[name], 
            device='cpu', 
            pin_memory=True
        )
        self.pinned_memory[name].copy_(self.cpu_tensors[name])
    
    def prefetch_tensor(self, name: str):
        """
        预取张量到GPU
        
        Args:
            name: 张量名称
        """
        if name not in self.pinned_memory or not torch.cuda.is_available():
            return None
            
        # 将固定内存中的张量传输到GPU
        return self.pinned_memory[name].to(self.device, non_blocking=True)
    
    def get_memory_stats(self) -> Dict:
        """
        获取内存统计信息
        
        Returns:
            内存统计信息字典
        """
        stats = {
            "device": str(self.device),
            "cpu_tensors_count": len(self.cpu_tensors),
            "pinned_memory_count": len(self.pinned_memory)
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_memory_allocated": torch.cuda.memory_allocated() / (1024 ** 3),  # GB
                "gpu_memory_reserved": torch.cuda.memory_reserved() / (1024 ** 3),  # GB
                "gpu_memory_fraction": self.gpu_memory_fraction
            })
        
        return stats


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="运行增强型增量训练器")
    parser.add_argument("--db_url", type=str, default="sqlite:///./timeseries_gpt.db", help="数据库URL")
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "MSFT", "GOOG"], help="交易符号列表")
    parser.add_argument("--market_events_db", type=str, help="市场事件数据库URL")
    parser.add_argument("--performance_mode", type=str, choices=["speed", "accuracy", "balanced"], default="balanced", help="性能模式")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.8, help="GPU内存比例")
    
    args = parser.parse_args()
    
    # 创建基础增量学习器
    learner = IncrementalLearner(
        db_url=args.db_url,
        symbols=args.symbols,
        input_dim=5,
        hidden_dim=128,
        forecast_horizon=10,
        sequence_length=60,
        batch_size=64
    )
    
    # 创建增强型训练器
    trainer = AdaptiveIncrementalTrainer(
        incremental_learner=learner,
        market_events_db_url=args.market_events_db,
        performance_mode=args.performance_mode
    )
    
    # 创建内存管理器
    memory_manager = HybridMemoryManager(gpu_memory_fraction=args.gpu_memory_fraction)
    
    # 启动增量学习
    learner.start_incremental_learning()
    
    # 启动增强型训练
    trainer.start_enhanced_training(continuous=True)
    
    # 保持主线程运行
    try:
        while True:
            # 定期打印状态
            status = trainer.get_training_status()
            memory_stats = memory_manager.get_memory_stats()
            
            logger.info(f"训练状态: {status}")
            logger.info(f"内存统计: {memory_stats}")
            
            time.sleep(3600)  # 休眠一小时
    except KeyboardInterrupt:
        # 退出时保存状态
        learner.save_state()
        logger.info("增强型增量训练器已停止，状态已保存") 