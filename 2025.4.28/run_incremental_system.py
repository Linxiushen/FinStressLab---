#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TimeSeriesGPT增量学习系统启动脚本
"""

import os
import logging
import argparse
import time
from datetime import datetime

from incremental_learner import IncrementalLearner, DataProcessor, ParallelDataProcessor
from incremental_trainer import AdaptiveIncrementalTrainer, HybridMemoryManager
from model_evaluator import TimeSeriesEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"incremental_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('incremental_system')


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="运行TimeSeriesGPT增量学习系统")
    
    # 数据库配置
    parser.add_argument("--db_url", type=str, default=os.environ.get("DATABASE_URL", "sqlite:///./timeseries_gpt.db"), 
                        help="TimescaleDB数据库URL")
    parser.add_argument("--market_events_db", type=str, 
                        help="市场事件数据库URL（可选）")
    
    # 交易符号
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"], 
                        help="交易符号列表")
    
    # 模型和训练配置
    parser.add_argument("--input_dim", type=int, default=5, help="输入维度")
    parser.add_argument("--hidden_dim", type=int, default=128, help="隐藏层维度")
    parser.add_argument("--forecast_horizon", type=int, default=10, help="预测时间范围")
    parser.add_argument("--sequence_length", type=int, default=60, help="序列长度")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--swap_threshold", type=float, default=0.05, help="模型交换阈值")
    parser.add_argument("--min_swap_interval", type=int, default=3600, help="最小交换间隔（秒）")
    
    # 性能配置
    parser.add_argument("--performance_mode", type=str, choices=["speed", "accuracy", "balanced"], 
                        default="balanced", help="性能模式")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.8, help="GPU内存比例")
    parser.add_argument("--enable_cpu_offloading", action="store_true", help="启用CPU卸载")
    
    # 评估配置
    parser.add_argument("--evaluation_interval", type=int, default=24, help="评估间隔（小时）")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/incremental", help="检查点目录")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="评估结果输出目录")
    
    # 启动选项
    parser.add_argument("--skip_initial_training", action="store_true", help="跳过初始训练")
    parser.add_argument("--load_checkpoint", action="store_true", help="加载现有检查点")
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析参数
    args = parse_arguments()
    
    logger.info("启动TimeSeriesGPT增量学习系统...")
    logger.info(f"配置: {args}")
    
    # 创建检查点目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. 初始化内存管理器
    memory_manager = HybridMemoryManager(
        gpu_memory_fraction=args.gpu_memory_fraction,
        enable_cpu_offloading=args.enable_cpu_offloading
    )
    
    # 2. 初始化增量学习器
    learner = IncrementalLearner(
        db_url=args.db_url,
        symbols=args.symbols,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        forecast_horizon=args.forecast_horizon,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        swap_threshold=args.swap_threshold,
        min_swap_interval=args.min_swap_interval,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 3. 如果指定，加载现有检查点
    if args.load_checkpoint and os.path.exists(args.checkpoint_dir):
        logger.info(f"从 {args.checkpoint_dir} 加载检查点...")
        learner.load_state()
        
    # 4. 初始化增强型训练器
    trainer = AdaptiveIncrementalTrainer(
        incremental_learner=learner,
        market_events_db_url=args.market_events_db,
        performance_mode=args.performance_mode
    )
    
    # 5. 初始化评估器
    evaluator = TimeSeriesEvaluator(
        incremental_learner=learner,
        output_dir=args.output_dir
    )
    
    # 6. 启动系统组件
    
    # 除非指定跳过，否则启动增量学习
    if not args.skip_initial_training:
        logger.info("启动增量学习...")
        learner.start_incremental_learning()
    else:
        logger.info("跳过初始训练，仅启动数据处理器...")
        learner.data_processor.start_data_fetcher(args.symbols)
    
    # 启动增强型训练
    logger.info("启动增强型训练...")
    training_thread = trainer.start_enhanced_training(continuous=True)
    
    # 启动定期评估
    logger.info(f"启动定期评估（间隔：{args.evaluation_interval}小时）...")
    evaluation_thread = evaluator.start_periodic_evaluation(interval_hours=args.evaluation_interval)
    
    # 7. 运行初始评估
    logger.info("运行初始系统评估...")
    evaluation_results = evaluator.evaluate_all_symbols()
    logger.info(f"初始评估结果: {evaluation_results['overall']}")
    evaluator.plot_metrics_history()
    
    # 8. 保持主线程运行
    try:
        logger.info("增量学习系统已启动并正在运行...")
        
        while True:
            # 每小时记录一次状态
            time.sleep(3600)
            status = trainer.get_training_status()
            memory_stats = memory_manager.get_memory_stats()
            
            logger.info(f"系统状态: 训练中: {status['is_training']}, "
                       f"当前符号: {status['current_symbol']}, "
                       f"训练迭代: {status['training_iterations']}")
            logger.info(f"模型性能: 活动模型损失: {status['active_model_loss']:.6f}, "
                       f"影子模型损失: {status['shadow_model_loss']:.6f}")
            logger.info(f"内存状态: {memory_stats}")
    
    except KeyboardInterrupt:
        logger.info("接收到中断信号，正在保存状态...")
        learner.save_state()
        logger.info("增量学习系统已停止，状态已保存")


if __name__ == "__main__":
    main() 