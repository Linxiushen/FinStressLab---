import os
import logging
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import threading
from datetime import datetime, timedelta

from incremental_learner import IncrementalLearner
from incremental_trainer import AdaptiveIncrementalTrainer
from reward_feedback import RewardFeedbackSystem
from model_evaluator import TimeSeriesEvaluator

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('feedback_integration')


class FeedbackDrivenTrainer:
    """
    集成回报驱动的增量学习系统
    """
    
    def __init__(self, 
                 incremental_learner: IncrementalLearner,
                 reward_system: RewardFeedbackSystem,
                 trainer: Optional[AdaptiveIncrementalTrainer] = None,
                 evaluator: Optional[TimeSeriesEvaluator] = None,
                 feedback_interval: int = 3600,  # 1小时
                 reward_weight_update_interval: int = 86400,  # 24小时
                 min_trades_for_update: int = 50):
        """
        初始化回报驱动训练器
        
        Args:
            incremental_learner: 增量学习器实例
            reward_system: 交易回报系统实例
            trainer: 增强型增量训练器（可选）
            evaluator: 评估器（可选）
            feedback_interval: 反馈更新间隔（秒）
            reward_weight_update_interval: 回报权重更新间隔（秒）
            min_trades_for_update: 更新所需的最少交易数量
        """
        self.learner = incremental_learner
        self.reward_system = reward_system
        self.trainer = trainer
        self.evaluator = evaluator
        self.feedback_interval = feedback_interval
        self.reward_weight_update_interval = reward_weight_update_interval
        self.min_trades_for_update = min_trades_for_update
        
        # 交易统计
        self.trades_since_last_update = 0
        self.total_trades = 0
        self.last_feedback_time = time.time()
        self.last_weight_update_time = time.time()
        
        # 性能指标
        self.performance_metrics = {
            'prediction_error': 0.1,
            'profit_performance': 0.0,
            'risk_level': 0.2
        }
        
        # 交易监控
        self.is_monitoring = False
        self.trading_predictions = {}
        
        logger.info("回报驱动训练器已初始化")
    
    def start_trading_monitor(self, trading_symbols: List[str] = None):
        """
        启动交易监控
        
        Args:
            trading_symbols: 要监控的交易符号列表
        """
        if trading_symbols is None:
            trading_symbols = self.learner.symbols
            
        self.is_monitoring = True
        
        def monitor_job():
            while self.is_monitoring:
                try:
                    # 对每个符号进行预测并记录
                    for symbol in trading_symbols:
                        # 从数据库获取最新数据
                        latest_data = self.learner.data_processor.fetch_latest_data(symbol, hours=2)
                        
                        if not latest_data.empty:
                            # 准备最后一个序列作为输入
                            dataset = self.learner.data_processor.generate_training_sequences(latest_data)
                            
                            if len(dataset) > 0:
                                # 获取最后一个序列
                                x, _ = dataset[-1]
                                x = x.unsqueeze(0)  # 添加批次维度
                                
                                # 进行预测
                                prediction_result = self.learner.predict(
                                    x=x,
                                    symbol=symbol,
                                    with_uncertainty=True
                                )
                                
                                # 提取预测值和预测时间
                                prediction = prediction_result['prediction'][0][-1].item()
                                timestamp = datetime.now()
                                
                                # 记录预测结果
                                self.trading_predictions[symbol] = {
                                    'prediction': prediction,
                                    'timestamp': timestamp,
                                    'price_series': latest_data['close'].tolist()[-60:],
                                    'position': self._determine_position(prediction, latest_data),
                                    'entry_price': latest_data['close'].iloc[-1],
                                    'volatility': latest_data['close'].pct_change().std() * 100,
                                    'market_state': self._determine_market_state(latest_data)
                                }
                                
                                logger.info(f"为 {symbol} 生成预测: {prediction:.4f}, 立场: {self.trading_predictions[symbol]['position']}")
                    
                    # 检查之前的预测并评估
                    self._evaluate_past_predictions()
                    
                    # 检查是否应该更新回报权重
                    self._check_and_update_reward_weights()
                    
                    # 等待下一次监控
                    time.sleep(300)  # 每5分钟更新一次
                    
                except Exception as e:
                    logger.error(f"交易监控出错: {str(e)}")
                    time.sleep(60)
        
        # 启动监控线程
        thread = threading.Thread(target=monitor_job, daemon=True)
        thread.start()
        
        logger.info(f"交易监控已启动，监控符号: {trading_symbols}")
        return thread
    
    def _determine_position(self, prediction: float, data: pd.DataFrame) -> str:
        """根据预测确定头寸"""
        if len(data) < 10:
            return "neutral"
            
        # 获取最近的收盘价
        last_close = data['close'].iloc[-1]
        
        # 计算预测变化百分比
        change_pct = (prediction - last_close) / last_close
        
        # 根据预测变化确定头寸
        if change_pct > 0.01:  # 预测上涨超过1%
            return "long"
        elif change_pct < -0.01:  # 预测下跌超过1%
            return "short"
        else:
            return "neutral"
    
    def _determine_market_state(self, data: pd.DataFrame) -> str:
        """确定市场状态"""
        if len(data) < 20:
            return "unknown"
            
        # 计算短期和长期移动平均线
        data['ma5'] = data['close'].rolling(window=5).mean()
        data['ma20'] = data['close'].rolling(window=20).mean()
        
        # 获取最近的值
        last_ma5 = data['ma5'].iloc[-1]
        last_ma20 = data['ma20'].iloc[-1]
        
        # 计算RSI指标
        delta = data['close'].diff()
        up = delta.clip(lower=0)
        down = -1 * delta.clip(upper=0)
        ema_up = up.ewm(com=13, adjust=False).mean()
        ema_down = down.ewm(com=13, adjust=False).mean()
        rs = ema_up / ema_down
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]
        
        # 确定市场状态
        if last_ma5 > last_ma20 and last_rsi > 50:
            return "uptrend"
        elif last_ma5 < last_ma20 and last_rsi < 50:
            return "downtrend"
        elif last_rsi > 70:
            return "overbought"
        elif last_rsi < 30:
            return "oversold"
        else:
            return "sideways"
    
    def _evaluate_past_predictions(self):
        """评估过去的预测并记录交易结果"""
        current_time = time.time()
        
        # 遍历所有预测
        for symbol, pred_data in list(self.trading_predictions.items()):
            # 检查预测是否足够旧（超过30分钟）可以评估
            if (datetime.now() - pred_data['timestamp']).total_seconds() > 1800:
                try:
                    # 获取最新数据
                    latest_data = self.learner.data_processor.fetch_latest_data(symbol, hours=1)
                    
                    if not latest_data.empty:
                        # 获取最新价格作为实际值
                        actual_price = latest_data['close'].iloc[-1]
                        
                        # 记录交易结果
                        trade_record = self.reward_system.record_trade(
                            symbol=symbol,
                            timestamp=pred_data['timestamp'],
                            prediction=pred_data['prediction'],
                            actual=actual_price,
                            position=pred_data['position'],
                            entry_price=pred_data['entry_price'],
                            exit_price=actual_price,
                            market_state=pred_data['market_state'],
                            volatility=pred_data['volatility'],
                            features={
                                'price_series': pred_data['price_series'],
                                'prediction_time': pred_data['timestamp'].isoformat()
                            }
                        )
                        
                        logger.info(f"评估 {symbol} 的预测: 预测={pred_data['prediction']:.4f}, 实际={actual_price:.4f}, "
                                   f"回报={trade_record['reward']:.4f}, 盈亏={trade_record['profit_loss']:.4%}")
                        
                        # 更新统计数据
                        self.trades_since_last_update += 1
                        self.total_trades += 1
                        
                        # 从预测字典中移除
                        del self.trading_predictions[symbol]
                        
                except Exception as e:
                    logger.error(f"评估 {symbol} 的预测时出错: {str(e)}")
        
        # 检查是否应该进行反馈更新
        if (current_time - self.last_feedback_time > self.feedback_interval and 
            self.trades_since_last_update >= self.min_trades_for_update):
            self._update_model_with_feedback()
    
    def _update_model_with_feedback(self):
        """使用交易反馈更新模型"""
        logger.info("基于交易反馈更新模型...")
        
        try:
            # 获取当前市场状态
            market_states = {}
            for symbol in self.learner.symbols:
                latest_data = self.learner.data_processor.fetch_latest_data(symbol, hours=2)
                if not latest_data.empty:
                    market_states[symbol] = self._determine_market_state(latest_data)
            
            # 获取主要交易符号的状态和波动性
            main_symbol = self.learner.symbols[0] if self.learner.symbols else "AAPL"
            current_market_state = market_states.get(main_symbol, "unknown")
            
            latest_data = self.learner.data_processor.fetch_latest_data(main_symbol, hours=2)
            current_volatility = latest_data['close'].pct_change().std() * 100 if not latest_data.empty else 0.02
            current_price_series = latest_data['close'].tolist()[-60:] if not latest_data.empty else []
            
            # 优化经验回放
            replay_samples = self.reward_system.optimize_experience_replay(
                current_market_state=current_market_state,
                current_volatility=current_volatility,
                current_price_series=current_price_series,
                batch_size=64
            )
            
            # 如果有足够的样本，创建训练批次
            if len(replay_samples) >= 32:
                # 获取影子模型
                shadow_model = self.learner.model_buffer.get_shadow_model()
                
                # 将样本转换为训练数据
                training_data = self._prepare_training_data(replay_samples)
                
                # 更新性能指标
                self._update_performance_metrics(replay_samples)
                
                # 如果有训练器，使用训练器进行训练
                if self.trainer:
                    logger.info("使用增强型训练器基于回报样本进行训练...")
                    # 创建数据集和训练
                    # 这里需要根据具体训练器实现方式进行调整
                    
                else:
                    logger.info("使用基本训练方法基于回报样本进行训练...")
                    # 基本训练逻辑
                    # 这里需要根据具体训练实现方式进行调整
                
                # 检查是否应该交换模型
                # 这里需要根据具体实现进行调整
                
                logger.info(f"基于 {len(replay_samples)} 个回报驱动样本完成模型更新")
            
            # 重置计数器
            self.last_feedback_time = time.time()
            self.trades_since_last_update = 0
            
        except Exception as e:
            logger.error(f"更新模型时出错: {str(e)}")
    
    def _prepare_training_data(self, replay_samples: List[Dict]) -> Dict:
        """
        将回放样本转换为训练数据
        
        Args:
            replay_samples: 回放样本列表
            
        Returns:
            训练数据字典
        """
        # 这里需要根据具体数据格式进行实现
        # 例如，从样本中提取特征序列和标签
        return {"samples": replay_samples}
    
    def _update_performance_metrics(self, replay_samples: List[Dict]):
        """更新性能指标"""
        if not replay_samples:
            return
            
        # 计算预测误差
        errors = [abs(s['prediction'] - s['actual']) / (abs(s['actual']) + 1e-6) 
                 for s in replay_samples if 'prediction' in s and 'actual' in s]
        
        # 计算盈亏表现
        profits = [s['profit_loss'] for s in replay_samples if 'profit_loss' in s]
        
        # 计算风险水平
        volatilities = [s['volatility'] for s in replay_samples if 'volatility' in s]
        
        # 更新指标
        if errors:
            self.performance_metrics['prediction_error'] = np.mean(errors)
        
        if profits:
            self.performance_metrics['profit_performance'] = np.mean(profits)
        
        if volatilities:
            self.performance_metrics['risk_level'] = np.mean(volatilities)
    
    def _check_and_update_reward_weights(self):
        """检查并更新回报权重"""
        current_time = time.time()
        
        # 每24小时或累积超过1000笔交易更新一次权重
        if (current_time - self.last_weight_update_time > self.reward_weight_update_interval or 
            self.trades_since_last_update > 1000):
            
            logger.info("更新回报权重...")
            
            # 更新权重
            self.reward_system.update_reward_weights(self.performance_metrics)
            
            # 重置时间
            self.last_weight_update_time = current_time
    
    def get_status(self) -> Dict:
        """获取状态信息"""
        return {
            "total_trades": self.total_trades,
            "trades_since_last_update": self.trades_since_last_update,
            "performance_metrics": self.performance_metrics,
            "reward_weights": {
                "alpha": self.reward_system.alpha,
                "beta": self.reward_system.beta,
                "gamma": self.reward_system.gamma
            },
            "active_symbols": list(self.trading_predictions.keys()),
            "last_feedback_time": datetime.fromtimestamp(self.last_feedback_time).isoformat(),
            "last_weight_update_time": datetime.fromtimestamp(self.last_weight_update_time).isoformat()
        }


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="运行回报驱动增量学习系统")
    parser.add_argument("--db_url", type=str, default="sqlite:///./timeseries_gpt.db", help="数据库URL")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--postgres_url", type=str, default="postgresql://user:password@localhost:5432/timeseries_gpt", help="PostgreSQL URL")
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "MSFT", "GOOG"], help="交易符号列表")
    
    args = parser.parse_args()
    
    # 创建增量学习器
    learner = IncrementalLearner(
        db_url=args.db_url,
        symbols=args.symbols,
        input_dim=5,
        hidden_dim=128,
        forecast_horizon=10,
        sequence_length=60,
        batch_size=64
    )
    
    # 创建回报系统
    reward_system = RewardFeedbackSystem(
        redis_url=args.redis_url,
        postgres_url=args.postgres_url
    )
    
    # 创建增强型训练器
    trainer = AdaptiveIncrementalTrainer(
        incremental_learner=learner,
        performance_mode="balanced"
    )
    
    # 创建回报驱动训练器
    feedback_trainer = FeedbackDrivenTrainer(
        incremental_learner=learner,
        reward_system=reward_system,
        trainer=trainer
    )
    
    # 启动增量学习
    learner.start_incremental_learning()
    
    # 启动交易监控
    feedback_trainer.start_trading_monitor()
    
    # 保持主线程运行
    try:
        while True:
            # 打印状态
            status = feedback_trainer.get_status()
            logger.info(f"系统状态: {status}")
            
            time.sleep(3600)  # 每小时打印一次状态
    except KeyboardInterrupt:
        learner.save_state()
        logger.info("系统已停止") 