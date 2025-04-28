import os
import logging
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
import redis
import json
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('reward_feedback')

# 声明数据库模型
Base = declarative_base()

class TradeRecord(Base):
    """交易记录模型"""
    __tablename__ = 'trade_records'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    timestamp = Column(DateTime, index=True)
    prediction = Column(Float)
    actual = Column(Float)
    position = Column(String(10))  # 'long', 'short', 'neutral'
    entry_price = Column(Float)
    exit_price = Column(Float)
    profit_loss = Column(Float)
    reward = Column(Float)
    market_state = Column(String(20))
    volatility = Column(Float)
    is_disaster = Column(Boolean, default=False)
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)


class DisasterSample(Base):
    """灾难样本模型"""
    __tablename__ = 'disaster_samples'
    
    id = Column(Integer, primary_key=True)
    symbol = Column(String(10), index=True)
    timestamp = Column(DateTime, index=True)
    description = Column(Text)
    severity = Column(Float)
    market_data = Column(JSON)
    features = Column(JSON)
    created_at = Column(DateTime, default=datetime.now)


class RewardFeedbackSystem:
    """
    交易回报驱动系统
    """
    
    def __init__(self, 
                 redis_url: str = "redis://localhost:6379/0",
                 postgres_url: str = "postgresql://user:password@localhost:5432/timeseries_gpt",
                 alpha: float = 0.4,
                 beta: float = 0.4,
                 gamma: float = 0.2,
                 short_term_capacity: int = 1000,
                 dtw_window_size: int = 20):
        """
        初始化交易回报驱动系统
        
        Args:
            redis_url: Redis连接URL (短期记忆)
            postgres_url: PostgreSQL连接URL (长期记忆和灾难样本)
            alpha: 预测准确率权重
            beta: 盈亏率权重
            gamma: 风险系数权重
            short_term_capacity: 短期记忆容量
            dtw_window_size: DTW相似度窗口大小
        """
        # 连接Redis (短期记忆)
        self.redis_client = redis.from_url(redis_url)
        self.short_term_key = "timeseries_gpt:short_term_memory"
        self.short_term_capacity = short_term_capacity
        
        # 连接PostgreSQL (长期记忆和灾难样本)
        self.postgres_url = postgres_url
        self.engine = create_engine(postgres_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # 回报函数参数
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        # DTW相似度匹配
        self.dtw_window_size = dtw_window_size
        
        logger.info("交易回报驱动系统已初始化")
    
    def calculate_reward(self, 
                        prediction: float, 
                        actual: float, 
                        profit_loss: float, 
                        volatility: float) -> float:
        """
        计算多目标回报
        
        Args:
            prediction: 预测值
            actual: 实际值
            profit_loss: 交易盈亏 (百分比)
            volatility: 波动性 (风险因子)
            
        Returns:
            计算得到的回报值
        """
        # 预测准确率 (使用1 - 相对误差)
        prediction_accuracy = max(0, 1 - abs(prediction - actual) / (abs(actual) + 1e-6))
        
        # 盈亏率 (归一化到[-1, 1]区间)
        profit_normalized = np.tanh(profit_loss)
        
        # 风险惩罚 (基于波动性)
        risk_penalty = min(1, volatility)
        
        # 计算加权回报
        reward = (self.alpha * prediction_accuracy + 
                 self.beta * profit_normalized - 
                 self.gamma * risk_penalty)
        
        return reward
    
    def record_trade(self, 
                    symbol: str,
                    timestamp: datetime,
                    prediction: float,
                    actual: float,
                    position: str,
                    entry_price: float,
                    exit_price: float,
                    market_state: str,
                    volatility: float,
                    features: Dict) -> Dict:
        """
        记录交易结果并计算回报
        
        Args:
            symbol: 交易符号
            timestamp: 交易时间
            prediction: 预测价格
            actual: 实际价格
            position: 持仓方向 ('long', 'short', 'neutral')
            entry_price: 入场价格
            exit_price: 出场价格
            market_state: 市场状态描述
            volatility: 市场波动性
            features: 交易相关特征
            
        Returns:
            包含交易记录和回报的字典
        """
        # 计算盈亏
        if position == 'long':
            profit_loss = (exit_price - entry_price) / entry_price
        elif position == 'short':
            profit_loss = (entry_price - exit_price) / entry_price
        else:
            profit_loss = 0.0
            
        # 计算回报
        reward = self.calculate_reward(
            prediction=prediction,
            actual=actual,
            profit_loss=profit_loss,
            volatility=volatility
        )
        
        # 创建交易记录
        trade_record = {
            "symbol": symbol,
            "timestamp": timestamp.isoformat(),
            "prediction": float(prediction),
            "actual": float(actual),
            "position": position,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "profit_loss": float(profit_loss),
            "reward": float(reward),
            "market_state": market_state,
            "volatility": float(volatility),
            "features": features,
            "is_disaster": False
        }
        
        # 保存到短期记忆 (Redis)
        self._add_to_short_term_memory(trade_record)
        
        # 保存到长期记忆 (PostgreSQL)
        self._add_to_long_term_memory(trade_record)
        
        # 检查是否为灾难样本
        if self._is_disaster_sample(trade_record):
            trade_record["is_disaster"] = True
            self._add_to_disaster_samples(trade_record)
        
        return trade_record
    
    def _add_to_short_term_memory(self, trade_record: Dict) -> None:
        """添加交易记录到短期记忆 (Redis)"""
        # 转换为JSON字符串
        record_json = json.dumps(trade_record)
        
        # 添加到Redis列表
        self.redis_client.lpush(self.short_term_key, record_json)
        
        # 保持列表大小不超过容量
        self.redis_client.ltrim(self.short_term_key, 0, self.short_term_capacity - 1)
    
    def _add_to_long_term_memory(self, trade_record: Dict) -> None:
        """添加交易记录到长期记忆 (PostgreSQL)"""
        with self.Session() as session:
            db_record = TradeRecord(
                symbol=trade_record["symbol"],
                timestamp=datetime.fromisoformat(trade_record["timestamp"]) if isinstance(trade_record["timestamp"], str) else trade_record["timestamp"],
                prediction=trade_record["prediction"],
                actual=trade_record["actual"],
                position=trade_record["position"],
                entry_price=trade_record["entry_price"],
                exit_price=trade_record["exit_price"],
                profit_loss=trade_record["profit_loss"],
                reward=trade_record["reward"],
                market_state=trade_record["market_state"],
                volatility=trade_record["volatility"],
                is_disaster=trade_record["is_disaster"],
                features=trade_record["features"]
            )
            session.add(db_record)
            session.commit()
    
    def _is_disaster_sample(self, trade_record: Dict) -> bool:
        """
        判断是否为灾难样本 (黑天鹅事件)
        
        当满足以下任一条件时，认为是灾难样本：
        1. 预测和实际值的相对误差超过50%
        2. 交易盈亏超过-15%
        3. 市场波动性超过历史90%分位数
        """
        prediction = trade_record["prediction"]
        actual = trade_record["actual"]
        profit_loss = trade_record["profit_loss"]
        volatility = trade_record["volatility"]
        
        # 计算相对误差
        relative_error = abs(prediction - actual) / (abs(actual) + 1e-6)
        
        # 判断条件
        if relative_error > 0.5:
            return True
        if profit_loss < -0.15:
            return True
        if volatility > self._get_volatility_threshold(trade_record["symbol"]):
            return True
            
        return False
    
    def _get_volatility_threshold(self, symbol: str) -> float:
        """获取波动性阈值 (90%分位数)"""
        try:
            with self.Session() as session:
                # 获取该符号的波动性历史数据
                result = session.query(TradeRecord.volatility).filter(
                    TradeRecord.symbol == symbol
                ).order_by(TradeRecord.volatility.desc()).limit(100).all()
                
                if result:
                    volatilities = [r[0] for r in result]
                    # 计算90%分位数
                    threshold = np.percentile(volatilities, 90)
                    return threshold
                else:
                    # 没有足够数据时返回默认值
                    return 0.03
        except Exception as e:
            logger.error(f"获取波动性阈值出错: {str(e)}")
            return 0.03
    
    def _add_to_disaster_samples(self, trade_record: Dict) -> None:
        """添加灾难样本"""
        with self.Session() as session:
            # 计算严重程度 (基于回报的负值)
            severity = -trade_record["reward"]
            
            disaster = DisasterSample(
                symbol=trade_record["symbol"],
                timestamp=datetime.fromisoformat(trade_record["timestamp"]) if isinstance(trade_record["timestamp"], str) else trade_record["timestamp"],
                description=f"灾难样本 - {trade_record['market_state']}",
                severity=severity,
                market_data={
                    "prediction": trade_record["prediction"],
                    "actual": trade_record["actual"],
                    "profit_loss": trade_record["profit_loss"],
                    "volatility": trade_record["volatility"]
                },
                features=trade_record["features"]
            )
            session.add(disaster)
            session.commit()
            
            logger.info(f"已添加灾难样本: {trade_record['symbol']} at {trade_record['timestamp']}, 严重程度: {severity:.4f}")
    
    def get_short_term_samples(self, limit: int = 100) -> List[Dict]:
        """
        获取短期记忆样本
        
        Args:
            limit: 返回样本数量上限
            
        Returns:
            短期记忆样本列表
        """
        # 从Redis获取最近的交易记录
        records_json = self.redis_client.lrange(self.short_term_key, 0, limit - 1)
        records = [json.loads(r) for r in records_json]
        return records
    
    def get_long_term_samples(self, 
                             symbol: Optional[str] = None, 
                             market_state: Optional[str] = None,
                             limit: int = 100) -> List[Dict]:
        """
        获取长期记忆样本
        
        Args:
            symbol: 交易符号过滤
            market_state: 市场状态过滤
            limit: 返回样本数量上限
            
        Returns:
            长期记忆样本列表
        """
        with self.Session() as session:
            query = session.query(TradeRecord)
            
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
                
            if market_state:
                query = query.filter(TradeRecord.market_state == market_state)
                
            # 按回报降序排序，获取最有价值的样本
            records = query.order_by(TradeRecord.reward.desc()).limit(limit).all()
            
            # 转换为字典列表
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "timestamp": r.timestamp.isoformat(),
                    "prediction": r.prediction,
                    "actual": r.actual,
                    "position": r.position,
                    "entry_price": r.entry_price,
                    "exit_price": r.exit_price,
                    "profit_loss": r.profit_loss,
                    "reward": r.reward,
                    "market_state": r.market_state,
                    "volatility": r.volatility,
                    "features": r.features,
                    "is_disaster": r.is_disaster
                }
                for r in records
            ]
    
    def get_disaster_samples(self, 
                            symbol: Optional[str] = None,
                            min_severity: float = 0.5,
                            limit: int = 50) -> List[Dict]:
        """
        获取灾难样本
        
        Args:
            symbol: 交易符号过滤
            min_severity: 最小严重程度
            limit: 返回样本数量上限
            
        Returns:
            灾难样本列表
        """
        with self.Session() as session:
            query = session.query(DisasterSample)
            
            if symbol:
                query = query.filter(DisasterSample.symbol == symbol)
                
            query = query.filter(DisasterSample.severity >= min_severity)
            
            # 按严重程度降序排序
            records = query.order_by(DisasterSample.severity.desc()).limit(limit).all()
            
            # 转换为字典列表
            return [
                {
                    "id": r.id,
                    "symbol": r.symbol,
                    "timestamp": r.timestamp.isoformat(),
                    "description": r.description,
                    "severity": r.severity,
                    "market_data": r.market_data,
                    "features": r.features
                }
                for r in records
            ]
    
    def find_similar_patterns_dtw(self, 
                                 time_series: np.ndarray, 
                                 symbol: Optional[str] = None,
                                 top_k: int = 5) -> List[Dict]:
        """
        使用DTW算法查找相似的历史模式
        
        Args:
            time_series: 输入时间序列
            symbol: 交易符号过滤
            top_k: 返回最相似的k个样本
            
        Returns:
            相似样本列表
        """
        # 确保输入是numpy数组
        time_series = np.asarray(time_series).flatten()
        
        with self.Session() as session:
            # 获取候选样本
            query = session.query(TradeRecord)
            if symbol:
                query = query.filter(TradeRecord.symbol == symbol)
                
            # 限制样本数量，避免计算量过大
            candidate_records = query.order_by(TradeRecord.timestamp.desc()).limit(1000).all()
            
            # 计算DTW距离
            distances = []
            for record in candidate_records:
                if 'price_series' in record.features:
                    # 从特征中提取价格序列
                    candidate_series = np.array(record.features['price_series'])
                    
                    # 如果序列长度不同，进行简单的插值调整
                    if len(candidate_series) != len(time_series):
                        from scipy.interpolate import interp1d
                        x_orig = np.linspace(0, 1, len(candidate_series))
                        x_new = np.linspace(0, 1, len(time_series))
                        f = interp1d(x_orig, candidate_series)
                        candidate_series = f(x_new)
                    
                    # 计算DTW距离
                    distance, _ = fastdtw(time_series, candidate_series, radius=self.dtw_window_size)
                    distances.append((record, distance))
            
            # 按距离排序
            distances.sort(key=lambda x: x[1])
            
            # 返回最相似的k个样本
            similar_samples = []
            for record, distance in distances[:top_k]:
                similar_samples.append({
                    "id": record.id,
                    "symbol": record.symbol,
                    "timestamp": record.timestamp.isoformat(),
                    "distance": distance,
                    "reward": record.reward,
                    "market_state": record.market_state,
                    "features": record.features
                })
                
            return similar_samples
    
    def optimize_experience_replay(self, 
                                  current_market_state: str,
                                  current_volatility: float,
                                  current_price_series: List[float],
                                  batch_size: int = 64) -> List[Dict]:
        """
        优化经验回放，构建训练批次
        
        使用三阶段回放策略：
        1. 短期记忆：最近的交易记录
        2. 长期记忆：相似市场状态下的成功交易
        3. 灾难样本：防止过度拟合和应对极端情况
        
        Args:
            current_market_state: 当前市场状态
            current_volatility: 当前波动性
            current_price_series: 当前价格序列
            batch_size: 批次大小
            
        Returns:
            优化后的训练样本
        """
        # 样本分配比例
        short_term_ratio = 0.5  # 短期记忆
        long_term_ratio = 0.3   # 长期记忆
        disaster_ratio = 0.2    # 灾难样本
        
        # 计算各类样本数量
        short_term_count = int(batch_size * short_term_ratio)
        long_term_count = int(batch_size * long_term_ratio)
        disaster_count = batch_size - short_term_count - long_term_count
        
        # 获取短期记忆样本
        short_term_samples = self.get_short_term_samples(limit=short_term_count)
        
        # 获取长期记忆样本 (相似市场状态)
        long_term_samples = self.get_long_term_samples(
            market_state=current_market_state,
            limit=long_term_count * 2  # 获取更多，后续用DTW筛选
        )
        
        # 如果有价格序列，使用DTW进一步筛选长期记忆样本
        if current_price_series and len(long_term_samples) > long_term_count:
            # 转换为numpy数组
            price_series = np.array(current_price_series)
            
            # 计算DTW距离
            distances = []
            for sample in long_term_samples:
                if 'price_series' in sample['features']:
                    sample_series = np.array(sample['features']['price_series'])
                    
                    # 长度调整
                    if len(sample_series) != len(price_series):
                        from scipy.interpolate import interp1d
                        x_orig = np.linspace(0, 1, len(sample_series))
                        x_new = np.linspace(0, 1, len(price_series))
                        f = interp1d(x_orig, sample_series)
                        sample_series = f(x_new)
                    
                    # 计算DTW距离
                    distance, _ = fastdtw(price_series, sample_series, radius=self.dtw_window_size)
                    distances.append((sample, distance))
            
            # 按距离排序并选择最相似的样本
            distances.sort(key=lambda x: x[1])
            long_term_samples = [item[0] for item in distances[:long_term_count]]
        else:
            # 如果没有价格序列或样本不足，直接截取
            long_term_samples = long_term_samples[:long_term_count]
        
        # 获取灾难样本 (高波动性条件)
        disaster_samples = self.get_disaster_samples(
            min_severity=current_volatility,  # 基于当前波动性设置阈值
            limit=disaster_count
        )
        
        # 合并所有样本
        all_samples = short_term_samples + long_term_samples + disaster_samples
        
        # 如果样本不足，填充随机样本
        if len(all_samples) < batch_size:
            additional_count = batch_size - len(all_samples)
            additional_samples = self.get_long_term_samples(limit=additional_count)
            all_samples.extend(additional_samples)
        
        # 随机打乱
        np.random.shuffle(all_samples)
        
        return all_samples[:batch_size]
    
    def update_reward_weights(self, 
                             performance_metrics: Dict[str, float]) -> None:
        """
        根据系统性能自动调整回报权重
        
        Args:
            performance_metrics: 性能指标字典
        """
        # 提取关键指标
        prediction_error = performance_metrics.get('prediction_error', 0.1)
        profit_performance = performance_metrics.get('profit_performance', 0.0)
        risk_level = performance_metrics.get('risk_level', 0.2)
        
        # 根据性能指标调整权重
        
        # 如果预测误差大，增加预测准确率权重
        if prediction_error > 0.15:
            self.alpha = min(0.6, self.alpha + 0.05)
        else:
            self.alpha = max(0.2, self.alpha - 0.02)
            
        # 如果利润表现差，增加盈亏率权重
        if profit_performance < 0.05:
            self.beta = min(0.6, self.beta + 0.05)
        else:
            self.beta = max(0.2, self.beta - 0.02)
            
        # 如果风险高，增加风险系数权重
        if risk_level > 0.3:
            self.gamma = min(0.5, self.gamma + 0.05)
        else:
            self.gamma = max(0.1, self.gamma - 0.02)
            
        # 归一化权重
        total = self.alpha + self.beta + self.gamma
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        logger.info(f"更新回报权重: α={self.alpha:.2f}, β={self.beta:.2f}, γ={self.gamma:.2f}")


if __name__ == "__main__":
    # 示例用法
    import argparse
    
    parser = argparse.ArgumentParser(description="运行交易回报驱动系统")
    parser.add_argument("--redis_url", type=str, default="redis://localhost:6379/0", help="Redis URL")
    parser.add_argument("--postgres_url", type=str, default="postgresql://user:password@localhost:5432/timeseries_gpt", help="PostgreSQL URL")
    parser.add_argument("--alpha", type=float, default=0.4, help="预测准确率权重")
    parser.add_argument("--beta", type=float, default=0.4, help="盈亏率权重")
    parser.add_argument("--gamma", type=float, default=0.2, help="风险系数权重")
    
    args = parser.parse_args()
    
    # 创建回报驱动系统
    reward_system = RewardFeedbackSystem(
        redis_url=args.redis_url,
        postgres_url=args.postgres_url,
        alpha=args.alpha,
        beta=args.beta,
        gamma=args.gamma
    )
    
    # 示例：记录一笔交易
    trade_record = reward_system.record_trade(
        symbol="AAPL",
        timestamp=datetime.now(),
        prediction=150.5,
        actual=152.3,
        position="long",
        entry_price=149.8,
        exit_price=152.3,
        market_state="uptrend",
        volatility=0.015,
        features={
            "price_series": [148.5, 149.2, 149.8, 150.5, 151.2, 152.3],
            "volume": 1250000,
            "macd": 0.5,
            "rsi": 65
        }
    )
    
    print(f"交易记录: {trade_record}")
    print(f"计算的回报: {trade_record['reward']:.4f}")
    
    # 示例：优化经验回放
    replay_samples = reward_system.optimize_experience_replay(
        current_market_state="uptrend",
        current_volatility=0.02,
        current_price_series=[145.6, 146.8, 148.2, 149.5, 150.3],
        batch_size=32
    )
    
    print(f"回放样本数量: {len(replay_samples)}")
    
    # 示例：更新回报权重
    reward_system.update_reward_weights({
        'prediction_error': 0.12,
        'profit_performance': 0.08,
        'risk_level': 0.25
    }) 