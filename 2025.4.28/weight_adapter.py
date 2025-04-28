import numpy as np
import logging
from typing import List, Dict, Tuple, Union, Optional
import pandas as pd
from scipy.special import softmax
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger('weight_adapter')

class WeightAdapter:
    """
    模型权重动态分配器
    
    使用多种策略动态调整不同模型（ARIMA、LSTM、TimeGPT）的权重，
    基于历史预测误差、市场状态和强化学习反馈。
    """
    
    def __init__(self, 
                model_names: List[str] = ["ARIMA", "LSTM", "TimeGPT"],
                half_life_days: int = 7,
                lookback_window: int = 30,
                min_weight: float = 0.1,
                rag_correction_factor: float = 0.15,
                learning_rate: float = 0.01,
                reward_decay: float = 0.95):
        """
        初始化权重适配器
        
        Args:
            model_names: 模型名称列表
            half_life_days: 指数衰减的半衰期（天）
            lookback_window: 回溯窗口大小（天）
            min_weight: 最小权重限制
            rag_correction_factor: RAG检索增强修正因子
            learning_rate: 强化学习的学习率
            reward_decay: 奖励衰减因子
        """
        self.model_names = model_names
        self.num_models = len(model_names)
        self.half_life_days = half_life_days
        self.lookback_window = lookback_window
        self.min_weight = min_weight
        self.rag_correction_factor = rag_correction_factor
        self.learning_rate = learning_rate
        self.reward_decay = reward_decay
        
        # 初始化权重（默认均匀分布）
        self.current_weights = np.ones(self.num_models) / self.num_models
        
        # 初始化历史性能记录
        self.historical_weights = []
        self.historical_errors = []
        self.historical_rewards = []
        
        # 初始化强化学习状态
        self.rl_state = np.zeros((self.lookback_window, self.num_models))
        
        logger.info(f"权重适配器已初始化，模型: {model_names}")
    
    def calculate_weights(self, 
                         history_errors: np.ndarray, 
                         market_state: Optional[str] = None,
                         volatility: Optional[float] = None) -> np.ndarray:
        """
        计算模型权重
        
        Args:
            history_errors: 各模型最近N天预测误差矩阵，shape为(lookback_window, num_models)
            market_state: 当前市场状态（可选）
            volatility: 当前市场波动率（可选）
        
        Returns:
            各模型的权重数组，shape为(num_models,)
        """
        # 确保输入数据的形状是正确的
        if history_errors.shape[1] != self.num_models:
            raise ValueError(f"误差矩阵的列数 {history_errors.shape[1]} 与模型数量 {self.num_models} 不匹配")
            
        # 裁剪到回溯窗口大小
        if history_errors.shape[0] > self.lookback_window:
            history_errors = history_errors[-self.lookback_window:]
        
        # 更新强化学习状态
        self.rl_state = history_errors.copy()
        
        # 步骤1: 计算指数衰减权重
        weights = self._calculate_exp_decay_weights(history_errors)
        
        # 步骤2: 应用RAG检索增强修正
        weights = self._apply_rag_correction(weights, market_state, volatility)
        
        # 步骤3: 使用梯度归一化
        weights = self._apply_gradient_normalization(weights)
        
        # 步骤4: 确保所有权重都高于最小阈值
        weights = self._enforce_min_weights(weights)
        
        # 保存计算出的权重
        self.current_weights = weights
        self.historical_weights.append(weights)
        self.historical_errors.append(np.mean(history_errors, axis=0))
        
        logger.info(f"已计算新权重: {dict(zip(self.model_names, weights.round(3)))}")
        return weights
    
    def _calculate_exp_decay_weights(self, history_errors: np.ndarray) -> np.ndarray:
        """
        计算基于指数衰减的权重
        
        使用指数衰减加权平均，使最近的误差影响更大
        """
        # 计算指数衰减因子
        decay_factor = np.log(2) / self.half_life_days
        
        # 创建衰减权重向量（最近的日期权重最高）
        days = np.arange(history_errors.shape[0])
        decay_weights = np.exp(-decay_factor * (history_errors.shape[0] - 1 - days))
        decay_weights = decay_weights / decay_weights.sum()  # 归一化
        
        # 计算加权平均误差
        weighted_errors = np.zeros(self.num_models)
        for i in range(self.num_models):
            valid_indices = ~np.isnan(history_errors[:, i])
            if np.any(valid_indices):
                weighted_errors[i] = np.sum(history_errors[valid_indices, i] * decay_weights[valid_indices]) / np.sum(decay_weights[valid_indices])
            else:
                weighted_errors[i] = np.inf
        
        # 反转误差到权重（误差越小权重越大）
        # 使用softmax的负指数转换
        raw_weights = np.exp(-weighted_errors)
        
        # 处理任何可能的无穷大或NaN值
        raw_weights = np.nan_to_num(raw_weights, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 如果所有权重都是0，则使用均匀权重
        if np.sum(raw_weights) == 0:
            return np.ones(self.num_models) / self.num_models
            
        # 归一化权重
        return raw_weights / np.sum(raw_weights)
    
    def _apply_rag_correction(self, 
                            weights: np.ndarray, 
                            market_state: Optional[str] = None,
                            volatility: Optional[float] = None) -> np.ndarray:
        """
        应用RAG（检索增强生成）修正
        
        基于当前市场状态和历史性能，调整权重
        """
        # 复制原始权重
        corrected_weights = weights.copy()
        
        # 基于市场状态的权重调整
        if market_state:
            # 不同市场状态下的模型偏好
            state_prefs = {
                "uptrend": np.array([0.1, 0.3, 0.6]),     # 上涨趋势偏好TimeGPT
                "downtrend": np.array([0.3, 0.3, 0.4]),   # 下跌趋势平衡ARIMA和LSTM
                "sideways": np.array([0.4, 0.3, 0.3]),    # 横盘偏好ARIMA
                "volatile": np.array([0.2, 0.5, 0.3]),    # 高波动偏好LSTM
                "calm": np.array([0.3, 0.2, 0.5])         # 低波动偏好TimeGPT
            }
            
            if market_state in state_prefs:
                # 应用市场状态偏好修正
                state_correction = state_prefs[market_state]
                corrected_weights = (1 - self.rag_correction_factor) * corrected_weights + self.rag_correction_factor * state_correction
        
        # 基于波动率的权重调整
        if volatility is not None:
            # 高波动性时增加LSTM的权重，低波动性时增加TimeGPT的权重
            if volatility > 0.02:  # 高波动阈值
                volatility_factor = min(0.2, volatility)  # 限制最大影响
                lstm_index = self.model_names.index("LSTM") if "LSTM" in self.model_names else 1
                timegpt_index = self.model_names.index("TimeGPT") if "TimeGPT" in self.model_names else 2
                
                # 从其他模型转移权重到LSTM
                for i in range(self.num_models):
                    if i != lstm_index:
                        transfer = corrected_weights[i] * volatility_factor * 0.5
                        corrected_weights[i] -= transfer
                        corrected_weights[lstm_index] += transfer
            elif volatility < 0.01:  # 低波动阈值
                timegpt_index = self.model_names.index("TimeGPT") if "TimeGPT" in self.model_names else 2
                
                # 从其他模型小幅转移权重到TimeGPT
                for i in range(self.num_models):
                    if i != timegpt_index:
                        transfer = corrected_weights[i] * 0.05
                        corrected_weights[i] -= transfer
                        corrected_weights[timegpt_index] += transfer
        
        # 确保权重归一化
        return corrected_weights / np.sum(corrected_weights)
    
    def _apply_gradient_normalization(self, weights: np.ndarray) -> np.ndarray:
        """
        应用梯度归一化
        
        使用Gradient Rescaled (GRs)算法，防止权重变化过大
        """
        if not self.historical_weights:
            return weights
            
        # 获取上一个权重
        prev_weights = self.historical_weights[-1]
        
        # 计算权重的绝对变化
        weight_changes = np.abs(weights - prev_weights)
        
        # 如果变化太大，进行梯度缩放
        if np.max(weight_changes) > 0.2:  # 阈值可调整
            # 计算缩放因子，限制最大单步变化
            scaling_factor = 0.2 / np.max(weight_changes)
            
            # 应用缩放因子
            scaled_changes = weight_changes * scaling_factor
            directions = np.sign(weights - prev_weights)
            
            # 更新权重
            normalized_weights = prev_weights + directions * scaled_changes
            
            # 确保权重归一化
            normalized_weights = normalized_weights / np.sum(normalized_weights)
            return normalized_weights
        
        return weights
    
    def _enforce_min_weights(self, weights: np.ndarray) -> np.ndarray:
        """
        确保所有权重都高于最小阈值
        """
        # 复制权重
        adjusted_weights = weights.copy()
        
        # 找出低于最小阈值的权重
        below_min = adjusted_weights < self.min_weight
        
        if np.any(below_min):
            # 需要调整的总量
            deficit = np.sum(self.min_weight - adjusted_weights[below_min])
            
            # 从其他权重中按比例减少
            above_min = ~below_min
            if np.any(above_min):
                available = np.sum(adjusted_weights[above_min]) - self.min_weight * np.sum(above_min)
                if available > deficit:  # 确保有足够的权重可以重新分配
                    reduction_factor = deficit / np.sum(adjusted_weights[above_min])
                    adjusted_weights[above_min] *= (1 - reduction_factor)
                    
                    # 将低于阈值的权重设为最小值
                    adjusted_weights[below_min] = self.min_weight
                else:
                    # 如果无法满足最小权重要求，则使用均匀分布
                    return np.ones(self.num_models) / self.num_models
            else:
                # 如果所有权重都低于最小值，则使用均匀分布
                return np.ones(self.num_models) / self.num_models
        
        # 确保权重归一化
        return adjusted_weights / np.sum(adjusted_weights)
    
    def update_with_reward(self, reward: float, prediction_errors: np.ndarray) -> np.ndarray:
        """
        使用奖励更新权重 (强化学习方法)
        
        Args:
            reward: 当前预测的奖励值
            prediction_errors: 各模型的预测误差
            
        Returns:
            更新后的权重
        """
        if len(self.historical_weights) == 0:
            return self.current_weights
            
        # 记录奖励
        self.historical_rewards.append(reward)
        
        # 使用策略梯度更新
        # 正向奖励增加高性能模型的权重，负向奖励减少低性能模型的权重
        error_relative = prediction_errors / np.sum(prediction_errors)
        
        # 奖励的符号决定是增加还是减少权重
        reward_sign = 1 if reward > 0 else -1
        
        # 计算梯度更新，误差更小的模型获得更多增强
        inv_errors = 1.0 / (prediction_errors + 1e-6)
        gradient = reward_sign * (inv_errors / np.sum(inv_errors) - error_relative)
        
        # 应用学习率和更新
        updated_weights = self.current_weights + self.learning_rate * gradient
        
        # 确保权重归一化和最小阈值
        updated_weights = np.clip(updated_weights, self.min_weight, 1.0)
        updated_weights = updated_weights / np.sum(updated_weights)
        
        # 保存并返回更新后的权重
        self.current_weights = updated_weights
        self.historical_weights.append(updated_weights)
        
        logger.info(f"基于奖励 {reward:.4f} 更新权重: {dict(zip(self.model_names, updated_weights.round(3)))}")
        return updated_weights
    
    def get_weights_for_market_state(self, 
                                   market_state: str, 
                                   volatility: float,
                                   history_errors: Optional[np.ndarray] = None) -> np.ndarray:
        """
        获取特定市场状态下的最优权重
        
        Args:
            market_state: 当前市场状态
            volatility: 当前波动率
            history_errors: 可选的误差历史
            
        Returns:
            调整后的权重
        """
        if history_errors is not None:
            # 如果提供了误差历史，先计算基本权重
            base_weights = self.calculate_weights(history_errors)
        else:
            # 否则使用当前权重作为基础
            base_weights = self.current_weights
        
        # 应用市场状态修正
        return self._apply_rag_correction(base_weights, market_state, volatility)
    
    def get_performance_stats(self) -> Dict:
        """
        获取权重和性能统计
        
        Returns:
            包含性能统计信息的字典
        """
        stats = {
            "current_weights": dict(zip(self.model_names, self.current_weights.round(3))),
            "weight_stability": self._calculate_weight_stability(),
            "model_contribution": self._calculate_model_contribution()
        }
        
        if len(self.historical_errors) > 0:
            mean_errors = np.mean(self.historical_errors, axis=0)
            stats["average_errors"] = dict(zip(self.model_names, mean_errors.round(4)))
            
        if len(self.historical_rewards) > 0:
            stats["average_reward"] = np.mean(self.historical_rewards)
            stats["reward_trend"] = self._calculate_reward_trend()
            
        return stats
    
    def _calculate_weight_stability(self) -> Dict:
        """计算权重稳定性统计"""
        if len(self.historical_weights) < 2:
            return {"stability_score": 1.0}
            
        # 转换为numpy数组
        weights_array = np.array(self.historical_weights)
        
        # 计算每个模型权重的标准差
        weight_std = np.std(weights_array, axis=0)
        
        # 计算权重变化的平均幅度
        weight_changes = np.abs(np.diff(weights_array, axis=0))
        avg_change = np.mean(weight_changes)
        
        return {
            "stability_score": 1.0 - min(avg_change * 5, 0.9),  # 转换为0-1分数
            "std_per_model": dict(zip(self.model_names, weight_std.round(3))),
            "avg_change": avg_change.round(3)
        }
    
    def _calculate_model_contribution(self) -> Dict:
        """计算每个模型对整体性能的贡献"""
        if len(self.historical_weights) < 5 or len(self.historical_rewards) < 5:
            return {}
            
        # 将历史权重和奖励转换为numpy数组
        weights_array = np.array(self.historical_weights[-5:])
        rewards_array = np.array(self.historical_rewards[-5:])
        
        # 简单相关性估计（权重与奖励的相关性）
        correlations = {}
        for i, model in enumerate(self.model_names):
            # 计算当前模型权重与奖励的相关性
            corr = np.corrcoef(weights_array[:, i], rewards_array)[0, 1]
            correlations[model] = round(corr, 3) if not np.isnan(corr) else 0.0
            
        return {
            "weight_reward_correlation": correlations
        }
    
    def _calculate_reward_trend(self) -> float:
        """计算奖励趋势（正值表示上升趋势，负值表示下降趋势）"""
        if len(self.historical_rewards) < 5:
            return 0.0
            
        # 使用最近5个奖励计算趋势
        recent_rewards = self.historical_rewards[-5:]
        
        # 简单线性趋势（正斜率表示上升趋势）
        x = np.arange(len(recent_rewards))
        slope = np.polyfit(x, recent_rewards, 1)[0]
        
        return float(slope)


def calculate_weights(history_errors: np.ndarray,
                     market_state: Optional[str] = None,
                     volatility: Optional[float] = None) -> np.ndarray:
    """
    计算ARIMA/LSTM/TimeGPT的权重数组
    
    Args:
        history_errors: 各模型最近30天预测误差矩阵，shape为(30, 3)
        market_state: 当前市场状态（可选）
        volatility: 当前市场波动率（可选）
    
    Returns:
        ARIMA/LSTM/TimeGPT的权重数组，shape为(3,)
    """
    # 创建权重适配器实例
    adapter = WeightAdapter(
        model_names=["ARIMA", "LSTM", "TimeGPT"],
        half_life_days=7,
        lookback_window=30,
        min_weight=0.1,
        rag_correction_factor=0.15
    )
    
    # 计算权重
    weights = adapter.calculate_weights(
        history_errors=history_errors,
        market_state=market_state,
        volatility=volatility
    )
    
    return weights


if __name__ == "__main__":
    # 示例演示
    # 创建一个30天的随机误差矩阵 (30天 x 3个模型)
    # 值越低表示误差越小，模型性能越好
    np.random.seed(42)
    
    # 为ARIMA、LSTM和TimeGPT创建误差样本，体现不同模型特点
    arima_errors = np.random.normal(0.2, 0.1, 30)  # ARIMA保持较稳定的中等误差
    lstm_errors = np.random.normal(0.15, 0.15, 30)  # LSTM波动较大
    timegpt_errors = np.ones(30) * 0.3  # TimeGPT开始误差较大
    timegpt_errors[-15:] = np.random.normal(0.1, 0.05, 15)  # 后半段TimeGPT性能提升
    
    # 组合成误差矩阵
    error_matrix = np.column_stack([arima_errors, lstm_errors, timegpt_errors])
    
    # 计算权重
    weights = calculate_weights(
        history_errors=error_matrix,
        market_state="uptrend",
        volatility=0.015
    )
    
    print("模型权重:")
    print(f"ARIMA: {weights[0]:.3f}")
    print(f"LSTM: {weights[1]:.3f}")
    print(f"TimeGPT: {weights[2]:.3f}")
    
    # 创建一个长期运行的适配器示例
    adapter = WeightAdapter()
    
    # 模拟10次迭代
    print("\n模拟权重演变:")
    for i in range(10):
        # 模拟误差变化
        step_errors = error_matrix[np.random.choice(30, size=30, replace=True)]
        
        # 随机市场状态
        market_states = ["uptrend", "downtrend", "sideways", "volatile", "calm"]
        market_state = market_states[i % len(market_states)]
        
        # 随机波动率
        volatility = np.random.uniform(0.005, 0.03)
        
        # 计算权重
        weights = adapter.calculate_weights(step_errors, market_state, volatility)
        
        # 模拟收到奖励
        reward = np.random.normal(0.2, 0.3)  # 随机奖励
        adapter.update_with_reward(reward, np.mean(step_errors, axis=0))
        
        print(f"迭代 {i+1} - 市场状态: {market_state}, 波动率: {volatility:.3f}, 奖励: {reward:.3f}")
        print(f"  权重: ARIMA={weights[0]:.3f}, LSTM={weights[1]:.3f}, TimeGPT={weights[2]:.3f}")
    
    # 打印性能统计
    print("\n性能统计:")
    stats = adapter.get_performance_stats()
    import json
    print(json.dumps(stats, indent=2)) 