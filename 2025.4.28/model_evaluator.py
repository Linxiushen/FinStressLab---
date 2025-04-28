import os
import logging
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
import json
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from concurrent.futures import ThreadPoolExecutor

from incremental_learner import IncrementalLearner
from model import HybridModel
from data_utils import TimeSeriesDataset

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluator')


class TimeSeriesEvaluator:
    """
    时间序列模型评估器，用于评估增量学习系统性能
    """
    
    def __init__(self, 
                 incremental_learner: IncrementalLearner,
                 evaluation_window: int = 30,  # 评估窗口（天）
                 metrics_history_size: int = 100,  # 历史指标数量
                 output_dir: str = "evaluation_results"):
        """
        初始化评估器
        
        Args:
            incremental_learner: 增量学习器实例
            evaluation_window: 评估窗口（天）
            metrics_history_size: 历史指标数量
            output_dir: 输出目录
        """
        self.learner = incremental_learner
        self.evaluation_window = evaluation_window
        self.metrics_history_size = metrics_history_size
        self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 初始化指标历史
        self.metrics_history = {
            "timestamp": [],
            "mse": [],
            "mae": [],
            "r2": [],
            "latency": [],
            "swap_times": [],
            "training_time": []
        }
        
        # 初始化符号指标
        self.symbol_metrics = {}
        for symbol in self.learner.symbols:
            self.symbol_metrics[symbol] = {
                "mse": [],
                "mae": [],
                "r2": [],
                "last_evaluation": None
            }
            
        logger.info(f"时间序列评估器已初始化，评估窗口: {evaluation_window} 天")
    
    def fetch_evaluation_data(self, symbol: str) -> pd.DataFrame:
        """
        获取评估数据
        
        Args:
            symbol: 交易符号
            
        Returns:
            评估数据DataFrame
        """
        # 计算日期范围
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.evaluation_window)
        
        # 从数据库获取数据
        with self.learner.data_processor.Session() as session:
            from sqlalchemy import text
            
            query = text("""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = :symbol 
                  AND timestamp BETWEEN :start_date AND :end_date
                ORDER BY timestamp ASC
            """)
            
            result = session.execute(query, {
                "symbol": symbol,
                "start_date": start_date,
                "end_date": end_date
            })
            
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                
        return df
    
    def evaluate_model(self, 
                      symbol: str, 
                      data: pd.DataFrame = None,
                      batch_size: int = 64,
                      confidence_interval: float = 0.95) -> Dict:
        """
        评估模型性能
        
        Args:
            symbol: 交易符号
            data: 评估数据（可选，如果为None则获取）
            batch_size: 批处理大小
            confidence_interval: 置信区间
            
        Returns:
            评估指标字典
        """
        if data is None:
            data = self.fetch_evaluation_data(symbol)
            
        if data.empty:
            logger.warning(f"符号 {symbol} 没有评估数据")
            return {}
            
        # 创建评估数据集
        dataset = TimeSeriesDataset(
            data=data,
            feature_columns=self.learner.data_processor.feature_columns,
            sequence_length=self.learner.sequence_length,
            forecast_horizon=self.learner.forecast_horizon,
            train=False,
            scale_data=True,
            scaler=self.learner.data_processor.scaler
        )
        
        if len(dataset) == 0:
            logger.warning(f"符号 {symbol} 的数据集为空")
            return {}
            
        # 创建数据加载器
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=torch.cuda.is_available()
        )
        
        # 获取活动模型
        model = self.learner.model_buffer.get_active_model()
        model.eval()
        
        # 预测和实际值
        y_true = []
        y_pred = []
        y_lower = []
        y_upper = []
        inference_times = []
        
        logger.info(f"开始评估符号 {symbol} 的模型...")
        
        # 模型评估
        with torch.no_grad():
            for x, y in dataloader:
                # 移动到设备
                x = x.to(model.lstm.weight_ih_l0.device)
                y = y.to(model.lstm.weight_ih_l0.device)
                
                # 时间测量
                start_time = time.time()
                
                # 使用不确定性估计进行预测
                outputs = model.predict_with_uncertainty(x)
                
                # 记录推理时间
                inference_time = (time.time() - start_time) * 1000  # 毫秒
                inference_times.append(inference_time)
                
                # 收集结果
                y_true.append(y.cpu().numpy())
                y_pred.append(outputs['prediction'].cpu().numpy())
                
                # 收集不确定性估计（如果有）
                if 'lower_bound' in outputs and 'upper_bound' in outputs:
                    y_lower.append(outputs['lower_bound'].cpu().numpy())
                    y_upper.append(outputs['upper_bound'].cpu().numpy())
        
        # 合并批次结果
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        
        if y_lower and y_upper:
            y_lower = np.concatenate(y_lower)
            y_upper = np.concatenate(y_upper)
            
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 计算平均推理延迟
        avg_latency = np.mean(inference_times)
        p95_latency = np.percentile(inference_times, 95)
        p99_latency = np.percentile(inference_times, 99)
        
        # 计算预测覆盖率（如果有不确定性估计）
        coverage = None
        if y_lower and y_upper:
            in_interval = np.logical_and(y_true >= y_lower, y_true <= y_upper)
            coverage = np.mean(in_interval)
        
        # 更新符号指标
        self.symbol_metrics[symbol]["mse"].append(mse)
        self.symbol_metrics[symbol]["mae"].append(mae)
        self.symbol_metrics[symbol]["r2"].append(r2)
        self.symbol_metrics[symbol]["last_evaluation"] = datetime.now()
        
        # 限制历史大小
        if len(self.symbol_metrics[symbol]["mse"]) > self.metrics_history_size:
            self.symbol_metrics[symbol]["mse"] = self.symbol_metrics[symbol]["mse"][-self.metrics_history_size:]
            self.symbol_metrics[symbol]["mae"] = self.symbol_metrics[symbol]["mae"][-self.metrics_history_size:]
            self.symbol_metrics[symbol]["r2"] = self.symbol_metrics[symbol]["r2"][-self.metrics_history_size:]
            
        logger.info(f"符号 {symbol} 评估完成。MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.6f}")
        
        # 构建结果
        results = {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "latency": {
                    "avg_ms": avg_latency,
                    "p95_ms": p95_latency,
                    "p99_ms": p99_latency
                }
            },
            "data_points": len(y_true)
        }
        
        if coverage is not None:
            results["metrics"]["uncertainty"] = {
                "coverage": coverage,
                "confidence_interval": confidence_interval
            }
            
        return results
    
    def evaluate_all_symbols(self, parallel: bool = True) -> Dict:
        """
        评估所有符号
        
        Args:
            parallel: 是否并行评估
            
        Returns:
            所有评估结果的字典
        """
        if parallel:
            # 并行评估
            with ThreadPoolExecutor(max_workers=min(8, len(self.learner.symbols))) as executor:
                futures = {executor.submit(self.evaluate_model, symbol): symbol for symbol in self.learner.symbols}
                results = {symbol: futures[future].result() for future, symbol in futures.items()}
        else:
            # 顺序评估
            results = {symbol: self.evaluate_model(symbol) for symbol in self.learner.symbols}
            
        # 计算整体指标
        all_mse = [results[symbol]["metrics"]["mse"] for symbol in self.learner.symbols if symbol in results and results[symbol]]
        all_mae = [results[symbol]["metrics"]["mae"] for symbol in self.learner.symbols if symbol in results and results[symbol]]
        all_r2 = [results[symbol]["metrics"]["r2"] for symbol in self.learner.symbols if symbol in results and results[symbol]]
        all_latency = [results[symbol]["metrics"]["latency"]["avg_ms"] for symbol in self.learner.symbols if symbol in results and results[symbol]]
        
        if all_mse:
            # 更新整体指标历史
            now = datetime.now().isoformat()
            self.metrics_history["timestamp"].append(now)
            self.metrics_history["mse"].append(np.mean(all_mse))
            self.metrics_history["mae"].append(np.mean(all_mae))
            self.metrics_history["r2"].append(np.mean(all_r2))
            self.metrics_history["latency"].append(np.mean(all_latency))
            self.metrics_history["swap_times"].append(np.mean(self.learner.metrics["swap_times"][-5:]) if self.learner.metrics["swap_times"] else 0)
            self.metrics_history["training_time"].append(self.learner.training_time)
            
            # 限制历史大小
            if len(self.metrics_history["timestamp"]) > self.metrics_history_size:
                for key in self.metrics_history:
                    self.metrics_history[key] = self.metrics_history[key][-self.metrics_history_size:]
                    
        # 计算整体指标
        overall_metrics = {
            "timestamp": datetime.now().isoformat(),
            "overall": {
                "mse": np.mean(all_mse) if all_mse else None,
                "mae": np.mean(all_mae) if all_mae else None,
                "r2": np.mean(all_r2) if all_r2 else None,
                "latency_ms": np.mean(all_latency) if all_latency else None
            },
            "symbols": results
        }
        
        # 保存评估结果
        self.save_evaluation_results(overall_metrics)
        
        return overall_metrics
    
    def save_evaluation_results(self, results: Dict) -> None:
        """
        保存评估结果
        
        Args:
            results: 评估结果
        """
        # 生成文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        
        # 保存JSON
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
            
        logger.info(f"评估结果已保存到 {filepath}")
    
    def plot_metrics_history(self, save_path: Optional[str] = None) -> None:
        """
        绘制指标历史图表
        
        Args:
            save_path: 保存路径（可选）
        """
        if not self.metrics_history["timestamp"]:
            logger.warning("没有历史指标可绘制")
            return
            
        # 转换时间戳
        timestamps = [datetime.fromisoformat(ts) for ts in self.metrics_history["timestamp"]]
        
        # 创建图表
        fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
        
        # 绘制错误指标
        axs[0].plot(timestamps, self.metrics_history["mse"], 'r-', label='MSE')
        axs[0].plot(timestamps, self.metrics_history["mae"], 'b-', label='MAE')
        axs[0].set_title('预测错误随时间变化')
        axs[0].set_ylabel('错误值')
        axs[0].legend()
        axs[0].grid(True)
        
        # 绘制R²
        axs[1].plot(timestamps, self.metrics_history["r2"], 'g-', label='R²')
        axs[1].set_title('R² 分数随时间变化')
        axs[1].set_ylabel('R²')
        axs[1].legend()
        axs[1].grid(True)
        
        # 绘制延迟和训练时间
        ax2 = axs[2].twinx()
        axs[2].plot(timestamps, self.metrics_history["latency"], 'c-', label='推理延迟 (ms)')
        axs[2].set_ylabel('延迟 (ms)')
        ax2.plot(timestamps, self.metrics_history["training_time"], 'm-', label='训练时间 (s)')
        ax2.set_ylabel('训练时间 (s)')
        axs[2].set_title('性能指标随时间变化')
        axs[2].set_xlabel('时间')
        axs[2].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axs[2].grid(True)
        
        plt.tight_layout()
        
        # 保存或显示
        if save_path:
            plt.savefig(save_path)
            logger.info(f"指标历史图表已保存到 {save_path}")
        else:
            save_path = os.path.join(self.output_dir, f"metrics_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
            plt.savefig(save_path)
            logger.info(f"指标历史图表已保存到 {save_path}")
    
    def start_periodic_evaluation(self, interval_hours: int = 24) -> None:
        """
        启动定期评估
        
        Args:
            interval_hours: 评估间隔（小时）
        """
        import threading
        
        def evaluation_job():
            while True:
                try:
                    logger.info(f"开始定期评估 (间隔: {interval_hours} 小时)...")
                    
                    # 评估所有符号
                    self.evaluate_all_symbols()
                    
                    # 绘制指标历史
                    self.plot_metrics_history()
                    
                    # 等待下一次评估
                    logger.info(f"评估完成，等待 {interval_hours} 小时进行下一次评估")
                    time.sleep(interval_hours * 3600)
                    
                except Exception as e:
                    logger.error(f"定期评估出错: {str(e)}")
                    time.sleep(3600)  # 出错后等待1小时
        
        # 启动评估线程
        thread = threading.Thread(target=evaluation_job, daemon=True)
        thread.start()
        
        logger.info(f"定期评估已启动，间隔: {interval_hours} 小时")
        
        return thread


if __name__ == "__main__":
    # 解析命令行参数
    import argparse
    
    parser = argparse.ArgumentParser(description="评估增量学习系统")
    parser.add_argument("--db_url", type=str, default="sqlite:///./timeseries_gpt.db", help="数据库URL")
    parser.add_argument("--symbols", type=str, nargs="+", default=["AAPL", "MSFT", "GOOG"], help="交易符号列表")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/incremental", help="检查点目录")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="输出目录")
    parser.add_argument("--interval", type=int, default=24, help="评估间隔（小时）")
    parser.add_argument("--once", action="store_true", help="只评估一次")
    
    args = parser.parse_args()
    
    # 创建增量学习器
    learner = IncrementalLearner(
        db_url=args.db_url,
        symbols=args.symbols,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # 如果检查点存在，加载状态
    if os.path.exists(args.checkpoint_dir):
        learner.load_state()
    
    # 创建评估器
    evaluator = TimeSeriesEvaluator(
        incremental_learner=learner,
        output_dir=args.output_dir
    )
    
    if args.once:
        # 评估一次
        results = evaluator.evaluate_all_symbols()
        evaluator.plot_metrics_history()
        print(json.dumps(results["overall"], indent=2))
    else:
        # 开始定期评估
        evaluator.start_periodic_evaluation(interval_hours=args.interval)
        
        # 保持主线程运行
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            logger.info("评估器已停止") 