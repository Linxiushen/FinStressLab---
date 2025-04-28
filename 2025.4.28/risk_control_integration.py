import os
import logging
import argparse
import json
from typing import Dict, List, Optional
from datetime import datetime

from risk_control import RiskControlSystem
from incremental_learner import IncrementalLearner
from model import HybridModel
from data_utils import TimeSeriesRetriever
from weight_adapter import WeightAdapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"risk_control_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('risk_control_integration')


class RiskControlIntegration:
    """
    风险控制系统集成器
    
    将风险控制系统与增量学习系统集成，提供自动熔断、模型回滚和异常检测功能。
    """
    
    def __init__(self, 
                incremental_learner: IncrementalLearner,
                risk_control: RiskControlSystem,
                weight_adapter: Optional[WeightAdapter] = None,
                notification_webhook: Optional[str] = None,
                max_rollbacks_per_day: int = 3,
                safe_mode: bool = True):
        """
        初始化风险控制集成
        
        Args:
            incremental_learner: 增量学习器实例
            risk_control: 风险控制系统实例
            weight_adapter: 权重适配器实例（可选）
            notification_webhook: 通知WebHook URL（可选）
            max_rollbacks_per_day: 每日最大回滚次数
            safe_mode: 是否启用安全模式（默认为True）
        """
        self.learner = incremental_learner
        self.risk_control = risk_control
        self.weight_adapter = weight_adapter
        self.notification_webhook = notification_webhook
        self.max_rollbacks_per_day = max_rollbacks_per_day
        self.safe_mode = safe_mode
        
        # 初始化回滚计数器
        self.rollback_count = 0
        self.last_rollback_time = None
        
        # 设置回调函数
        self.risk_control.on_circuit_breaker_triggered = self._on_circuit_breaker
        self.risk_control.on_model_rollback = self._on_model_rollback
        self.risk_control.on_risk_report_generated = self._on_risk_report
        
        # 创建初始模型快照
        self._create_initial_snapshot()
        
        logger.info("风险控制集成已初始化")
    
    def _create_initial_snapshot(self):
        """创建初始模型快照"""
        try:
            # 获取当前模型
            model = self.learner.model_buffer.get_active_model()
            
            if model is not None:
                # 创建模型快照目录
                snapshot_dir = os.path.join(self.risk_control.models_dir, "initial")
                os.makedirs(snapshot_dir, exist_ok=True)
                
                # 保存模型
                snapshot_path = os.path.join(snapshot_dir, f"model_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")
                torch.save(model.state_dict(), snapshot_path)
                
                # 获取性能指标
                performance_metrics = {
                    "timestamp": datetime.now().isoformat(),
                    "version": "initial",
                    "symbols": self.learner.symbols
                }
                
                # 创建快照记录
                self.risk_control.create_model_snapshot(
                    model_path=snapshot_path,
                    performance_metrics=performance_metrics,
                    is_stable=True,
                    description="初始模型快照"
                )
                
                logger.info(f"已创建初始模型快照: {snapshot_path}")
            else:
                logger.warning("无法获取活跃模型，跳过初始快照创建")
        
        except Exception as e:
            logger.error(f"创建初始模型快照失败: {str(e)}")
    
    def start(self):
        """启动风险控制集成"""
        # 确保增量学习系统已启动
        if not self.learner.is_running:
            logger.warning("增量学习系统未运行，尝试启动...")
            self.learner.start_incremental_learning()
        
        # 启动风险控制监控
        self.risk_control.start_monitoring()
        
        # 设置定期快照
        self._setup_snapshot_scheduler()
        
        logger.info("风险控制集成已启动")
    
    def stop(self):
        """停止风险控制集成"""
        # 停止风险控制监控
        self.risk_control.stop_monitoring()
        
        logger.info("风险控制集成已停止")
    
    def _setup_snapshot_scheduler(self):
        """设置定期快照调度器"""
        import threading
        import time
        
        def snapshot_job():
            while self.risk_control.monitoring_active:
                try:
                    # 创建模型快照
                    self._create_periodic_snapshot()
                    
                    # 每天执行一次
                    time.sleep(86400)  # 24小时
                    
                except Exception as e:
                    logger.error(f"定期快照出错: {str(e)}")
                    time.sleep(3600)  # 出错后等待1小时再重试
        
        # 启动快照线程
        snapshot_thread = threading.Thread(target=snapshot_job, daemon=True)
        snapshot_thread.start()
        
        logger.info("定期快照调度器已启动")
    
    def _create_periodic_snapshot(self):
        """创建定期模型快照"""
        try:
            # 获取当前模型
            model = self.learner.model_buffer.get_active_model()
            
            if model is not None:
                # 评估模型性能
                is_stable = self._evaluate_model_stability()
                
                # 创建模型快照目录
                snapshot_dir = os.path.join(self.risk_control.models_dir, "periodic")
                os.makedirs(snapshot_dir, exist_ok=True)
                
                # 保存模型
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                snapshot_path = os.path.join(snapshot_dir, f"model_snapshot_{timestamp}.pt")
                torch.save(model.state_dict(), snapshot_path)
                
                # 获取性能指标
                performance_metrics = self._get_model_performance_metrics()
                
                # 创建快照记录
                self.risk_control.create_model_snapshot(
                    model_path=snapshot_path,
                    performance_metrics=performance_metrics,
                    is_stable=is_stable,
                    description="定期模型快照"
                )
                
                logger.info(f"已创建定期模型快照: {snapshot_path}, 稳定性: {is_stable}")
        
        except Exception as e:
            logger.error(f"创建定期模型快照失败: {str(e)}")
    
    def _evaluate_model_stability(self) -> bool:
        """评估模型稳定性"""
        try:
            # 获取最近的预测和实际值
            with self.risk_control.Session() as session:
                query = """
                    SELECT symbol, prediction, actual, timestamp 
                    FROM predictions
                    WHERE timestamp > NOW() - INTERVAL '3 days'
                    ORDER BY symbol, timestamp DESC
                """
                
                results = pd.read_sql(query, session.bind)
                
                # 计算相对误差
                results['rel_error'] = abs(results['prediction'] - results['actual']) / (abs(results['actual']) + 1e-6)
                
                # 计算平均误差
                avg_error = results['rel_error'].mean()
                
                # 如果平均误差小于阈值，认为模型稳定
                is_stable = avg_error < self.risk_control.error_threshold * 0.7
                
                return is_stable
                
        except Exception as e:
            logger.error(f"评估模型稳定性失败: {str(e)}")
            # 默认为稳定
            return True
    
    def _get_model_performance_metrics(self) -> Dict:
        """获取模型性能指标"""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "symbols": self.learner.symbols
        }
        
        # 如果有权重适配器，添加当前权重
        if self.weight_adapter:
            metrics["model_weights"] = self.weight_adapter.get_performance_stats()
        
        # 添加其他性能指标
        # TODO: 从评估器获取更多指标
        
        return metrics
    
    def _on_circuit_breaker(self, symbol: str, event_id: int):
        """熔断触发回调"""
        logger.warning(f"熔断已触发: 交易对 {symbol}, 事件 ID {event_id}")
        
        # 通知相关人员
        self._send_notification(
            title=f"交易熔断触发: {symbol}",
            message=f"交易对 {symbol} 已触发熔断。连续预测误差超过阈值。事件ID: {event_id}",
            level="critical"
        )
        
        # 暂停该交易对的增量学习
        self._pause_symbol_learning(symbol)
    
    def _on_model_rollback(self, symbol: str, snapshot_id: int):
        """模型回滚回调"""
        logger.warning(f"模型已回滚: 交易对 {symbol}, 快照 ID {snapshot_id}")
        
        # 更新回滚计数
        self.rollback_count += 1
        self.last_rollback_time = datetime.now()
        
        # 通知相关人员
        self._send_notification(
            title=f"模型已回滚: {symbol}",
            message=f"交易对 {symbol} 的模型已回滚到稳定版本。快照ID: {snapshot_id}",
            level="high"
        )
        
        # 重置权重适配器
        if self.weight_adapter:
            try:
                # 重置当前权重为均匀分布
                uniform_weights = np.ones(len(self.weight_adapter.model_names)) / len(self.weight_adapter.model_names)
                self.weight_adapter.current_weights = uniform_weights
                
                logger.info(f"已重置权重适配器权重为均匀分布")
            except Exception as e:
                logger.error(f"重置权重适配器失败: {str(e)}")
    
    def _on_risk_report(self, symbol: str, event_id: int, report_path: str):
        """风险报告生成回调"""
        logger.info(f"风险报告已生成: 交易对 {symbol}, 事件 ID {event_id}, 路径 {report_path}")
        
        # 通知相关人员
        self._send_notification(
            title=f"风险报告: {symbol}",
            message=f"交易对 {symbol} 的风险报告已生成。事件ID: {event_id}",
            level="medium",
            attachment=report_path
        )
    
    def _send_notification(self, title: str, message: str, level: str = "info", attachment: Optional[str] = None):
        """发送通知"""
        if not self.notification_webhook:
            return
            
        try:
            import requests
            
            # 构造通知数据
            data = {
                "title": title,
                "message": message,
                "level": level,
                "timestamp": datetime.now().isoformat()
            }
            
            # 如果有附件
            if attachment and os.path.exists(attachment):
                with open(attachment, 'r', encoding='utf-8') as f:
                    try:
                        # 尝试解析JSON
                        data["attachment"] = json.load(f)
                    except json.JSONDecodeError:
                        # 如果不是JSON，添加文本内容
                        data["attachment_text"] = f.read()
            
            # 发送通知
            response = requests.post(
                self.notification_webhook,
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=10
            )
            
            if response.status_code >= 400:
                logger.error(f"发送通知失败: HTTP {response.status_code}, {response.text}")
                
        except Exception as e:
            logger.error(f"发送通知出错: {str(e)}")
    
    def _pause_symbol_learning(self, symbol: str):
        """暂停指定交易对的增量学习"""
        try:
            # 将交易对从学习队列中移除
            if symbol in self.learner.symbols:
                # 保存当前学习状态
                self.learner.save_state()
                
                logger.info(f"已暂停 {symbol} 的增量学习")
        except Exception as e:
            logger.error(f"暂停增量学习失败: {str(e)}")
    
    def can_perform_rollback(self) -> bool:
        """检查是否可以执行回滚"""
        # 检查今日回滚次数
        today = datetime.now().date()
        if self.last_rollback_time and self.last_rollback_time.date() == today:
            # 如果今天已经回滚超过最大次数，禁止回滚
            if self.rollback_count >= self.max_rollbacks_per_day:
                logger.warning(f"今日回滚次数 ({self.rollback_count}) 已达到上限 ({self.max_rollbacks_per_day})，禁止回滚")
                return False
        else:
            # 新的一天，重置计数器
            self.rollback_count = 0
            
        return True
    
    def reset_circuit_breaker(self, symbol: str, force: bool = False):
        """
        重置熔断，恢复交易
        
        Args:
            symbol: 交易对
            force: 是否强制重置，即使在安全模式下
        """
        if self.safe_mode and not force:
            logger.warning(f"安全模式下禁止自动重置熔断，请使用force=True强制重置")
            return False
            
        # 重置熔断状态
        self.risk_control.reset_trading_freeze(symbol)
        
        # 解决所有未解决的风险事件
        events = self.risk_control.get_active_risk_events(resolved=False)
        for event in events:
            if event["symbol"] == symbol:
                self.risk_control.resolve_risk_event(
                    event_id=event["id"],
                    resolution_notes="系统重置熔断，恢复交易"
                )
        
        logger.info(f"已重置 {symbol} 的熔断状态")
        return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="运行风险控制集成")
    
    # 数据库配置
    parser.add_argument("--db_url", type=str, default=os.environ.get("DATABASE_URL", "sqlite:///./timeseries_gpt.db"), 
                        help="主数据库URL")
    parser.add_argument("--timescale_url", type=str, default=os.environ.get("TIMESCALE_URL", "postgresql://user:password@localhost:5432/timescale"),
                        help="TimescaleDB URL")
    
    # 模型配置
    parser.add_argument("--models_dir", type=str, default="./models/snapshots", help="模型快照目录")
    
    # 风控配置
    parser.add_argument("--error_threshold", type=float, default=0.2, help="误差阈值")
    parser.add_argument("--consecutive_errors", type=int, default=3, help="连续误差次数限制")
    parser.add_argument("--monitoring_interval", type=int, default=300, help="监控间隔（秒）")
    parser.add_argument("--max_rollbacks", type=int, default=3, help="每日最大回滚次数")
    
    # 集成选项
    parser.add_argument("--safe_mode", action="store_true", help="启用安全模式")
    parser.add_argument("--webhook_url", type=str, help="通知WebHook URL")
    parser.add_argument("--deepseek_api_key", type=str, help="DeepSeek API密钥")
    
    args = parser.parse_args()
    
    try:
        # 创建增量学习器
        from incremental_learner import IncrementalLearner
        
        learner = IncrementalLearner(
            db_url=args.db_url,
            symbols=["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"],  # 示例，实际应从配置中读取
            input_dim=5,
            hidden_dim=128,
            forecast_horizon=10,
            sequence_length=60,
            batch_size=64
        )
        
        # 创建风险控制系统
        risk_system = RiskControlSystem(
            db_url=args.db_url,
            timescale_url=args.timescale_url,
            models_dir=args.models_dir,
            error_threshold=args.error_threshold,
            consecutive_errors_limit=args.consecutive_errors,
            deepseek_api_key=args.deepseek_api_key,
            monitoring_interval=args.monitoring_interval,
            enable_auto_rollback=True
        )
        
        # 创建权重适配器（可选）
        weight_adapter = None
        try:
            from weight_adapter import WeightAdapter
            weight_adapter = WeightAdapter()
        except ImportError:
            logger.warning("未找到WeightAdapter模块，跳过权重适配")
        
        # 创建集成
        integration = RiskControlIntegration(
            incremental_learner=learner,
            risk_control=risk_system,
            weight_adapter=weight_adapter,
            notification_webhook=args.webhook_url,
            max_rollbacks_per_day=args.max_rollbacks,
            safe_mode=args.safe_mode
        )
        
        # 启动集成
        integration.start()
        
        # 保持主程序运行
        import time
        try:
            while True:
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("接收到终止信号，停止服务...")
            integration.stop()
            
    except Exception as e:
        logger.error(f"启动风险控制集成失败: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    main() 