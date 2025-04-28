import os
import logging
import time
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
import requests
from datetime import datetime, timedelta
import threading
import psycopg2
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON, Boolean, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import torch

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('risk_control')

# 声明数据库模型
Base = declarative_base()

class RiskEvent(Base):
    """风险事件记录"""
    __tablename__ = 'risk_events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, default=datetime.now)
    event_type = Column(String(50))
    symbol = Column(String(20), index=True)
    severity = Column(String(20))  # 'low', 'medium', 'high', 'critical'
    description = Column(Text)
    error_values = Column(JSON)
    model_state = Column(String(255))  # 模型状态路径
    action_taken = Column(String(50))
    resolved = Column(Boolean, default=False)
    resolved_at = Column(DateTime, nullable=True)
    report_path = Column(String(255), nullable=True)


class ModelSnapshot(Base):
    """模型快照记录"""
    __tablename__ = 'model_snapshots'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, index=True, default=datetime.now)
    model_path = Column(String(255))
    performance_metrics = Column(JSON)
    is_stable = Column(Boolean, default=True)
    description = Column(Text, nullable=True)
    

class RiskControlSystem:
    """
    实时风险控制系统
    
    功能:
    1. 监控预测误差
    2. 自动熔断机制
    3. 风险报告生成
    4. 模型回滚
    """
    
    def __init__(self,
                 db_url: str,
                 timescale_url: str,
                 models_dir: str = "./models/snapshots",
                 error_threshold: float = 0.2,
                 consecutive_errors_limit: int = 3,
                 deepseek_api_key: Optional[str] = None,
                 timegpt_api_url: Optional[str] = None,
                 monitoring_interval: int = 300,  # 5分钟检查一次
                 enable_auto_rollback: bool = True):
        """
        初始化风险控制系统
        
        Args:
            db_url: 数据库连接URL
            timescale_url: TimescaleDB连接URL
            models_dir: 模型快照目录
            error_threshold: 误差阈值
            consecutive_errors_limit: 连续误差次数限制
            deepseek_api_key: DeepSeek API密钥
            timegpt_api_url: TimeGPT API URL
            monitoring_interval: 监控间隔（秒）
            enable_auto_rollback: 启用自动回滚
        """
        self.db_url = db_url
        self.timescale_url = timescale_url
        self.models_dir = models_dir
        self.error_threshold = error_threshold
        self.consecutive_errors_limit = consecutive_errors_limit
        self.deepseek_api_key = deepseek_api_key
        self.timegpt_api_url = timegpt_api_url
        self.monitoring_interval = monitoring_interval
        self.enable_auto_rollback = enable_auto_rollback
        
        # 确保模型快照目录存在
        os.makedirs(self.models_dir, exist_ok=True)
        
        # 连接数据库
        self.engine = create_engine(db_url)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
        # 连接TimescaleDB
        self.timescale_engine = create_engine(timescale_url)
        
        # 错误监控状态
        self.error_counts = {}  # 按symbol存储连续错误次数
        self.last_errors = {}   # 按symbol存储最近的错误值
        self.trading_frozen = {}  # 按symbol存储交易冻结状态
        
        # 监控线程
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 事件回调
        self.on_circuit_breaker_triggered = None
        self.on_model_rollback = None
        self.on_risk_report_generated = None
        
        logger.info("风险控制系统已初始化")
    
    def start_monitoring(self):
        """启动风险监控"""
        if self.monitoring_active:
            logger.warning("监控已经在运行中")
            return
            
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        logger.info("风险监控已启动")
    
    def stop_monitoring(self):
        """停止风险监控"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        logger.info("风险监控已停止")
    
    def _monitoring_loop(self):
        """监控循环"""
        while self.monitoring_active:
            try:
                # 检查所有活跃交易对的误差
                self._check_prediction_errors()
                
                # 等待下一个监控周期
                time.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"监控循环出错: {str(e)}")
                time.sleep(60)  # 出错后等待1分钟再重试
    
    def _check_prediction_errors(self):
        """检查预测误差"""
        # 从数据库获取最近的预测和实际值
        with self.Session() as session:
            # 这里假设有一个predictions表存储了预测结果
            query = """
                SELECT symbol, prediction, actual, timestamp 
                FROM predictions
                WHERE timestamp > NOW() - INTERVAL '1 day'
                ORDER BY symbol, timestamp DESC
            """
            
            try:
                results = pd.read_sql(query, session.bind)
                
                # 按交易对分组
                for symbol, group in results.groupby('symbol'):
                    # 只取最近的记录
                    recent_preds = group.head(10)
                    
                    # 计算相对误差
                    recent_preds['rel_error'] = abs(recent_preds['prediction'] - recent_preds['actual']) / (abs(recent_preds['actual']) + 1e-6)
                    
                    # 检查最近3次预测的误差
                    recent_errors = recent_preds.head(3)['rel_error'].values
                    
                    # 如果有3个记录，检查连续误差
                    if len(recent_errors) >= 3:
                        self._process_symbol_errors(symbol, recent_errors, recent_preds.iloc[0])
            
            except Exception as e:
                logger.error(f"检查预测误差时出错: {str(e)}")
    
    def _process_symbol_errors(self, symbol: str, errors: np.ndarray, latest_data: pd.Series):
        """处理指定交易对的误差"""
        # 检查是否所有误差都超过阈值
        all_exceed = all(error > self.error_threshold for error in errors)
        
        # 更新连续错误计数
        if all_exceed:
            self.error_counts[symbol] = self.error_counts.get(symbol, 0) + 1
            self.last_errors[symbol] = errors
        else:
            # 重置错误计数
            self.error_counts[symbol] = 0
            
        # 检查是否触发熔断
        if self.error_counts.get(symbol, 0) >= self.consecutive_errors_limit:
            # 触发熔断
            self._trigger_circuit_breaker(symbol)
    
    def _trigger_circuit_breaker(self, symbol: str):
        """触发交易熔断"""
        if self.trading_frozen.get(symbol, False):
            # 已经冻结，不需要再次触发
            return
            
        logger.warning(f"触发 {symbol} 的交易熔断! 连续误差次数: {self.error_counts.get(symbol)}")
        
        # 冻结自动交易
        self.trading_frozen[symbol] = True
        
        # 记录风险事件
        event_id = self._record_risk_event(
            symbol=symbol,
            event_type="circuit_breaker",
            severity="high",
            description=f"连续{self.consecutive_errors_limit}次预测误差超过阈值{self.error_threshold}",
            error_values=self.last_errors.get(symbol, []).tolist()
        )
        
        # 调用DeepSeek生成风险分析报告
        report_path = self._generate_risk_report(symbol, event_id)
        
        # 触发TimeGPT异常检测
        self._trigger_anomaly_detection(symbol)
        
        # 如果启用了自动回滚，执行回滚
        if self.enable_auto_rollback:
            self._rollback_to_stable_state(symbol)
            
        # 调用回调函数
        if self.on_circuit_breaker_triggered:
            self.on_circuit_breaker_triggered(symbol, event_id)
    
    def _record_risk_event(self, symbol: str, event_type: str, severity: str, description: str, error_values: List) -> int:
        """记录风险事件"""
        with self.Session() as session:
            # 获取当前模型状态
            model_state = self._get_current_model_state()
            
            # 创建风险事件记录
            event = RiskEvent(
                timestamp=datetime.now(),
                event_type=event_type,
                symbol=symbol,
                severity=severity,
                description=description,
                error_values=error_values,
                model_state=model_state,
                action_taken="trading_frozen"
            )
            
            session.add(event)
            session.commit()
            
            logger.info(f"已记录风险事件 ID: {event.id}, 类型: {event_type}, 交易对: {symbol}")
            return event.id
    
    def _get_current_model_state(self) -> str:
        """获取当前模型状态"""
        # 这里应返回当前模型的路径或标识符
        # 实际实现需要根据你的模型管理方式
        return "current_model_path"
    
    def _generate_risk_report(self, symbol: str, event_id: int) -> str:
        """调用DeepSeek生成风险分析报告"""
        if not self.deepseek_api_key:
            logger.warning("DeepSeek API key未设置，跳过风险报告生成")
            return ""
            
        try:
            # 收集风险相关数据
            risk_data = self._collect_risk_data(symbol)
            
            # 调用DeepSeek API
            report = self._call_deepseek_api(risk_data)
            
            # 保存报告
            report_path = f"reports/risk_report_{symbol}_{event_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            os.makedirs(os.path.dirname(report_path), exist_ok=True)
            
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            
            # 更新风险事件记录
            with self.Session() as session:
                event = session.query(RiskEvent).filter(RiskEvent.id == event_id).first()
                if event:
                    event.report_path = report_path
                    session.commit()
            
            logger.info(f"已生成风险报告: {report_path}")
            
            # 调用回调函数
            if self.on_risk_report_generated:
                self.on_risk_report_generated(symbol, event_id, report_path)
                
            return report_path
            
        except Exception as e:
            logger.error(f"生成风险报告时出错: {str(e)}")
            return ""
    
    def _collect_risk_data(self, symbol: str) -> Dict:
        """收集风险相关数据"""
        # 从TimescaleDB获取市场数据
        market_data = self._get_market_data(symbol)
        
        # 获取最近的预测误差
        prediction_errors = self._get_prediction_errors(symbol)
        
        # 获取模型状态
        model_info = self._get_model_info()
        
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "market_data": market_data,
            "prediction_errors": prediction_errors,
            "model_info": model_info,
            "error_threshold": self.error_threshold,
            "consecutive_errors": self.error_counts.get(symbol, 0)
        }
    
    def _get_market_data(self, symbol: str) -> Dict:
        """从TimescaleDB获取市场数据"""
        try:
            # 查询最近的市场数据
            query = f"""
                SELECT time, open, high, low, close, volume
                FROM market_data
                WHERE symbol = '{symbol}'
                AND time > NOW() - INTERVAL '1 day'
                ORDER BY time DESC
                LIMIT 100
            """
            
            df = pd.read_sql(query, self.timescale_engine)
            
            # 计算一些基本的市场指标
            if not df.empty:
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(window=20).std() * np.sqrt(252)
                
                # 计算移动均线
                df['sma_20'] = df['close'].rolling(window=20).mean()
                df['sma_50'] = df['close'].rolling(window=50).mean()
                
                # 填充NaN值
                df = df.fillna(0)
                
                # 转换为字典
                market_data = {
                    "recent_data": df.head(20).to_dict(orient='records'),
                    "current_price": float(df['close'].iloc[0]),
                    "daily_change": float(df['returns'].iloc[0]) if not np.isnan(df['returns'].iloc[0]) else 0,
                    "current_volatility": float(df['volatility'].iloc[0]) if not np.isnan(df['volatility'].iloc[0]) else 0,
                }
                
                return market_data
            
            return {"error": "No market data found"}
            
        except Exception as e:
            logger.error(f"获取市场数据时出错: {str(e)}")
            return {"error": str(e)}
    
    def _get_prediction_errors(self, symbol: str) -> List:
        """获取最近的预测误差"""
        try:
            # 查询最近的预测结果
            query = """
                SELECT timestamp, prediction, actual, ABS(prediction - actual) / NULLIF(ABS(actual), 0) as rel_error
                FROM predictions
                WHERE symbol = %s
                AND timestamp > NOW() - INTERVAL '1 day'
                ORDER BY timestamp DESC
                LIMIT 20
            """
            
            with self.Session() as session:
                result = session.execute(query, {"symbol": symbol})
                errors = [dict(row) for row in result]
                return errors
                
        except Exception as e:
            logger.error(f"获取预测误差时出错: {str(e)}")
            return []
    
    def _get_model_info(self) -> Dict:
        """获取模型信息"""
        # 实际实现需要根据项目的模型管理方式
        return {
            "model_type": "TimeSeriesGPT",
            "last_updated": datetime.now().isoformat(),
            "parameters": {
                "input_dim": 5,
                "hidden_dim": 128,
                "num_layers": 2
            }
        }
    
    def _call_deepseek_api(self, risk_data: Dict) -> Dict:
        """调用DeepSeek API生成风险分析报告"""
        if not self.deepseek_api_key:
            return {"error": "DeepSeek API key not configured"}
            
        try:
            # 构造请求
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "model": "deepseek-chat",
                "messages": [
                    {
                        "role": "system", 
                        "content": "你是一个金融风险分析专家，请基于提供的市场数据和预测误差，生成一份全面的风险分析报告。"
                    },
                    {
                        "role": "user",
                        "content": f"""
                        请基于以下数据生成金融交易风险分析报告:
                        
                        交易对: {risk_data['symbol']}
                        时间: {risk_data['timestamp']}
                        当前价格: {risk_data.get('market_data', {}).get('current_price', 'N/A')}
                        当日涨跌幅: {risk_data.get('market_data', {}).get('daily_change', 'N/A')}
                        当前波动率: {risk_data.get('market_data', {}).get('current_volatility', 'N/A')}
                        
                        预测误差情况:
                        - 误差阈值: {risk_data['error_threshold']}
                        - 连续超过阈值次数: {risk_data['consecutive_errors']}
                        
                        请提供以下分析:
                        1. 风险事件摘要
                        2. 可能的原因分析
                        3. 市场状况评估
                        4. 建议的缓解措施
                        5. 后续操作建议
                        
                        以JSON格式返回，包含以上各节内容。
                        """
                    }
                ]
            }
            
            # 发送请求
            # 注意：这是示例代码，实际DeepSeek API可能有所不同，请查阅其官方文档
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                # 解析DeepSeek响应，这里假设它直接返回JSON格式的报告
                report_text = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
                
                # 尝试解析JSON报告
                try:
                    report = json.loads(report_text)
                    return report
                except json.JSONDecodeError:
                    # 如果不是JSON格式，就将文本封装成JSON
                    return {"raw_report": report_text}
            else:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "detail": response.text
                }
                
        except Exception as e:
            logger.error(f"调用DeepSeek API时出错: {str(e)}")
            return {"error": str(e)}
    
    def _trigger_anomaly_detection(self, symbol: str):
        """触发TimeGPT异常检测"""
        if not self.timegpt_api_url:
            logger.warning("TimeGPT API URL未设置，跳过异常检测")
            return
            
        try:
            # 构造请求
            payload = {
                "symbol": symbol,
                "start_date": (datetime.now() - timedelta(days=30)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "detection_mode": "anomaly"
            }
            
            # 发送请求
            response = requests.post(
                f"{self.timegpt_api_url}/detect-anomalies",
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # 检查是否检测到异常
                if result.get("anomalies_detected", False):
                    anomalies = result.get("anomalies", [])
                    logger.warning(f"TimeGPT异常检测结果: 检测到 {len(anomalies)} 个异常")
                    
                    # 添加异常信息到风险事件
                    with self.Session() as session:
                        # 查找最近的风险事件
                        event = session.query(RiskEvent)\
                            .filter(RiskEvent.symbol == symbol)\
                            .order_by(RiskEvent.timestamp.desc())\
                            .first()
                        
                        if event:
                            # 更新描述
                            event.description += f" TimeGPT检测到{len(anomalies)}个异常"
                            session.commit()
                else:
                    logger.info(f"TimeGPT异常检测结果: 未检测到异常")
            else:
                logger.error(f"TimeGPT API请求失败，状态码: {response.status_code}")
                
        except Exception as e:
            logger.error(f"触发TimeGPT异常检测时出错: {str(e)}")
    
    def _rollback_to_stable_state(self, symbol: str):
        """回滚到最近的稳定版本"""
        logger.info(f"开始回滚 {symbol} 的模型到最近的稳定版本")
        
        try:
            # 查找最近的稳定模型快照
            stable_snapshot = self._find_latest_stable_snapshot()
            
            if not stable_snapshot:
                logger.warning("未找到稳定的模型快照，无法执行回滚")
                return False
                
            # 执行模型回滚
            rollback_success = self._perform_model_rollback(stable_snapshot)
            
            if rollback_success:
                # 恢复TimescaleDB中的参数
                self._restore_timescale_parameters(stable_snapshot.timestamp)
                
                # 记录回滚事件
                with self.Session() as session:
                    # 更新最近的风险事件
                    event = session.query(RiskEvent)\
                        .filter(RiskEvent.symbol == symbol)\
                        .order_by(RiskEvent.timestamp.desc())\
                        .first()
                    
                    if event:
                        event.action_taken = "model_rollback"
                        session.commit()
                
                # 调用回调函数
                if self.on_model_rollback:
                    self.on_model_rollback(symbol, stable_snapshot.id)
                    
                logger.info(f"成功回滚到模型快照 ID: {stable_snapshot.id}, 路径: {stable_snapshot.model_path}")
                return True
            else:
                logger.error(f"模型回滚失败")
                return False
                
        except Exception as e:
            logger.error(f"执行模型回滚时出错: {str(e)}")
            return False
    
    def _find_latest_stable_snapshot(self) -> Optional[ModelSnapshot]:
        """查找最近的稳定模型快照"""
        with self.Session() as session:
            # 查询最近30天内的稳定快照
            snapshot = session.query(ModelSnapshot)\
                .filter(
                    ModelSnapshot.is_stable == True,
                    ModelSnapshot.timestamp > datetime.now() - timedelta(days=30)
                )\
                .order_by(ModelSnapshot.timestamp.desc())\
                .first()
                
            return snapshot
    
    def _perform_model_rollback(self, snapshot: ModelSnapshot) -> bool:
        """执行模型回滚"""
        try:
            # 检查快照文件是否存在
            if not os.path.exists(snapshot.model_path):
                logger.error(f"模型快照文件不存在: {snapshot.model_path}")
                return False
                
            # 加载模型
            model = torch.load(snapshot.model_path)
            
            # TODO: 这里需要将模型设置为当前活跃模型
            # 实际实现取决于你的模型管理方式
            
            logger.info(f"已加载模型快照: {snapshot.model_path}")
            return True
            
        except Exception as e:
            logger.error(f"执行模型回滚时出错: {str(e)}")
            return False
    
    def _restore_timescale_parameters(self, timestamp: datetime):
        """恢复TimescaleDB中的对应时段参数"""
        try:
            # 这里需要根据实际情况还原参数
            # 例如从参数历史表中查询对应时间点的参数
            
            query = """
                SELECT parameter_name, parameter_value
                FROM parameter_history
                WHERE timestamp <= %s
                ORDER BY timestamp DESC
                LIMIT 1
            """
            
            with self.timescale_engine.connect() as conn:
                result = conn.execute(query, (timestamp,))
                parameters = [dict(row) for row in result]
                
                if parameters:
                    logger.info(f"恢复参数: {parameters}")
                    
                    # TODO: 应用参数
                    # 实际实现取决于你的系统架构
                else:
                    logger.warning(f"未找到时间点 {timestamp} 之前的参数记录")
                    
        except Exception as e:
            logger.error(f"恢复TimescaleDB参数时出错: {str(e)}")
    
    def create_model_snapshot(self, model_path: str, performance_metrics: Dict, is_stable: bool = True, description: str = None) -> int:
        """
        创建模型快照
        
        Args:
            model_path: 模型文件路径
            performance_metrics: 性能指标
            is_stable: 是否标记为稳定版本
            description: 快照描述
            
        Returns:
            快照ID
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # 记录快照信息
            with self.Session() as session:
                snapshot = ModelSnapshot(
                    timestamp=datetime.now(),
                    model_path=model_path,
                    performance_metrics=performance_metrics,
                    is_stable=is_stable,
                    description=description
                )
                
                session.add(snapshot)
                session.commit()
                
                logger.info(f"已创建模型快照 ID: {snapshot.id}, 路径: {model_path}")
                return snapshot.id
                
        except Exception as e:
            logger.error(f"创建模型快照时出错: {str(e)}")
            return -1
    
    def reset_trading_freeze(self, symbol: Optional[str] = None):
        """
        重置交易冻结状态
        
        Args:
            symbol: 指定交易对，如果为None则重置所有
        """
        if symbol:
            if symbol in self.trading_frozen:
                self.trading_frozen[symbol] = False
                self.error_counts[symbol] = 0
                logger.info(f"已重置 {symbol} 的交易冻结状态")
        else:
            # 重置所有交易对
            for sym in list(self.trading_frozen.keys()):
                self.trading_frozen[sym] = False
                self.error_counts[sym] = 0
            
            logger.info("已重置所有交易对的冻结状态")
    
    def is_trading_frozen(self, symbol: str) -> bool:
        """
        检查交易对是否被冻结
        
        Args:
            symbol: 交易对
            
        Returns:
            是否冻结
        """
        return self.trading_frozen.get(symbol, False)
    
    def get_active_risk_events(self, resolved: bool = False) -> List[Dict]:
        """
        获取活跃的风险事件
        
        Args:
            resolved: 是否包含已解决的事件
            
        Returns:
            风险事件列表
        """
        with self.Session() as session:
            query = session.query(RiskEvent)
            
            if not resolved:
                query = query.filter(RiskEvent.resolved == False)
                
            events = query.order_by(RiskEvent.timestamp.desc()).limit(100).all()
            
            return [
                {
                    "id": event.id,
                    "timestamp": event.timestamp.isoformat(),
                    "event_type": event.event_type,
                    "symbol": event.symbol,
                    "severity": event.severity,
                    "description": event.description,
                    "action_taken": event.action_taken,
                    "resolved": event.resolved
                }
                for event in events
            ]
    
    def resolve_risk_event(self, event_id: int, resolution_notes: str = None) -> bool:
        """
        将风险事件标记为已解决
        
        Args:
            event_id: 事件ID
            resolution_notes: 解决说明
            
        Returns:
            是否成功
        """
        try:
            with self.Session() as session:
                event = session.query(RiskEvent).filter(RiskEvent.id == event_id).first()
                
                if event:
                    event.resolved = True
                    event.resolved_at = datetime.now()
                    
                    if resolution_notes:
                        event.description += f"\n解决说明: {resolution_notes}"
                        
                    session.commit()
                    logger.info(f"已将风险事件 ID: {event_id} 标记为已解决")
                    
                    # 如果交易对被冻结，解除冻结
                    if event.symbol in self.trading_frozen:
                        self.reset_trading_freeze(event.symbol)
                        
                    return True
                else:
                    logger.warning(f"未找到风险事件 ID: {event_id}")
                    return False
                    
        except Exception as e:
            logger.error(f"解决风险事件时出错: {str(e)}")
            return False


# 示例用法
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="运行风险控制系统")
    parser.add_argument("--db_url", type=str, default="sqlite:///./timeseries_gpt.db", help="数据库URL")
    parser.add_argument("--timescale_url", type=str, default="postgresql://user:password@localhost:5432/timescale", help="TimescaleDB URL")
    parser.add_argument("--error_threshold", type=float, default=0.2, help="误差阈值")
    parser.add_argument("--models_dir", type=str, default="./models/snapshots", help="模型快照目录")
    parser.add_argument("--deepseek_api_key", type=str, help="DeepSeek API密钥")
    
    args = parser.parse_args()
    
    # 创建风险控制系统
    risk_system = RiskControlSystem(
        db_url=args.db_url,
        timescale_url=args.timescale_url,
        models_dir=args.models_dir,
        error_threshold=args.error_threshold,
        deepseek_api_key=args.deepseek_api_key
    )
    
    # 启动监控
    risk_system.start_monitoring()
    
    # 保持主线程运行
    try:
        while True:
            # 打印活跃风险事件
            events = risk_system.get_active_risk_events()
            if events:
                print(f"当前有 {len(events)} 个活跃风险事件")
                for event in events[:3]:  # 只显示前3个
                    print(f"  - {event['timestamp']}: {event['symbol']} ({event['severity']}) - {event['description']}")
            
            time.sleep(60)
    except KeyboardInterrupt:
        risk_system.stop_monitoring()
        print("风险控制系统已停止") 