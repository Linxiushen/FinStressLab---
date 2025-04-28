#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import json
import logging
import threading
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Any

# GPU管理
import torch
from torch.cuda import nvml
import psutil
import gc

# Celery
from celery import Celery, bootsteps, signals
from celery.worker.consumer import Consumer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/worker_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("prediction_worker")

# 从环境变量加载配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
MAX_WORKERS = int(os.getenv("MAX_WORKERS", "1"))
GPU_MEMORY_LIMIT = float(os.getenv("GPU_MEMORY_LIMIT", "0.9"))  # 默认使用90%的GPU内存
CPU_MEMORY_LIMIT = float(os.getenv("CPU_MEMORY_LIMIT", "0.8"))  # 默认使用80%的CPU内存
WORKER_ID = os.getenv("WORKER_ID", f"worker-{os.getpid()}")
MODEL_CACHE_ENABLED = os.getenv("MODEL_CACHE_ENABLED", "1") == "1"

# 初始化Celery应用
app = Celery('batch_prediction_tasks', broker=REDIS_URL, backend=REDIS_URL)
app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=MAX_WORKERS,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        'batch_prediction_tasks.*': {'queue': 'prediction_queue'}
    },
    worker_prefetch_multiplier=1,  # 每个worker一次只处理一个任务
    worker_max_tasks_per_child=50, # 每个worker处理50个任务后重启，防止内存泄漏
    task_time_limit=600,     # 任务超时10分钟
    task_soft_time_limit=540 # 软超时9分钟
)

# 创建日志目录
os.makedirs("logs", exist_ok=True)

# 模型缓存，用于在workers之间共享模型
class ModelCache:
    def __init__(self):
        self.models = {}
        self.lock = threading.Lock()
        self.last_used = {}
        self.enabled = MODEL_CACHE_ENABLED
        logger.info(f"模型缓存已初始化，缓存状态: {'启用' if self.enabled else '禁用'}")
    
    def get_model(self, model_path: str, device: torch.device = None):
        """获取缓存的模型，如果不存在则加载"""
        if not self.enabled:
            # 如果缓存禁用，则每次都重新加载模型
            return self._load_model(model_path, device)
        
        with self.lock:
            model_key = f"{model_path}_{device if device else 'cpu'}"
            
            if model_key in self.models:
                logger.info(f"从缓存加载模型: {model_path}")
                model = self.models[model_key]
                self.last_used[model_key] = time.time()
                return model
            
            # 清理可能过期的模型
            self._cleanup_cache()
            
            # 加载新模型
            model = self._load_model(model_path, device)
            self.models[model_key] = model
            self.last_used[model_key] = time.time()
            
            return model
    
    def _load_model(self, model_path: str, device: torch.device = None):
        """加载时间序列预测模型"""
        from model import TimeSeriesGPT
        
        logger.info(f"加载模型: {model_path}, 设备: {device if device else 'cpu'}")
        
        start_time = time.time()
        
        try:
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            model = TimeSeriesGPT.load_from_checkpoint(model_path)
            model.to(device)
            model.eval()
            
            logger.info(f"模型加载完成, 耗时: {time.time() - start_time:.2f}秒")
            return model
        
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            logger.error(traceback.format_exc())
            raise
    
    def _cleanup_cache(self):
        """清理长时间未使用的模型"""
        if not self.models:
            return
        
        current_time = time.time()
        # 模型缓存过期时间：30分钟
        expiry_time = 30 * 60
        
        expired_keys = [
            k for k, last_time in self.last_used.items() 
            if current_time - last_time > expiry_time
        ]
        
        for key in expired_keys:
            if key in self.models:
                logger.info(f"清理过期模型: {key}")
                del self.models[key]
                del self.last_used[key]
                # 手动GC
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

# 初始化全局模型缓存
model_cache = ModelCache()

# GPU资源管理器
class GPUManager:
    def __init__(self):
        self.have_gpu = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.have_gpu else 0
        self.current_device = 0
        self.lock = threading.Lock()
        
        if self.have_gpu:
            try:
                nvml.nvmlInit()
                logger.info(f"GPU资源管理器初始化成功, 可用GPU: {self.device_count}个")
                for i in range(self.device_count):
                    device_handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    device_name = nvml.nvmlDeviceGetName(device_handle)
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(device_handle)
                    logger.info(f"GPU {i}: {device_name}, 内存总量: {mem_info.total / 1024**2:.0f}MB")
            except Exception as e:
                logger.error(f"NVML初始化失败: {str(e)}")
        else:
            logger.warning("未检测到GPU，将使用CPU进行预测")
    
    def get_optimal_device(self):
        """获取当前最优的计算设备"""
        if not self.have_gpu:
            return torch.device("cpu")
        
        with self.lock:
            try:
                # 找出内存负载最小的GPU
                min_used_memory = float('inf')
                optimal_device_idx = 0
                
                for i in range(self.device_count):
                    device_handle = nvml.nvmlDeviceGetHandleByIndex(i)
                    mem_info = nvml.nvmlDeviceGetMemoryInfo(device_handle)
                    used_memory = mem_info.used / mem_info.total
                    
                    if used_memory < min_used_memory:
                        min_used_memory = used_memory
                        optimal_device_idx = i
                
                # 如果所有GPU都很满，回退到CPU
                if min_used_memory > GPU_MEMORY_LIMIT:
                    logger.warning(f"所有GPU使用率过高 ({min_used_memory:.1%})，回退到CPU")
                    return torch.device("cpu")
                
                logger.info(f"选择GPU {optimal_device_idx}，当前使用率: {min_used_memory:.1%}")
                return torch.device(f"cuda:{optimal_device_idx}")
            
            except Exception as e:
                logger.error(f"获取最优GPU失败: {str(e)}")
                return torch.device("cpu")
    
    def cleanup(self):
        """清理GPU资源"""
        if self.have_gpu:
            try:
                torch.cuda.empty_cache()
                gc.collect()
            except Exception as e:
                logger.error(f"清理GPU资源失败: {str(e)}")

# 初始化全局GPU管理器
gpu_manager = GPUManager()

# Worker启动时的资源检查
@signals.worker_ready.connect
def check_resource_at_startup(sender, **kwargs):
    """Worker启动时检查资源"""
    logger.info(f"Worker {WORKER_ID} 启动")
    
    # 检查CPU资源
    cpu_percent = psutil.cpu_percent()
    memory = psutil.virtual_memory()
    logger.info(f"CPU使用率: {cpu_percent}%, 内存使用率: {memory.percent}%")
    
    # 检查GPU资源（如果有）
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        for i in range(torch.cuda.device_count()):
            try:
                device_handle = nvml.nvmlDeviceGetHandleByIndex(i)
                device_name = nvml.nvmlDeviceGetName(device_handle)
                mem_info = nvml.nvmlDeviceGetMemoryInfo(device_handle)
                used_percent = mem_info.used / mem_info.total * 100
                logger.info(f"GPU {i} ({device_name}) 内存使用率: {used_percent:.1f}%")
            except Exception as e:
                logger.error(f"获取GPU {i} 信息失败: {str(e)}")

# Worker关闭时清理资源
@signals.worker_shutdown.connect
def cleanup_at_shutdown(sender, **kwargs):
    """Worker关闭时清理资源"""
    logger.info(f"Worker {WORKER_ID} 关闭，清理资源")
    
    # 清理GPU资源
    gpu_manager.cleanup()
    
    # 关闭NVML
    if torch.cuda.is_available():
        try:
            nvml.nvmlShutdown()
        except Exception as e:
            logger.error(f"NVML关闭失败: {str(e)}")

# 任务执行前的资源检查
@signals.task_prerun.connect
def check_resources_before_task(sender, task_id, task, args, kwargs, **rest):
    """任务执行前检查资源"""
    logger.info(f"开始处理任务: {task_id}")
    
    # 检查内存使用率
    memory = psutil.virtual_memory()
    if memory.percent > CPU_MEMORY_LIMIT * 100:
        logger.warning(f"内存使用率过高: {memory.percent}%，可能影响性能")

# 任务执行后的资源清理
@signals.task_postrun.connect
def cleanup_after_task(sender, task_id, task, args, kwargs, retval, state, **rest):
    """任务执行后清理资源"""
    logger.info(f"任务完成: {task_id}, 状态: {state}")
    
    # 手动GC
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# 主预测任务
@app.task(name="batch_prediction_tasks.batch_prediction_task", bind=True, max_retries=2)
def batch_prediction_task(self, request_dict, request_id):
    try:
        logger.info(f"开始处理批量预测任务: {request_id}")
        start_time = time.time()
        
        from batch_prediction_api import BatchPredictionRequest, anonymize_data, redis_client, RESULT_EXPIRY
        
        # 解析请求
        request = BatchPredictionRequest(**request_dict)
        
        # 获取最优设备
        device = gpu_manager.get_optimal_device()
        
        # 从缓存加载模型
        model = model_cache.get_model("models/latest.ckpt", device)
        
        results = []
        errors = {}
        
        total_series = len(request.series)
        processed = 0
        
        for series in request.series:
            try:
                # 匿名化敏感数据用于日志
                series_hash = anonymize_data(series.values)
                processed += 1
                
                if processed % 100 == 0 or processed == total_series:
                    logger.info(f"进度: {processed}/{total_series} ({processed/total_series:.1%})")
                
                # 执行预测
                with torch.no_grad():
                    input_tensor = torch.tensor(series.values, dtype=torch.float32)
                    
                    # 准备特征数据（如果有）
                    features = None
                    if series.features:
                        features = torch.tensor(series.features, dtype=torch.float32)
                    
                    # 将数据移至设备
                    input_tensor = input_tensor.to(device)
                    if features is not None:
                        features = features.to(device)
                    
                    # 执行预测
                    forecast, confidence = model.predict(
                        input_tensor, 
                        features=features,
                        horizon=request.forecast_horizon,
                        return_confidence=request.confidence_intervals
                    )
                
                # 构建结果
                result = {
                    "id": series.id,
                    "forecast": forecast.cpu().numpy().tolist() if isinstance(forecast, torch.Tensor) else forecast.tolist(),
                }
                
                if request.confidence_intervals and confidence is not None:
                    if isinstance(confidence[0], torch.Tensor):
                        result["confidence_lower"] = confidence[0].cpu().numpy().tolist()
                        result["confidence_upper"] = confidence[1].cpu().numpy().tolist()
                    else:
                        result["confidence_lower"] = confidence[0].tolist()
                        result["confidence_upper"] = confidence[1].tolist()
                    
                if request.include_history:
                    result["history"] = series.values
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"处理时间序列 {series.id} 时出错: {str(e)}")
                logger.error(traceback.format_exc())
                errors[series.id] = str(e)
        
        processing_time = time.time() - start_time
        
        # 构建响应
        response = {
            "request_id": request_id,
            "status": "completed",
            "message": "预测完成",
            "results": results,
            "processing_time": processing_time
        }
        
        if errors:
            response["errors"] = errors
            response["status"] = "partial"
            response["message"] = f"部分预测完成，{len(errors)}/{len(request.series)}个序列失败"
        
        # 存储结果到Redis
        redis_client.setex(
            f"prediction_result:{request_id}", 
            RESULT_EXPIRY, 
            json.dumps(response)
        )
        
        # 更新统计信息
        try:
            stats = redis_client.get("prediction_stats")
            if stats:
                stats = json.loads(stats)
            else:
                stats = {"completed_count": 0, "failed_count": 0, "avg_processing_time": 0}
                
            # 更新处理时间
            current_avg = stats["avg_processing_time"]
            current_count = stats["completed_count"]
            new_avg = (current_avg * current_count + processing_time) / (current_count + 1)
            
            # 更新统计
            stats["completed_count"] += 1
            stats["avg_processing_time"] = new_avg
            
            redis_client.set("prediction_stats", json.dumps(stats))
        except Exception as e:
            logger.error(f"更新统计信息失败: {str(e)}")
        
        logger.info(f"批量预测任务 {request_id} 完成，处理时间: {processing_time:.2f}秒")
        return response
        
    except Exception as e:
        logger.error(f"批量预测任务 {request_id} 失败: {str(e)}")
        logger.error(traceback.format_exc())
        
        # 更新失败统计
        try:
            stats = redis_client.get("prediction_stats")
            if stats:
                stats = json.loads(stats)
                stats["failed_count"] += 1
                redis_client.set("prediction_stats", json.dumps(stats))
        except Exception as stats_error:
            logger.error(f"更新失败统计失败: {str(stats_error)}")
        
        # 如果是临时错误，尝试重试
        if "out of memory" in str(e).lower() or "cuda" in str(e).lower():
            # GPU资源不足，重试前先清理内存
            gpu_manager.cleanup()
            # 延迟10秒后重试
            raise self.retry(exc=e, countdown=10)
            
        # 存储错误到Redis
        error_response = {
            "request_id": request_id,
            "status": "failed",
            "message": f"预测任务失败: {str(e)}",
            "processing_time": time.time() - start_time
        }
        redis_client.setex(f"prediction_result:{request_id}", RESULT_EXPIRY, json.dumps(error_response))
        return error_response

if __name__ == "__main__":
    # 启动Celery Worker
    from celery.bin import worker
    
    worker = worker.worker(app=app)
    worker.run(
        loglevel='INFO',
        hostname=f"{WORKER_ID}@%h",
        concurrency=MAX_WORKERS
    ) 