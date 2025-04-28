#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
import hashlib
import traceback

# API框架
import fastapi
from fastapi import FastAPI, Depends, HTTPException, Request, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator, root_validator
from starlette.middleware.base import BaseHTTPMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.security.api_key import APIKeyHeader

# 异步处理
import redis
from celery import Celery
from celery.result import AsyncResult

# 本地模块
from model import TimeSeriesGPT
from data_utils import TimeSeriesDataProcessor
import config

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"logs/batch_api_{datetime.now().strftime('%Y%m%d')}.log")
    ]
)
logger = logging.getLogger("batch_prediction_api")

# 从环境变量加载配置
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
DB_URL = os.getenv("DB_URL", "postgresql://user:password@localhost:5432/timeseries_db")
API_KEYS = os.getenv("API_KEYS", "").split(",")
RATE_LIMIT = os.getenv("RATE_LIMIT", "100/minute")
RESULT_EXPIRY = int(os.getenv("RESULT_EXPIRY", "3600"))  # 结果保存1小时
MAX_SERIES_PER_BATCH = int(os.getenv("MAX_SERIES_PER_BATCH", "5000"))

# 初始化Redis
redis_client = redis.Redis.from_url(REDIS_URL)

# 初始化Celery
celery_app = Celery('batch_prediction_tasks', broker=REDIS_URL, backend=REDIS_URL)
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    worker_concurrency=3,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_routes={
        'batch_prediction_tasks.*': {'queue': 'prediction_queue'}
    }
)

# 初始化API
app = FastAPI(
    title="TimeSeriesGPT高性能批量预测API",
    description="金融级高吞吐量时间序列预测服务",
    version="1.0.0"
)

# 初始化速率限制器
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制来源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型
class TimeSeriesRequest(BaseModel):
    id: str
    values: List[float]
    features: Optional[List[List[float]]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    @validator('values')
    def validate_values_length(cls, v):
        if len(v) < 10:
            raise ValueError("时间序列长度必须至少为10个数据点")
        return v

class BatchPredictionRequest(BaseModel):
    series: List[TimeSeriesRequest]
    forecast_horizon: int = Field(default=5, ge=1, le=30)
    include_history: bool = Field(default=False)
    confidence_intervals: bool = Field(default=True)
    
    @validator('series')
    def validate_series_count(cls, v):
        if len(v) > MAX_SERIES_PER_BATCH:
            raise ValueError(f"每批次最多支持{MAX_SERIES_PER_BATCH}个时间序列")
        return v

class PredictionResult(BaseModel):
    id: str
    forecast: List[float]
    confidence_lower: Optional[List[float]] = None
    confidence_upper: Optional[List[float]] = None
    history: Optional[List[float]] = None

class BatchPredictionResponse(BaseModel):
    request_id: str
    status: str
    message: str
    results: Optional[List[PredictionResult]] = None
    errors: Optional[Dict[str, str]] = None
    processing_time: Optional[float] = None

# API密钥认证中间件
class APIKeyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path.startswith("/docs") or request.url.path.startswith("/redoc") or request.url.path == "/health":
            return await call_next(request)
        
        api_key = request.headers.get("X-API-Key")
        if not api_key or api_key not in API_KEYS:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "无效的API密钥"}
            )
        
        # 将API密钥附加到请求状态，用于速率限制
        request.state.api_key = api_key
        return await call_next(request)

# 添加API密钥中间件
app.add_middleware(APIKeyMiddleware)

# 获取限制器的键函数 - 基于API密钥而不是IP
def get_api_key(request: Request):
    return request.state.api_key

# 数据匿名化函数
def anonymize_data(data: List[float]) -> str:
    """对时间序列数据进行哈希，以保护敏感信息"""
    data_str = json.dumps(data)
    return hashlib.sha256(data_str.encode()).hexdigest()

# Celery任务
@celery_app.task(name="batch_prediction_tasks.batch_prediction_task")
def batch_prediction_task(request_dict, request_id):
    try:
        start_time = time.time()
        # 解析请求
        request = BatchPredictionRequest(**request_dict)
        
        # 初始化模型 - 这里应该使用缓存或模型池
        model = TimeSeriesGPT.load_from_checkpoint("models/latest.ckpt")
        
        results = []
        errors = {}
        
        for series in request.series:
            try:
                # 匿名化敏感数据用于日志
                series_hash = anonymize_data(series.values)
                logger.info(f"Processing series {series.id} (hash: {series_hash[:8]}...)")
                
                # 执行预测
                forecast, confidence = model.predict(
                    series.values, 
                    features=series.features,
                    horizon=request.forecast_horizon,
                    return_confidence=request.confidence_intervals
                )
                
                # 构建结果
                result = {
                    "id": series.id,
                    "forecast": forecast.tolist(),
                }
                
                if request.confidence_intervals and confidence is not None:
                    result["confidence_lower"] = confidence[0].tolist()
                    result["confidence_upper"] = confidence[1].tolist()
                    
                if request.include_history:
                    result["history"] = series.values
                    
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing series {series.id}: {str(e)}")
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
        
        return response
        
    except Exception as e:
        logger.error(f"Batch prediction task failed: {str(e)}")
        logger.error(traceback.format_exc())
        error_response = {
            "request_id": request_id,
            "status": "failed",
            "message": f"预测任务失败: {str(e)}",
            "processing_time": time.time() - start_time
        }
        redis_client.setex(f"prediction_result:{request_id}", RESULT_EXPIRY, json.dumps(error_response))
        return error_response

# API端点
@app.post("/api/v1/predictions/batch", response_model=BatchPredictionResponse)
@limiter.limit(RATE_LIMIT, key_func=get_api_key)
async def create_batch_prediction(request: BatchPredictionRequest, background_tasks: BackgroundTasks, req: Request):
    """
    提交批量预测请求
    """
    request_id = str(uuid.uuid4())
    logger.info(f"Received batch prediction request: {request_id} with {len(request.series)} series")
    
    # 初始化任务状态
    status_data = {
        "request_id": request_id,
        "status": "pending",
        "message": "预测任务已提交，正在排队处理",
        "submitted_at": datetime.now().isoformat()
    }
    
    # 保存任务状态到Redis
    redis_client.setex(f"prediction_status:{request_id}", RESULT_EXPIRY, json.dumps(status_data))
    
    # 提交Celery任务
    batch_prediction_task.apply_async(
        args=[request.dict(), request_id],
        task_id=request_id
    )
    
    return status_data

@app.get("/api/v1/predictions/batch/{request_id}", response_model=BatchPredictionResponse)
@limiter.limit(RATE_LIMIT, key_func=get_api_key)
async def get_batch_prediction(request_id: str, req: Request):
    """
    获取批量预测结果
    """
    # 先检查结果
    result = redis_client.get(f"prediction_result:{request_id}")
    if result:
        return json.loads(result)
    
    # 检查任务状态
    status = redis_client.get(f"prediction_status:{request_id}")
    if status:
        return json.loads(status)
    
    # 检查Celery任务状态
    task = AsyncResult(request_id, app=celery_app)
    if task.state == 'PENDING':
        return {
            "request_id": request_id,
            "status": "pending",
            "message": "预测任务正在排队处理"
        }
    elif task.state == 'STARTED':
        return {
            "request_id": request_id,
            "status": "processing",
            "message": "预测任务正在处理中"
        }
    elif task.state == 'FAILURE':
        return {
            "request_id": request_id,
            "status": "failed",
            "message": f"预测任务失败: {str(task.result)}"
        }
    
    # 如果以上都没有，说明任务不存在
    raise HTTPException(status_code=404, detail=f"未找到请求ID: {request_id}")

@app.delete("/api/v1/predictions/batch/{request_id}")
@limiter.limit(RATE_LIMIT, key_func=get_api_key)
async def delete_batch_prediction(request_id: str, req: Request):
    """
    删除批量预测结果
    """
    # 检查任务状态和结果是否存在
    status_exists = redis_client.exists(f"prediction_status:{request_id}")
    result_exists = redis_client.exists(f"prediction_result:{request_id}")
    
    if not (status_exists or result_exists):
        raise HTTPException(status_code=404, detail=f"未找到请求ID: {request_id}")
    
    # 删除任务状态和结果
    if status_exists:
        redis_client.delete(f"prediction_status:{request_id}")
    
    if result_exists:
        redis_client.delete(f"prediction_result:{request_id}")
    
    # 尝试终止正在运行的任务
    task = AsyncResult(request_id, app=celery_app)
    if task.state in ['PENDING', 'STARTED']:
        task.revoke(terminate=True)
    
    return {"message": f"已删除请求ID: {request_id}"}

@app.get("/health")
async def health_check():
    """
    健康检查
    """
    # 检查Redis连接
    redis_status = "healthy"
    try:
        redis_client.ping()
    except Exception as e:
        redis_status = f"unhealthy: {str(e)}"
    
    # 检查Celery连接
    celery_status = "healthy"
    try:
        i = celery_app.control.inspect()
        if not i.ping():
            celery_status = "unhealthy: no workers available"
    except Exception as e:
        celery_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if redis_status == "healthy" and celery_status == "healthy" else "unhealthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "api": "healthy",
            "redis": redis_status,
            "celery": celery_status
        }
    }

@app.get("/api/v1/system/queue-status")
@limiter.limit(RATE_LIMIT, key_func=get_api_key)
async def get_queue_status(req: Request):
    """
    获取队列状态
    """
    try:
        # 获取活跃任务数
        i = celery_app.control.inspect()
        active_tasks = i.active()
        if not active_tasks:
            active_tasks = {}
        
        # 统计活跃任务
        total_active = sum(len(tasks) for tasks in active_tasks.values())
        
        # 获取排队任务数 (使用Redis查询Celery队列)
        queue_length = redis_client.llen('celery')
        
        # 获取历史统计
        completed_count = 0
        failed_count = 0
        avg_processing_time = 0
        
        # 可以从Redis中获取统计数据，如果有存储
        stats = redis_client.get("prediction_stats")
        if stats:
            stats = json.loads(stats)
            completed_count = stats.get("completed_count", 0)
            failed_count = stats.get("failed_count", 0)
            avg_processing_time = stats.get("avg_processing_time", 0)
        
        return {
            "queue_length": queue_length,
            "active_tasks": total_active,
            "completed_count": completed_count,
            "failed_count": failed_count,
            "avg_processing_time": avg_processing_time,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting queue status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"获取队列状态失败: {str(e)}")

# 统计中间件 - 用于收集API使用统计
class StatsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # 只统计API调用
        if request.url.path.startswith("/api/"):
            # 增加请求计数
            redis_client.hincrby("api_stats", "request_count", 1)
            
            # 记录处理时间
            redis_client.lpush("api_processing_times", process_time)
            redis_client.ltrim("api_processing_times", 0, 999)  # 只保留最近1000个
            
            # 按路径统计
            path = request.url.path
            redis_client.hincrby("api_path_stats", path, 1)
            
            # 记录状态码
            status_code = response.status_code
            redis_client.hincrby("api_status_codes", str(status_code), 1)
        
        return response

# 添加统计中间件
app.add_middleware(StatsMiddleware)

# 启动事件
@app.on_event("startup")
async def startup_event():
    logger.info("Starting batch prediction API")
    # 创建日志目录
    os.makedirs("logs", exist_ok=True)
    
    # 初始化Redis统计数据
    if not redis_client.exists("prediction_stats"):
        redis_client.set("prediction_stats", json.dumps({
            "completed_count": 0,
            "failed_count": 0,
            "avg_processing_time": 0
        }))

# 关闭事件
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down batch prediction API")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "batch_prediction_api:app", 
        host="0.0.0.0", 
        port=8000, 
        reload=True
    ) 