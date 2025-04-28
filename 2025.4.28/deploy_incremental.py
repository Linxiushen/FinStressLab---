import os
import logging
import threading
import time
import argparse
import torch
import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any

from incremental_learner import IncrementalLearner, DataProcessor
from model import HybridModel
from api import app as base_app
from api import model_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('deploy_incremental')

# Create FastAPI app based on the base app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add all routes from the base app
app.include_router(base_app.router)

# Global variable to hold incremental learner
incremental_learner = None


class IncrementalConfig(BaseModel):
    """Configuration model for incremental learning"""
    symbols: List[str]
    input_dim: int = 5
    hidden_dim: int = 128
    forecast_horizon: int = 10
    sequence_length: int = 60
    sliding_window_step: int = 5
    batch_size: int = 64
    swap_threshold: float = 0.05
    min_swap_interval: int = 3600
    fetch_interval: int = 3600
    checkpoint_dir: str = 'checkpoints/incremental'
    enable_gpu_optimization: bool = True
    gpu_memory_fraction: float = 0.8


@app.post("/start-incremental-learning")
async def start_incremental_learning(config: IncrementalConfig, background_tasks: BackgroundTasks):
    """
    Start incremental learning process
    
    Args:
        config: Configuration for incremental learning
    """
    global incremental_learner
    
    try:
        # Initialize incremental learner
        db_url = os.environ.get("DATABASE_URL", "sqlite:///./timeseries_gpt.db")
        
        # GPU memory optimization
        if config.enable_gpu_optimization and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.set_per_process_memory_fraction(config.gpu_memory_fraction)
            logger.info(f"GPU memory fraction set to {config.gpu_memory_fraction}")
        
        incremental_learner = IncrementalLearner(
            db_url=db_url,
            symbols=config.symbols,
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            forecast_horizon=config.forecast_horizon,
            sequence_length=config.sequence_length,
            sliding_window_step=config.sliding_window_step,
            batch_size=config.batch_size,
            swap_threshold=config.swap_threshold,
            min_swap_interval=config.min_swap_interval,
            checkpoint_dir=config.checkpoint_dir
        )
        
        # Start incremental learning in background
        background_tasks.add_task(incremental_learner.start_incremental_learning)
        
        # Setup model manager integration
        setup_model_manager_integration(incremental_learner, model_manager)
        
        return {"status": "success", "message": "Incremental learning started"}
    except Exception as e:
        logger.error(f"Error starting incremental learning: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting incremental learning: {str(e)}")


@app.get("/incremental-status")
async def get_incremental_status():
    """Get status of incremental learning"""
    global incremental_learner
    
    if incremental_learner is None:
        raise HTTPException(status_code=404, detail="Incremental learning not started")
    
    try:
        metrics = incremental_learner.get_metrics()
        return metrics
    except Exception as e:
        logger.error(f"Error getting incremental status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting incremental status: {str(e)}")


@app.post("/force-model-swap")
async def force_model_swap():
    """Force swap of active and shadow models"""
    global incremental_learner
    
    if incremental_learner is None:
        raise HTTPException(status_code=404, detail="Incremental learning not started")
    
    if not incremental_learner.model_buffer.is_shadow_ready():
        raise HTTPException(status_code=400, detail="Shadow model is not ready for swapping")
    
    try:
        swap_time = incremental_learner.model_buffer.swap_models()
        incremental_learner.metrics['swap_times'].append(swap_time)
        incremental_learner.metrics['active_model_loss'] = incremental_learner.metrics['shadow_model_loss']
        
        # Save models after forced swap
        incremental_learner.save_state()
        
        return {
            "status": "success", 
            "message": f"Models swapped in {swap_time:.2f} ms",
            "swap_time_ms": swap_time
        }
    except Exception as e:
        logger.error(f"Error forcing model swap: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error forcing model swap: {str(e)}")


@app.post("/save-incremental-state")
async def save_incremental_state(path: Optional[str] = None):
    """Save incremental learner state"""
    global incremental_learner
    
    if incremental_learner is None:
        raise HTTPException(status_code=404, detail="Incremental learning not started")
    
    try:
        incremental_learner.save_state(path)
        return {"status": "success", "message": f"State saved to {path or incremental_learner.checkpoint_dir}"}
    except Exception as e:
        logger.error(f"Error saving incremental state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error saving incremental state: {str(e)}")


@app.post("/load-incremental-state")
async def load_incremental_state(path: Optional[str] = None):
    """Load incremental learner state"""
    global incremental_learner
    
    if incremental_learner is None:
        raise HTTPException(status_code=404, detail="Incremental learning not started")
    
    try:
        incremental_learner.load_state(path)
        
        # Update model manager with the active model
        active_model = incremental_learner.model_buffer.get_active_model()
        model_manager.model = active_model
        
        return {"status": "success", "message": f"State loaded from {path or incremental_learner.checkpoint_dir}"}
    except Exception as e:
        logger.error(f"Error loading incremental state: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading incremental state: {str(e)}")


def setup_model_manager_integration(learner: IncrementalLearner, manager: Any):
    """
    Setup integration between incremental learner and model manager
    
    Args:
        learner: IncrementalLearner instance
        manager: ModelManager instance
    """
    # Use the active model from incremental learner for inference
    active_model = learner.model_buffer.get_active_model()
    manager.model = active_model
    manager.is_loaded = True
    
    # Share data between incremental learner and model manager
    for symbol in learner.symbols:
        if symbol in learner.data_processor.patterns_cache:
            retriever = learner.data_processor.patterns_cache[symbol]
            manager.add_symbol_data(symbol, pd.DataFrame(), extract_patterns=False)
            manager.symbols[symbol]["retriever"] = retriever
    
    # Setup automatic model updates
    def model_update_monitor():
        while True:
            try:
                if learner.model_buffer.is_shadow_ready():
                    # If shadow model is ready and significantly better, update the model manager
                    improvement = (learner.metrics['active_model_loss'] - learner.metrics['shadow_model_loss']) / learner.metrics['active_model_loss']
                    
                    if improvement > learner.swap_threshold:
                        logger.info(f"Model improvement detected: {improvement:.2%}. Updating model manager...")
                        
                        # Swap models
                        swap_time = learner.model_buffer.swap_models()
                        learner.metrics['swap_times'].append(swap_time)
                        learner.metrics['active_model_loss'] = learner.metrics['shadow_model_loss']
                        
                        # Update model manager with the new active model
                        active_model = learner.model_buffer.get_active_model()
                        manager.model = active_model
                        
                        # Save state
                        learner.save_state()
                        
                        logger.info(f"Model manager updated. Swap completed in {swap_time:.2f} ms")
            except Exception as e:
                logger.error(f"Error in model update monitor: {str(e)}")
            
            # Check every 60 seconds
            time.sleep(60)
    
    # Start model update monitor in background
    thread = threading.Thread(target=model_update_monitor, daemon=True)
    thread.start()
    logger.info("Model manager integration setup complete")


class MiddlewareMonitor:
    """Middleware to monitor API performance"""
    
    def __init__(self, app):
        self.app = app
        self.request_counts = {}
        self.request_times = {}
        self.lock = threading.Lock()
    
    async def __call__(self, request: Request, call_next):
        # Track API performance
        path = request.url.path
        start_time = time.time()
        
        response = await call_next(request)
        
        # Calculate request time
        request_time = time.time() - start_time
        
        # Update statistics
        with self.lock:
            if path not in self.request_counts:
                self.request_counts[path] = 0
                self.request_times[path] = []
            
            self.request_counts[path] += 1
            self.request_times[path].append(request_time)
            
            # Keep only the last 100 request times
            if len(self.request_times[path]) > 100:
                self.request_times[path].pop(0)
        
        return response
    
    def get_statistics(self):
        """Get API performance statistics"""
        stats = {}
        
        with self.lock:
            for path in self.request_counts:
                if self.request_times[path]:
                    avg_time = sum(self.request_times[path]) / len(self.request_times[path])
                    max_time = max(self.request_times[path])
                    min_time = min(self.request_times[path])
                    
                    stats[path] = {
                        "count": self.request_counts[path],
                        "avg_time": avg_time,
                        "max_time": max_time,
                        "min_time": min_time
                    }
        
        return stats


# Add middleware
middleware_monitor = MiddlewareMonitor(app)
app.middleware("http")(middleware_monitor)


@app.get("/api-performance")
async def get_api_performance():
    """Get API performance statistics"""
    return middleware_monitor.get_statistics()


def start_server(host: str = "0.0.0.0", port: int = 8000):
    """
    Start API server
    
    Args:
        host: Host to bind to
        port: Port to bind to
    """
    uvicorn.run("deploy_incremental:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy TimeSeriesGPT with incremental learning")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--db_url", type=str, help="Database URL (overrides environment variable)")
    parser.add_argument("--gpu_memory_fraction", type=float, default=0.8, help="Fraction of GPU memory to use")
    
    args = parser.parse_args()
    
    # Set environment variables
    if args.db_url:
        os.environ["DATABASE_URL"] = args.db_url
    
    # Set GPU memory fraction
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction)
        logger.info(f"GPU memory fraction set to {args.gpu_memory_fraction}")
    
    # Start server
    start_server(args.host, args.port) 