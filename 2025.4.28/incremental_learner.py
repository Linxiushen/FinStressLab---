import os
import time
import copy
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from typing import Dict, List, Tuple, Optional, Union, Any
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text, func
from sqlalchemy.orm import sessionmaker
import logging
import queue
import multiprocessing as mp
from functools import partial

from model import HybridModel
from data_utils import TimeSeriesDataset, extract_market_patterns, TimeSeriesRetriever
from train import DynamicWeightTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('incremental_learner')


class ModelBuffer:
    """
    Dual-buffer system for hot-swapping models.
    
    Maintains two models:
    - active_model: used for inference
    - shadow_model: being updated in the background
    """
    
    def __init__(self, 
                 model_class: type, 
                 model_params: Dict[str, Any],
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize model buffer with two identical models
        
        Args:
            model_class: Class of the model to instantiate
            model_params: Parameters for model initialization
            device: Device to place models on
        """
        self.device = device
        self.model_class = model_class
        self.model_params = model_params
        
        # Create active model for inference
        self.active_model = model_class(**model_params).to(device)
        self.active_model.eval()
        
        # Create shadow model for training
        self.shadow_model = model_class(**model_params).to(device)
        self.shadow_model.train()
        
        # Model state trackers
        self.last_swap_time = time.time()
        self.shadow_model_ready = False
        self.swap_lock = threading.Lock()
        
    def swap_models(self) -> float:
        """
        Hot-swap active and shadow models
        
        Returns:
            Time taken to swap models in milliseconds
        """
        start_time = time.time()
        
        with self.swap_lock:
            # Set both models to eval mode during swap
            self.active_model.eval()
            self.shadow_model.eval()
            
            # Perform hot-swap using shallow copy for speed
            # Critical section - minimize time here
            self.active_model, self.shadow_model = self.shadow_model, self.active_model
            
            # Reset modes after swap
            self.active_model.eval()
            self.shadow_model.train()
            
            # Update state trackers
            self.last_swap_time = time.time()
            self.shadow_model_ready = False
        
        swap_time_ms = (time.time() - start_time) * 1000
        logger.info(f"Model swap completed in {swap_time_ms:.2f} ms")
        
        return swap_time_ms
    
    def get_active_model(self) -> nn.Module:
        """Get the currently active model for inference"""
        return self.active_model
    
    def get_shadow_model(self) -> nn.Module:
        """Get the shadow model for training"""
        return self.shadow_model
    
    def mark_shadow_ready(self) -> None:
        """Mark that the shadow model is ready to be swapped in"""
        self.shadow_model_ready = True
        
    def is_shadow_ready(self) -> bool:
        """Check if shadow model is ready to be swapped"""
        return self.shadow_model_ready
    
    def time_since_last_swap(self) -> float:
        """Get time in seconds since the last model swap"""
        return time.time() - self.last_swap_time
    
    def save_models(self, path: str) -> None:
        """Save both models to disk"""
        os.makedirs(path, exist_ok=True)
        active_path = os.path.join(path, "active_model.pt")
        shadow_path = os.path.join(path, "shadow_model.pt")
        
        torch.save(self.active_model.state_dict(), active_path)
        torch.save(self.shadow_model.state_dict(), shadow_path)
        logger.info(f"Models saved to {path}")
    
    def load_models(self, path: str) -> None:
        """Load both models from disk"""
        active_path = os.path.join(path, "active_model.pt")
        shadow_path = os.path.join(path, "shadow_model.pt")
        
        with self.swap_lock:
            self.active_model.load_state_dict(torch.load(active_path, map_location=self.device))
            self.shadow_model.load_state_dict(torch.load(shadow_path, map_location=self.device))
        
        logger.info(f"Models loaded from {path}")


class TimeSeriesGPTModule(pl.LightningModule):
    """
    PyTorch Lightning module for TimeSeriesGPT model
    """
    
    def __init__(self, 
                 model: HybridModel,
                 learning_rate: float = 1e-4):
        """
        Initialize Lightning module
        
        Args:
            model: HybridModel instance
            learning_rate: Learning rate for optimizer
        """
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        
        # Save hyperparameters for checkpointing
        self.save_hyperparameters(ignore=['model'])
        
    def forward(self, x, retrieval_vectors=None, prev_errors=None):
        """Forward pass"""
        return self.model(x, retrieval_vectors, prev_errors)
    
    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "frequency": 1
            }
        }
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        x, y = batch
        
        # Forward pass (no retrieval vectors in basic training)
        outputs = self.model(x)
        
        # Calculate loss
        prediction = outputs['prediction']
        loss = self.mse_loss(prediction, y)
        
        # Log metrics
        self.log('train_loss', loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        x, y = batch
        
        # Forward pass
        outputs = self.model(x)
        
        # Calculate loss
        prediction = outputs['prediction']
        loss = self.mse_loss(prediction, y)
        
        # Log metrics
        self.log('val_loss', loss, prog_bar=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        x, y = batch
        
        # Forward pass
        outputs = self.model(x)
        
        # Calculate loss
        prediction = outputs['prediction']
        loss = self.mse_loss(prediction, y)
        
        # Log metrics
        self.log('test_loss', loss)
        
        return loss


class DataProcessor:
    """
    Process data from TimescaleDB for incremental learning
    """
    
    def __init__(self, 
                 db_url: str,
                 feature_columns: List[str],
                 sequence_length: int = 60,
                 forecast_horizon: int = 10,
                 sliding_window_step: int = 5,
                 max_queue_size: int = 100):
        """
        Initialize data processor
        
        Args:
            db_url: URL for database connection
            feature_columns: List of column names to use as features
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            sliding_window_step: Step size for sliding window
            max_queue_size: Maximum size of the data queue
        """
        self.db_url = db_url
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        
        self.feature_columns = feature_columns
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.sliding_window_step = sliding_window_step
        
        # Data buffer for processing
        self.data_queue = queue.Queue(maxsize=max_queue_size)
        self.processing_lock = threading.Lock()
        
        # Scaler for standardizing data
        self.scaler = None
        self.patterns_cache = {}
        
    def fetch_latest_data(self, symbol: str, hours: int = 1) -> pd.DataFrame:
        """
        Fetch the latest data from TimescaleDB
        
        Args:
            symbol: Stock symbol to fetch
            hours: Number of hours of data to fetch
            
        Returns:
            DataFrame with latest data
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        with self.Session() as session:
            query = text(f"""
                SELECT timestamp, open, high, low, close, volume
                FROM market_data
                WHERE symbol = :symbol AND timestamp >= :cutoff_time
                ORDER BY timestamp ASC
            """)
            
            result = session.execute(query, {"symbol": symbol, "cutoff_time": cutoff_time})
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            
            if not df.empty:
                df.set_index('timestamp', inplace=True)
                
        return df
    
    def generate_training_sequences(self, df: pd.DataFrame) -> TimeSeriesDataset:
        """
        Generate training sequences using sliding window
        
        Args:
            df: DataFrame with time series data
            
        Returns:
            TimeSeriesDataset with training sequences
        """
        # Apply sliding window mechanism
        dataset = TimeSeriesDataset(
            data=df,
            feature_columns=self.feature_columns,
            sequence_length=self.sequence_length,
            forecast_horizon=self.forecast_horizon,
            train=True,
            scale_data=True
        )
        
        if self.scaler is None:
            self.scaler = dataset.scaler
        else:
            # For incremental learning, we use the existing scaler
            dataset.scaler = self.scaler
            
        return dataset
    
    def start_data_fetcher(self, symbols: List[str], fetch_interval: int = 3600):
        """
        Start background thread to fetch data periodically
        
        Args:
            symbols: List of symbols to fetch
            fetch_interval: Interval in seconds between fetches
        """
        def fetcher_job():
            while True:
                try:
                    for symbol in symbols:
                        df = self.fetch_latest_data(symbol)
                        if not df.empty:
                            with self.processing_lock:
                                if self.data_queue.full():
                                    # Remove oldest item if queue is full
                                    try:
                                        self.data_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    
                                self.data_queue.put((symbol, df))
                                logger.info(f"Fetched {len(df)} rows for {symbol}")
                    
                    # Sleep until next fetch
                    time.sleep(fetch_interval)
                    
                except Exception as e:
                    logger.error(f"Error in data fetcher: {str(e)}")
                    time.sleep(60)  # Wait a minute before retrying
        
        thread = threading.Thread(target=fetcher_job, daemon=True)
        thread.start()
        logger.info(f"Data fetcher started for symbols: {symbols}")
        
        return thread
    
    def get_next_batch(self) -> Tuple[str, TimeSeriesDataset]:
        """
        Get the next batch of data for training
        
        Returns:
            Tuple of (symbol, dataset)
        """
        symbol, df = self.data_queue.get()
        dataset = self.generate_training_sequences(df)
        return symbol, dataset
    
    def extract_patterns(self, symbol: str, df: pd.DataFrame) -> TimeSeriesRetriever:
        """
        Extract patterns for a symbol
        
        Args:
            symbol: Symbol to extract patterns for
            df: DataFrame with historical data
            
        Returns:
            TimeSeriesRetriever with patterns
        """
        # Use cached patterns if available
        if symbol in self.patterns_cache:
            return self.patterns_cache[symbol]
            
        # Extract patterns
        patterns = extract_market_patterns(df, 'close', window_size=30, min_patterns=50)
        retriever = TimeSeriesRetriever(patterns, top_k=5)
        
        # Cache patterns
        self.patterns_cache[symbol] = retriever
        
        return retriever


class IncrementalLearner:
    """
    Main class for incremental learning system
    """
    
    def __init__(self, 
                 db_url: str,
                 symbols: List[str],
                 input_dim: int = 5,
                 hidden_dim: int = 128,
                 forecast_horizon: int = 10,
                 sequence_length: int = 60,
                 sliding_window_step: int = 5,
                 batch_size: int = 64,
                 swap_threshold: float = 0.05,  # Threshold for model swapping (validation loss improvement)
                 min_swap_interval: int = 3600,  # Minimum time between swaps in seconds
                 learning_rate: float = 1e-4,
                 checkpoint_dir: str = 'checkpoints/incremental',
                 num_workers: int = 8):
        """
        Initialize incremental learner
        
        Args:
            db_url: URL for database connection
            symbols: List of symbols to train on
            input_dim: Input dimension for model
            hidden_dim: Hidden dimension for model
            forecast_horizon: Number of steps to forecast
            sequence_length: Length of input sequences
            sliding_window_step: Step size for sliding window
            batch_size: Batch size for training
            swap_threshold: Threshold for model swapping
            min_swap_interval: Minimum time between swaps in seconds
            learning_rate: Learning rate for optimizer
            checkpoint_dir: Directory for checkpoints
            num_workers: Number of worker processes for data loading
        """
        self.db_url = db_url
        self.symbols = symbols
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.swap_threshold = swap_threshold
        self.min_swap_interval = min_swap_interval
        self.learning_rate = learning_rate
        self.checkpoint_dir = checkpoint_dir
        self.num_workers = num_workers
        
        # Create directories
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize feature columns
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Initialize model buffer
        model_params = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'forecast_horizon': forecast_horizon
        }
        self.model_buffer = ModelBuffer(HybridModel, model_params)
        
        # Initialize data processor
        self.data_processor = DataProcessor(
            db_url=db_url,
            feature_columns=self.feature_columns,
            sequence_length=sequence_length,
            forecast_horizon=forecast_horizon,
            sliding_window_step=sliding_window_step
        )
        
        # Initialize metrics tracker
        self.metrics = {
            'active_model_loss': float('inf'),
            'shadow_model_loss': float('inf'),
            'swap_times': [],
            'training_iterations': 0
        }
        
        # Performance tracking
        self.data_process_time = 0
        self.training_time = 0
        self.inference_time = 0
        
        # Initialize Lightning module for shadow model
        self.pl_module = TimeSeriesGPTModule(
            model=self.model_buffer.get_shadow_model(),
            learning_rate=learning_rate
        )
        
        # Initialize callbacks
        self.callbacks = [
            ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='{epoch}-{val_loss:.4f}',
                save_top_k=3,
                monitor='val_loss',
                mode='min'
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                mode='min'
            )
        ]
        
        # GPU memory optimization
        self.gpu_memory_fraction = 0.8  # Reserve 20% for active model
        if torch.cuda.is_available():
            # Limit to fraction of GPU memory for training
            torch.cuda.set_per_process_memory_fraction(self.gpu_memory_fraction)
            # Enable memory caching for faster swapping
            torch.cuda.empty_cache()
            
        logger.info("Incremental learner initialized")
            
    def start_incremental_learning(self):
        """Start the incremental learning process"""
        # Start data fetcher in background
        self.data_processor.start_data_fetcher(self.symbols)
        
        # Initial datasets for all symbols
        initial_datasets = []
        initial_retrievers = {}
        
        logger.info("Fetching initial data for all symbols...")
        for symbol in self.symbols:
            # Fetch more data initially (24 hours)
            df = self.data_processor.fetch_latest_data(symbol, hours=24)
            if not df.empty:
                dataset = self.data_processor.generate_training_sequences(df)
                initial_datasets.append(dataset)
                
                # Extract patterns for each symbol
                retriever = self.data_processor.extract_patterns(symbol, df)
                initial_retrievers[symbol] = retriever
                
                logger.info(f"Initial data for {symbol}: {len(dataset)} sequences")
        
        if not initial_datasets:
            logger.error("No initial data available. Exiting.")
            return
        
        # Combine all initial datasets
        combined_dataset = ConcatDataset(initial_datasets)
        
        # Split into train/val
        train_size = int(len(combined_dataset) * 0.8)
        val_size = len(combined_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            combined_dataset, [train_size, val_size]
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        # Initial training of shadow model
        logger.info("Starting initial training of shadow model...")
        trainer = pl.Trainer(
            max_epochs=20,
            callbacks=self.callbacks,
            gpus=1 if torch.cuda.is_available() else 0,
            logger=True
        )
        
        trainer.fit(self.pl_module, train_loader, val_loader)
        
        # Mark shadow model as ready for first swap
        self.model_buffer.mark_shadow_ready()
        self.metrics['shadow_model_loss'] = self.pl_module.trainer.callback_metrics['val_loss'].item()
        
        # Perform initial swap
        self.model_buffer.swap_models()
        self.metrics['active_model_loss'] = self.metrics['shadow_model_loss']
        
        # Start continuous learning loop
        logger.info("Starting continuous learning loop...")
        
        def continuous_learning():
            """Background thread for continuous learning"""
            while True:
                try:
                    # Process new data as it arrives
                    symbol, new_dataset = self.data_processor.get_next_batch()
                    
                    # Skip if dataset is too small
                    if len(new_dataset) < self.batch_size:
                        logger.warning(f"Dataset for {symbol} too small, skipping")
                        continue
                    
                    start_time = time.time()
                    
                    # Split into train/val
                    train_size = int(len(new_dataset) * 0.8)
                    val_size = len(new_dataset) - train_size
                    train_dataset, val_dataset = torch.utils.data.random_split(
                        new_dataset, [train_size, val_size]
                    )
                    
                    # Create data loaders
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=self.batch_size,
                        shuffle=True,
                        num_workers=self.num_workers,
                        pin_memory=True
                    )
                    
                    val_loader = DataLoader(
                        val_dataset,
                        batch_size=self.batch_size,
                        shuffle=False,
                        num_workers=self.num_workers,
                        pin_memory=True
                    )
                    
                    # Get retriever for this symbol
                    retriever = self.data_processor.patterns_cache.get(symbol)
                    
                    # Fine-tune shadow model
                    logger.info(f"Fine-tuning shadow model with new data from {symbol}...")
                    
                    trainer = pl.Trainer(
                        max_epochs=5,  # Fewer epochs for incremental updates
                        callbacks=self.callbacks,
                        gpus=1 if torch.cuda.is_available() else 0,
                        logger=True
                    )
                    
                    trainer.fit(self.pl_module, train_loader, val_loader)
                    
                    # Get validation loss
                    val_loss = self.pl_module.trainer.callback_metrics['val_loss'].item()
                    self.metrics['shadow_model_loss'] = val_loss
                    self.metrics['training_iterations'] += 1
                    
                    training_time = time.time() - start_time
                    self.training_time = training_time
                    logger.info(f"Training completed in {training_time:.2f} seconds. Val loss: {val_loss:.6f}")
                    
                    # Check if we should swap models
                    time_since_swap = self.model_buffer.time_since_last_swap()
                    improvement = (self.metrics['active_model_loss'] - val_loss) / self.metrics['active_model_loss']
                    
                    if (improvement > self.swap_threshold and 
                        time_since_swap > self.min_swap_interval):
                        logger.info(f"Model improvement: {improvement:.2%}. Swapping models...")
                        
                        # Mark shadow model as ready
                        self.model_buffer.mark_shadow_ready()
                        
                        # Perform swap
                        swap_time = self.model_buffer.swap_models()
                        self.metrics['swap_times'].append(swap_time)
                        self.metrics['active_model_loss'] = val_loss
                        
                        # Save models after swap
                        self.model_buffer.save_models(self.checkpoint_dir)
                        
                    # Sleep briefly to avoid CPU hogging
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error in continuous learning: {str(e)}")
                    time.sleep(10)  # Wait before retrying
        
        # Start background thread for continuous learning
        thread = threading.Thread(target=continuous_learning, daemon=True)
        thread.start()
        
        return thread
    
    def predict(self, 
                x: torch.Tensor, 
                symbol: str = None,
                with_uncertainty: bool = True) -> Dict[str, np.ndarray]:
        """
        Make a prediction with the active model
        
        Args:
            x: Input tensor [batch_size, seq_len, input_dim]
            symbol: Symbol for retrieval (optional)
            with_uncertainty: Whether to include uncertainty estimates
            
        Returns:
            Prediction dictionary
        """
        start_time = time.time()
        
        # Get active model
        model = self.model_buffer.get_active_model()
        
        # Get retriever if symbol is provided
        retrieval_vectors = None
        if symbol is not None and symbol in self.data_processor.patterns_cache:
            retriever = self.data_processor.patterns_cache[symbol]
            sample_x = x[0].cpu().numpy()
            retrieval_vectors = retriever.get_retrieval_tensor(sample_x, self.hidden_dim)
            retrieval_vectors = retrieval_vectors.to(x.device)
            
        # Make prediction
        with torch.no_grad():
            if with_uncertainty:
                outputs = model.predict_with_uncertainty(x, retrieval_vectors)
            else:
                outputs = model(x, retrieval_vectors)
                
        # Convert to numpy
        result = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy()
            else:
                result[key] = value
                
        inference_time = time.time() - start_time
        self.inference_time = inference_time
        
        return result
    
    def get_metrics(self) -> Dict:
        """Get current metrics"""
        return {
            **self.metrics,
            'data_process_time': self.data_process_time,
            'training_time': self.training_time,
            'inference_time': self.inference_time * 1000,  # Convert to ms
            'shadow_model_ready': self.model_buffer.is_shadow_ready(),
            'time_since_last_swap': self.model_buffer.time_since_last_swap()
        }
    
    def save_state(self, path: str = None) -> None:
        """Save learner state to disk"""
        if path is None:
            path = self.checkpoint_dir
            
        os.makedirs(path, exist_ok=True)
        
        # Save models
        self.model_buffer.save_models(path)
        
        # Save metrics
        metrics_path = os.path.join(path, "metrics.json")
        with open(metrics_path, 'w') as f:
            import json
            json.dump(self.metrics, f)
            
        logger.info(f"Learner state saved to {path}")
    
    def load_state(self, path: str = None) -> None:
        """Load learner state from disk"""
        if path is None:
            path = self.checkpoint_dir
            
        # Load models
        self.model_buffer.load_models(path)
        
        # Load metrics if available
        metrics_path = os.path.join(path, "metrics.json")
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                import json
                self.metrics = json.load(f)
                
        logger.info(f"Learner state loaded from {path}")


# High-performance data processing utilities
class ParallelDataProcessor:
    """
    Parallel data processor for high-throughput data processing
    """
    
    def __init__(self, num_processes: int = mp.cpu_count()):
        """
        Initialize parallel data processor
        
        Args:
            num_processes: Number of parallel processes
        """
        self.num_processes = num_processes
        self.process_pool = None
        
    def start(self):
        """Start the process pool"""
        if self.process_pool is None:
            self.process_pool = mp.Pool(processes=self.num_processes)
            logger.info(f"Started parallel data processor with {self.num_processes} processes")
            
    def stop(self):
        """Stop the process pool"""
        if self.process_pool is not None:
            self.process_pool.close()
            self.process_pool.join()
            self.process_pool = None
            logger.info("Stopped parallel data processor")
            
    def process_batch(self, func, data_batch, *args, **kwargs):
        """
        Process a batch of data in parallel
        
        Args:
            func: Function to apply to each item
            data_batch: Batch of data items
            *args, **kwargs: Additional arguments for the function
            
        Returns:
            List of results
        """
        if self.process_pool is None:
            self.start()
            
        partial_func = partial(func, *args, **kwargs)
        return self.process_pool.map(partial_func, data_batch)


if __name__ == "__main__":
    # Example usage
    db_url = os.environ.get("DATABASE_URL", "sqlite:///./timeseries_gpt.db")
    symbols = ["AAPL", "MSFT", "GOOG", "AMZN", "TSLA"]
    
    # Create incremental learner
    learner = IncrementalLearner(
        db_url=db_url,
        symbols=symbols,
        input_dim=5,
        hidden_dim=128,
        forecast_horizon=10,
        sequence_length=60,
        batch_size=64
    )
    
    # Start incremental learning
    learner.start_incremental_learning()
    
    # Keep the main thread alive
    try:
        while True:
            # Print metrics occasionally
            metrics = learner.get_metrics()
            logger.info(f"Metrics: {metrics}")
            time.sleep(3600)  # Sleep for an hour
    except KeyboardInterrupt:
        # Save state on exit
        learner.save_state()
        logger.info("Incremental learner stopped and state saved") 