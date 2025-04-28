import time
import argparse
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
from typing import Dict, List, Tuple
import logging

from model import HybridModel
from data_utils import TimeSeriesDataset, extract_market_patterns, TimeSeriesRetriever
from incremental_learner import IncrementalLearner, ModelBuffer, DataProcessor, ParallelDataProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('benchmark')


def generate_synthetic_data(num_symbols: int = 5, 
                           days_per_symbol: int = 1000, 
                           points_per_day: int = 1440) -> Dict[str, pd.DataFrame]:
    """
    Generate synthetic market data for benchmarking
    
    Args:
        num_symbols: Number of symbols to generate
        days_per_symbol: Number of days of data per symbol
        points_per_day: Number of data points per day (1440 = 1-minute data)
        
    Returns:
        Dictionary mapping symbols to DataFrames
    """
    logger.info(f"Generating synthetic data for {num_symbols} symbols...")
    
    symbols = [f"SYM{i}" for i in range(num_symbols)]
    data_dict = {}
    
    for symbol in symbols:
        # Generate dates
        start_date = pd.Timestamp('2020-01-01')
        dates = [start_date + pd.Timedelta(minutes=i) for i in range(days_per_symbol * points_per_day)]
        
        # Generate price series with random walk
        np.random.seed(int(symbol[3:]) + 42)  # Different seed for each symbol
        
        # Start with a random base price between 50 and 500
        base_price = np.random.uniform(50, 500)
        
        # Generate returns with mean and volatility
        daily_returns = np.random.normal(0.0001, 0.0015, days_per_symbol * points_per_day)
        
        # Add some seasonality
        seasonality = 0.001 * np.sin(np.linspace(0, 40 * np.pi, days_per_symbol * points_per_day))
        daily_returns += seasonality
        
        # Convert returns to price
        log_returns = np.cumsum(daily_returns)
        close_prices = base_price * np.exp(log_returns)
        
        # Generate OHLCV data
        data = {
            'timestamp': dates,
            'open': close_prices * (1 + np.random.normal(0, 0.0005, len(close_prices))),
            'high': close_prices * (1 + np.abs(np.random.normal(0, 0.001, len(close_prices)))),
            'low': close_prices * (1 - np.abs(np.random.normal(0, 0.001, len(close_prices)))),
            'close': close_prices,
            'volume': np.random.lognormal(12, 1, len(close_prices)).astype(int)
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        data_dict[symbol] = df
        
    logger.info(f"Generated {sum(len(df) for df in data_dict.values())} total data points")
    return data_dict


def setup_benchmark_db(data_dict: Dict[str, pd.DataFrame], db_url: str) -> None:
    """
    Set up database with synthetic data for benchmarking
    
    Args:
        data_dict: Dictionary mapping symbols to DataFrames
        db_url: Database URL
    """
    from sqlalchemy import create_engine, Table, Column, Integer, Float, String, DateTime, MetaData
    
    logger.info(f"Setting up database at {db_url}...")
    
    # Create engine and tables
    engine = create_engine(db_url)
    metadata = MetaData()
    
    # Create market_data table if it doesn't exist
    if not engine.dialect.has_table(engine, 'market_data'):
        market_data = Table(
            'market_data', metadata,
            Column('id', Integer, primary_key=True),
            Column('symbol', String, index=True),
            Column('timestamp', DateTime, index=True),
            Column('open', Float),
            Column('high', Float),
            Column('low', Float),
            Column('close', Float),
            Column('volume', Integer)
        )
        metadata.create_all(engine)
    
    # Insert data in batches
    total_rows = 0
    batch_size = 10000
    
    for symbol, df in data_dict.items():
        df_copy = df.reset_index()
        df_copy['symbol'] = symbol
        
        # Insert in batches
        for i in range(0, len(df_copy), batch_size):
            batch = df_copy.iloc[i:i+batch_size]
            batch.to_sql('market_data', engine, if_exists='append', index=False)
            total_rows += len(batch)
            logger.info(f"Inserted {len(batch)} rows for {symbol}. Total: {total_rows}")
    
    logger.info(f"Database setup complete with {total_rows} rows")


def benchmark_data_throughput(db_url: str, 
                             symbols: List[str], 
                             num_points: int = 100000,
                             batch_size: int = 1000) -> Dict:
    """
    Benchmark data processing throughput
    
    Args:
        db_url: Database URL
        symbols: List of symbols to process
        num_points: Number of data points to process
        batch_size: Batch size for processing
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking data throughput for {len(symbols)} symbols...")
    
    # Initialize data processor
    data_processor = DataProcessor(
        db_url=db_url,
        feature_columns=['open', 'high', 'low', 'close', 'volume'],
        sequence_length=60,
        forecast_horizon=10
    )
    
    # Initialize parallel processor
    parallel_processor = ParallelDataProcessor()
    parallel_processor.start()
    
    # Start timing
    start_time = time.time()
    
    # Process data in batches
    points_processed = 0
    for symbol in symbols:
        # Fetch data
        df = data_processor.fetch_latest_data(symbol, hours=24)
        
        # Process in batches
        for i in range(0, len(df), batch_size):
            if points_processed >= num_points:
                break
                
            batch = df.iloc[i:i+batch_size]
            data_processor.generate_training_sequences(batch)
            
            points_processed += len(batch)
            
        if points_processed >= num_points:
            break
    
    # Calculate throughput
    elapsed_time = time.time() - start_time
    throughput = points_processed / elapsed_time
    
    # Benchmark parallel processing
    start_time = time.time()
    
    # Define a simple processing function
    def process_row(row):
        # Simulate some computation
        time.sleep(0.001)
        return row['close'] * 2
    
    # Process a batch of data in parallel
    sample_df = df.iloc[:min(1000, len(df))]
    results = parallel_processor.process_batch(process_row, [row for _, row in sample_df.iterrows()])
    
    parallel_elapsed_time = time.time() - start_time
    parallel_throughput = len(sample_df) / parallel_elapsed_time
    
    # Stop parallel processor
    parallel_processor.stop()
    
    return {
        "points_processed": points_processed,
        "elapsed_time": elapsed_time,
        "throughput": throughput,
        "points_per_second": throughput,
        "parallel_throughput": parallel_throughput
    }


def benchmark_model_swapping(input_dim: int = 5,
                           hidden_dim: int = 128,
                           forecast_horizon: int = 10,
                           num_swaps: int = 10) -> Dict:
    """
    Benchmark model hot-swapping performance
    
    Args:
        input_dim: Input dimension for model
        hidden_dim: Hidden dimension for model
        forecast_horizon: Forecast horizon for model
        num_swaps: Number of model swaps to perform
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking model swapping with {num_swaps} swaps...")
    
    # Initialize model buffer
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': hidden_dim,
        'forecast_horizon': forecast_horizon
    }
    model_buffer = ModelBuffer(HybridModel, model_params)
    
    # Perform swaps and measure time
    swap_times = []
    
    for i in range(num_swaps):
        # Make some changes to shadow model to simulate training
        shadow_model = model_buffer.get_shadow_model()
        
        # Swap models and measure time
        swap_time = model_buffer.swap_models()
        swap_times.append(swap_time)
        
        logger.info(f"Swap {i+1}/{num_swaps}: {swap_time:.2f} ms")
    
    return {
        "num_swaps": num_swaps,
        "avg_swap_time": np.mean(swap_times),
        "min_swap_time": np.min(swap_times),
        "max_swap_time": np.max(swap_times),
        "swap_times": swap_times
    }


def benchmark_inference_latency(model: HybridModel,
                              sequence_length: int = 60,
                              input_dim: int = 5,
                              batch_sizes: List[int] = [1, 8, 16, 32, 64],
                              num_runs: int = 100) -> Dict:
    """
    Benchmark inference latency
    
    Args:
        model: Model to benchmark
        sequence_length: Length of input sequences
        input_dim: Input dimension
        batch_sizes: List of batch sizes to test
        num_runs: Number of inference runs for each batch size
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Benchmarking inference latency with batch sizes {batch_sizes}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    
    results = {}
    
    for batch_size in batch_sizes:
        # Create random input data
        x = torch.randn(batch_size, sequence_length, input_dim, device=device)
        
        # Warm-up runs
        for _ in range(10):
            with torch.no_grad():
                _ = model(x)
        
        # Benchmark runs
        latencies = []
        
        for _ in range(num_runs):
            start_time = time.time()
            
            with torch.no_grad():
                _ = model(x)
                
            # Ensure CUDA synchronization if using GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
                
            latency = (time.time() - start_time) * 1000  # Convert to ms
            latencies.append(latency)
        
        # Calculate statistics
        results[batch_size] = {
            "mean": np.mean(latencies),
            "min": np.min(latencies),
            "max": np.max(latencies),
            "p95": np.percentile(latencies, 95),
            "p99": np.percentile(latencies, 99)
        }
        
        logger.info(f"Batch size {batch_size}: Mean latency {results[batch_size]['mean']:.2f} ms")
    
    return results


def plot_benchmark_results(results: Dict, output_dir: str = "benchmark_results") -> None:
    """
    Plot benchmark results
    
    Args:
        results: Dictionary with benchmark results
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot data throughput
    if 'data_throughput' in results:
        throughput = results['data_throughput']['throughput']
        plt.figure(figsize=(10, 6))
        plt.bar(['Sequential', 'Parallel'], 
                [throughput, results['data_throughput']['parallel_throughput']])
        plt.ylabel('Points processed per second')
        plt.title('Data Processing Throughput')
        plt.axhline(y=100000, color='r', linestyle='--', label='Target (100k/s)')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'data_throughput.png'))
    
    # Plot model swap times
    if 'model_swapping' in results:
        swap_times = results['model_swapping']['swap_times']
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(swap_times) + 1), swap_times, 'o-')
        plt.axhline(y=200, color='r', linestyle='--', label='Target (<200ms)')
        plt.xlabel('Swap #')
        plt.ylabel('Swap time (ms)')
        plt.title('Model Swap Times')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'model_swap_times.png'))
    
    # Plot inference latency
    if 'inference_latency' in results:
        latency_results = results['inference_latency']
        batch_sizes = list(latency_results.keys())
        mean_latencies = [latency_results[bs]['mean'] for bs in batch_sizes]
        p95_latencies = [latency_results[bs]['p95'] for bs in batch_sizes]
        
        plt.figure(figsize=(10, 6))
        plt.plot(batch_sizes, mean_latencies, 'o-', label='Mean')
        plt.plot(batch_sizes, p95_latencies, 's-', label='95th percentile')
        plt.xlabel('Batch Size')
        plt.ylabel('Latency (ms)')
        plt.title('Inference Latency vs Batch Size')
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'inference_latency.png'))
    
    # Save results as JSON
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)


def run_full_benchmark(db_url: str, output_dir: str = "benchmark_results") -> Dict:
    """
    Run full benchmark suite
    
    Args:
        db_url: Database URL
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all benchmark results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate synthetic data
    data_dict = generate_synthetic_data(num_symbols=5, days_per_symbol=30, points_per_day=1440)
    
    # Set up database
    setup_benchmark_db(data_dict, db_url)
    
    # Benchmark data throughput
    data_throughput = benchmark_data_throughput(
        db_url=db_url,
        symbols=list(data_dict.keys()),
        num_points=100000,
        batch_size=1000
    )
    
    # Benchmark model swapping
    model_swapping = benchmark_model_swapping(
        input_dim=5,
        hidden_dim=128,
        forecast_horizon=10,
        num_swaps=20
    )
    
    # Benchmark inference latency
    model = HybridModel(input_dim=5, hidden_dim=128, forecast_horizon=10)
    inference_latency = benchmark_inference_latency(
        model=model,
        sequence_length=60,
        input_dim=5,
        batch_sizes=[1, 8, 16, 32, 64],
        num_runs=100
    )
    
    # Combine results
    results = {
        "data_throughput": data_throughput,
        "model_swapping": model_swapping,
        "inference_latency": inference_latency
    }
    
    # Plot results
    plot_benchmark_results(results, output_dir)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark TimeSeriesGPT incremental learning system")
    parser.add_argument("--db_url", type=str, default="sqlite:///./benchmark.db", help="Database URL")
    parser.add_argument("--output_dir", type=str, default="benchmark_results", help="Output directory for results")
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_full_benchmark(args.db_url, args.output_dir)
    
    logger.info(f"Benchmark completed. Results saved to {args.output_dir}")
    
    # Print summary
    logger.info("=== Benchmark Summary ===")
    logger.info(f"Data throughput: {results['data_throughput']['throughput']:.2f} points/sec")
    logger.info(f"Average model swap time: {results['model_swapping']['avg_swap_time']:.2f} ms")
    logger.info(f"Inference latency (batch=1): {results['inference_latency'][1]['mean']:.2f} ms") 