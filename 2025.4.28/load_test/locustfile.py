import time
import random
import logging
import os
import json
import numpy as np
import psutil
import py_spy
import threading
import requests
from datetime import datetime, timedelta
from locust import HttpUser, TaskSet, task, between, events
from prometheus_client import start_http_server, Summary, Counter, Gauge
from locust.clients import HttpSession

# Configuration
CONFIG = {
    "base_url": os.getenv("TIMESERIES_GPT_API_URL", "http://localhost:8000"),
    "prometheus_port": int(os.getenv("PROMETHEUS_PORT", "9090")),
    "api_key": os.getenv("API_KEY", "test-api-key"),
    "flamegraph_interval": int(os.getenv("FLAMEGRAPH_INTERVAL", "300")),  # in seconds
    "anomaly_injection": {
        "enabled": os.getenv("INJECT_ANOMALIES", "False").lower() == "true",
        "packet_drop_rate": float(os.getenv("PACKET_DROP_RATE", "0.5")),
        "node_failure_prob": float(os.getenv("NODE_FAILURE_PROB", "0.01")),
        "adversarial_prob": float(os.getenv("ADVERSARIAL_PROB", "0.05"))
    }
}

# Metrics
class MetricsTracker:
    def __init__(self):
        # Start Prometheus HTTP server
        start_http_server(CONFIG["prometheus_port"])
        
        # Prometheus metrics
        self.request_latency = Summary('timeseries_gpt_request_latency_seconds', 
                                      'Request latency in seconds',
                                      ['endpoint', 'user_type'])
        self.total_requests = Counter('timeseries_gpt_requests_total', 
                                     'Total number of requests',
                                     ['endpoint', 'user_type', 'status'])
        self.active_users = Gauge('timeseries_gpt_active_users', 
                                 'Number of active users',
                                 ['user_type'])
        self.error_count = Counter('timeseries_gpt_errors_total', 
                                  'Total number of errors',
                                  ['error_type'])
        self.data_throughput = Counter('timeseries_gpt_data_throughput_bytes', 
                                      'Total data throughput in bytes')
        self.anomaly_events = Counter('timeseries_gpt_anomaly_events_total', 
                                     'Total number of anomaly events injected',
                                     ['anomaly_type'])
        self.prediction_count = Counter('timeseries_gpt_predictions_total', 
                                       'Total number of time series predictions made')
        
        # Initialize user counts
        self.active_users.labels(user_type="high_frequency").set(0)
        self.active_users.labels(user_type="institutional").set(0)
        self.active_users.labels(user_type="regulatory").set(0)

metrics = MetricsTracker()

# FlameGraph Profiling
def start_profiling():
    """Generates FlameGraphs periodically to identify bottlenecks"""
    while True:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"flamegraph_{timestamp}.svg"
        try:
            logging.info(f"Generating FlameGraph: {filename}")
            py_spy.generate_flamegraph(os.getpid(), filename)
        except Exception as e:
            logging.error(f"Failed to generate FlameGraph: {e}")
        
        time.sleep(CONFIG["flamegraph_interval"])

# Start profiling in a separate thread
if os.getenv("ENABLE_PROFILING", "False").lower() == "true":
    profiling_thread = threading.Thread(target=start_profiling, daemon=True)
    profiling_thread.start()

# Time Series Data Generation
def generate_time_series_data(length=100, symbols=None, frequency='1d', trend=0.01, 
                              volatility=0.02, seasonality=True, with_anomalies=False):
    """Generate realistic time series data for testing"""
    if symbols is None:
        symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]
    
    if isinstance(symbols, str):
        symbols = [symbols]
    
    num_symbols = len(symbols)
    result = {}
    
    # Determine date range based on frequency
    end_date = datetime.now()
    if frequency == '1d':
        start_date = end_date - timedelta(days=length)
        freq_timedelta = timedelta(days=1)
    elif frequency == '1h':
        start_date = end_date - timedelta(hours=length)
        freq_timedelta = timedelta(hours=1)
    elif frequency == '1m':
        start_date = end_date - timedelta(minutes=length)
        freq_timedelta = timedelta(minutes=1)
    else:
        raise ValueError(f"Unsupported frequency: {frequency}")
    
    # Generate timestamps
    timestamps = []
    current_date = start_date
    while current_date <= end_date:
        timestamps.append(current_date.isoformat())
        current_date += freq_timedelta
    
    for symbol in symbols:
        # Start with a base price
        base_price = random.uniform(50, 500)
        
        # Generate price movement
        prices = [base_price]
        for i in range(1, length):
            # Trend component
            trend_component = trend * base_price
            
            # Volatility component
            volatility_component = volatility * base_price * random.normalvariate(0, 1)
            
            # Seasonality component (if enabled)
            seasonality_component = 0
            if seasonality:
                seasonality_component = 0.01 * base_price * np.sin(np.pi * i / (length/4))
            
            # Anomaly injection (if enabled)
            anomaly_component = 0
            if with_anomalies and random.random() < 0.05:  # 5% chance of anomaly
                anomaly_component = random.choice([-0.05, 0.05]) * base_price
            
            # Calculate new price
            new_price = prices[-1] * (1 + trend_component + volatility_component + 
                                    seasonality_component + anomaly_component)
            
            # Ensure price doesn't go negative
            new_price = max(new_price, 0.01)
            prices.append(new_price)
        
        # Format the data
        result[symbol] = {
            "timestamps": timestamps[:len(prices)],
            "values": prices
        }
    
    return result

# Anomaly Injection
class AnomalyInjection:
    """Simulates network issues and adversarial conditions"""
    
    @staticmethod
    def should_drop_packet():
        """Randomly determines if a packet should be dropped"""
        if not CONFIG["anomaly_injection"]["enabled"]:
            return False
        return random.random() < CONFIG["anomaly_injection"]["packet_drop_rate"]
    
    @staticmethod
    def should_simulate_node_failure():
        """Randomly determines if a node failure should be simulated"""
        if not CONFIG["anomaly_injection"]["enabled"]:
            return False
        return random.random() < CONFIG["anomaly_injection"]["node_failure_prob"]
    
    @staticmethod
    def generate_adversarial_sample(data):
        """Adds adversarial components to time series data"""
        if not CONFIG["anomaly_injection"]["enabled"] or data is None:
            return data
        
        if random.random() < CONFIG["anomaly_injection"]["adversarial_prob"]:
            # Make a deep copy so we don't modify the original
            if isinstance(data, dict):
                modified_data = data.copy()
                for symbol in modified_data:
                    if "values" in modified_data[symbol]:
                        values = modified_data[symbol]["values"]
                        # Add a sudden spike or drop
                        index = random.randint(0, len(values) - 1)
                        multiplier = random.choice([1.2, 0.8])  # 20% up or down
                        values[index] = values[index] * multiplier
                        metrics.anomaly_events.labels(anomaly_type="adversarial").inc()
                return modified_data
            elif isinstance(data, list):
                # For batch requests
                modified_data = data.copy()
                for i in range(len(modified_data)):
                    if random.random() < 0.1:  # Only modify 10% of the batch
                        if isinstance(modified_data[i], dict) and "values" in modified_data[i]:
                            values = modified_data[i]["values"]
                            index = random.randint(0, len(values) - 1)
                            values[index] = values[index] * random.choice([1.2, 0.8])
                metrics.anomaly_events.labels(anomaly_type="adversarial").inc()
                return modified_data
        
        return data

# Base TimeSeriesGPT User
class TimeSeriesGPTUser(HttpUser):
    abstract = True
    wait_time = between(1, 3)  # Default wait time, will be overridden by subclasses
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {CONFIG['api_key']}"
        }
    
    def on_start(self):
        self.user_type = "base"  # Will be overridden by subclasses
        metrics.active_users.labels(user_type=self.user_type).inc()
        logging.info(f"{self.user_type} user started")
    
    def on_stop(self):
        metrics.active_users.labels(user_type=self.user_type).dec()
        logging.info(f"{self.user_type} user stopped")
    
    def send_request(self, method, endpoint, data=None, name=None):
        """Custom request method with anomaly injection"""
        if AnomalyInjection.should_drop_packet():
            # Simulate packet drop
            metrics.anomaly_events.labels(anomaly_type="packet_drop").inc()
            metrics.error_count.labels(error_type="packet_drop").inc()
            metrics.total_requests.labels(endpoint=endpoint, user_type=self.user_type, status="failed").inc()
            return None
        
        if AnomalyInjection.should_simulate_node_failure():
            # Simulate node failure
            metrics.anomaly_events.labels(anomaly_type="node_failure").inc()
            metrics.error_count.labels(error_type="node_failure").inc()
            metrics.total_requests.labels(endpoint=endpoint, user_type=self.user_type, status="failed").inc()
            time.sleep(5)  # Sleep to simulate timeout
            return None
        
        # Inject adversarial samples
        if data:
            data = AnomalyInjection.generate_adversarial_sample(data)
        
        start_time = time.time()
        try:
            response = self.client.request(method, endpoint, json=data, name=name or endpoint)
            duration = time.time() - start_time
            metrics.request_latency.labels(endpoint=endpoint, user_type=self.user_type).observe(duration)
            
            if response.status_code >= 200 and response.status_code < 300:
                metrics.total_requests.labels(endpoint=endpoint, user_type=self.user_type, status="success").inc()
                content_length = len(response.content)
                metrics.data_throughput.inc(content_length)
                return response
            else:
                metrics.total_requests.labels(endpoint=endpoint, user_type=self.user_type, status="failed").inc()
                metrics.error_count.labels(error_type=f"http_{response.status_code}").inc()
                logging.error(f"Request failed: {endpoint}, Status: {response.status_code}, Response: {response.text}")
                return response
        except Exception as e:
            duration = time.time() - start_time
            metrics.request_latency.labels(endpoint=endpoint, user_type=self.user_type).observe(duration)
            metrics.total_requests.labels(endpoint=endpoint, user_type=self.user_type, status="failed").inc()
            metrics.error_count.labels(error_type="exception").inc()
            logging.error(f"Request exception: {endpoint}, Error: {str(e)}")
            return None

# High Frequency Trader User
class HighFrequencyTrader(TimeSeriesGPTUser):
    wait_time = between(0.01, 0.1)  # Fast requests (10-100 per second)
    
    def on_start(self):
        self.user_type = "high_frequency"
        super().on_start()
        # Pre-generate some symbols for consistent use
        self.symbols = ["AAPL", "GOOGL", "MSFT", "AMZN", "META", "TSLA", "NFLX", "NVDA"]
    
    @task(7)
    def predict_next_value(self):
        """Make a single prediction for a specific symbol"""
        symbol = random.choice(self.symbols)
        data = generate_time_series_data(length=30, symbols=[symbol], frequency='1m')
        
        response = self.send_request(
            "POST", 
            "/api/v1/predict", 
            data={
                "symbol": symbol,
                "data": data[symbol],
                "horizon": 5,
                "confidence_interval": 0.95
            },
            name="Single Prediction"
        )
        
        if response and response.status_code == 200:
            metrics.prediction_count.inc(1)
    
    @task(2)
    def predict_multiple_symbols(self):
        """Make predictions for multiple symbols in a single request"""
        # Select 3-5 random symbols
        selected_symbols = random.sample(self.symbols, random.randint(3, 5))
        data = generate_time_series_data(length=30, symbols=selected_symbols, frequency='1m')
        
        response = self.send_request(
            "POST", 
            "/api/v1/predict-multiple", 
            data={
                "data": data,
                "horizon": 5,
                "confidence_interval": 0.95
            },
            name="Multi-Symbol Prediction"
        )
        
        if response and response.status_code == 200:
            metrics.prediction_count.inc(len(selected_symbols))
    
    @task(1)
    def check_api_health(self):
        """Check API health status"""
        self.send_request("GET", "/health", name="Health Check")

# Institutional Client User
class InstitutionalClient(TimeSeriesGPTUser):
    wait_time = between(5, 30)  # Larger, less frequent requests
    
    def on_start(self):
        self.user_type = "institutional"
        super().on_start()
        # Generate a large set of symbols
        self.all_symbols = [f"STOCK_{i}" for i in range(1, 1001)]
    
    @task(3)
    def submit_batch_prediction(self):
        """Submit a batch prediction job"""
        # Select 100-500 symbols for batch processing
        batch_size = random.randint(100, 500)
        selected_symbols = random.sample(self.all_symbols, batch_size)
        
        data = generate_time_series_data(
            length=60, 
            symbols=selected_symbols[:10],  # Limit data generation to avoid memory issues in the test
            frequency='1d',
            seasonality=True
        )
        
        # Simulate having data for all selected symbols
        full_data = {symbol: data[list(data.keys())[i % len(data)]] for i, symbol in enumerate(selected_symbols)}
        
        response = self.send_request(
            "POST", 
            "/api/v1/batch/submit", 
            data={
                "job_name": f"batch_job_{int(time.time())}",
                "data": full_data,
                "horizon": 30,
                "confidence_interval": 0.95,
                "callback_url": "https://example.com/callback"
            },
            name="Batch Job Submission"
        )
        
        if response and response.status_code == 202:
            job_id = response.json().get("job_id")
            if job_id:
                self.job_id = job_id
                metrics.prediction_count.inc(batch_size)
    
    @task(2)
    def check_batch_status(self):
        """Check the status of a batch prediction job"""
        if hasattr(self, 'job_id'):
            self.send_request(
                "GET", 
                f"/api/v1/batch/status/{self.job_id}", 
                name="Batch Job Status"
            )
    
    @task(1)
    def download_batch_results(self):
        """Download results of a completed batch job"""
        if hasattr(self, 'job_id'):
            response = self.send_request(
                "GET", 
                f"/api/v1/batch/results/{self.job_id}", 
                name="Batch Job Results"
            )
            
            # Clean up - "forget" this job ID 20% of the time to simulate new jobs
            if random.random() < 0.2:
                delattr(self, 'job_id')

# Regulatory Auditor User
class RegulatoryAuditor(TimeSeriesGPTUser):
    wait_time = between(60, 300)  # Very infrequent but intensive requests
    
    def on_start(self):
        self.user_type = "regulatory"
        super().on_start()
    
    @task(1)
    def historical_data_scan(self):
        """Request a scan of historical predictions for compliance"""
        start_date = (datetime.now() - timedelta(days=random.randint(30, 365))).strftime("%Y-%m-%d")
        end_date = (datetime.now() - timedelta(days=random.randint(1, 29))).strftime("%Y-%m-%d")
        
        response = self.send_request(
            "POST", 
            "/api/v1/audit/historical-scan", 
            data={
                "start_date": start_date,
                "end_date": end_date,
                "symbols": ["*"],  # All symbols
                "include_metadata": True,
                "include_model_versions": True
            },
            name="Historical Scan"
        )
        
        if response and response.status_code == 202:
            scan_id = response.json().get("scan_id")
            if scan_id:
                self.scan_id = scan_id
    
    @task(2)
    def check_scan_status(self):
        """Check the status of a historical scan"""
        if hasattr(self, 'scan_id'):
            self.send_request(
                "GET", 
                f"/api/v1/audit/scan-status/{self.scan_id}", 
                name="Scan Status"
            )
    
    @task(2)
    def performance_metrics_report(self):
        """Request performance metrics for a time period"""
        start_date = (datetime.now() - timedelta(days=random.randint(7, 90))).strftime("%Y-%m-%d")
        end_date = datetime.now().strftime("%Y-%m-%d")
        
        self.send_request(
            "POST", 
            "/api/v1/audit/performance-report", 
            data={
                "start_date": start_date,
                "end_date": end_date,
                "metrics": ["rmse", "mae", "mape", "r2", "prediction_bias"],
                "group_by": ["symbol", "model_version"]
            },
            name="Performance Report"
        )

# Event hooks for reporting
@events.request_failure.add_listener
def request_failure_handler(request_type, name, response_time, exception, **kwargs):
    logging.error(f"Request failure: {name}, Error: {str(exception)}")

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    logging.info(f"Load test started with {environment.runner.user_count} users")
    logging.info(f"Anomaly injection: {'enabled' if CONFIG['anomaly_injection']['enabled'] else 'disabled'}")

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    logging.info("Load test completed")
    # Generate summary report
    try:
        report = {
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": int(time.time() - environment.runner.start_time),
            "user_count": environment.runner.user_count,
            "request_stats": {
                "total": environment.stats.total.num_requests,
                "failures": environment.stats.total.num_failures,
                "median_response_time": environment.stats.total.median_response_time,
                "avg_response_time": environment.stats.total.avg_response_time,
                "min_response_time": environment.stats.total.min_response_time,
                "max_response_time": environment.stats.total.max_response_time,
                "current_rps": environment.stats.total.current_rps,
                "total_rps": environment.stats.total.total_rps,
            },
            "anomalies_injected": {
                "packet_drops": int(metrics.anomaly_events._metrics.get(("packet_drop",), {}).get("count", 0)),
                "node_failures": int(metrics.anomaly_events._metrics.get(("node_failure",), {}).get("count", 0)),
                "adversarial": int(metrics.anomaly_events._metrics.get(("adversarial",), {}).get("count", 0)),
            }
        }
        
        # Save report to file
        with open(f"load_test_report_{int(time.time())}.json", "w") as f:
            json.dump(report, f, indent=2)
        
        logging.info(f"Test Summary: {json.dumps(report, indent=2)}")
    except Exception as e:
        logging.error(f"Failed to generate summary report: {e}") 