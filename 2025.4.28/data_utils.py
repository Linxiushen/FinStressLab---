import torch
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict, Optional, Union
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


def dynamic_time_warping(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Dynamic Time Warping (DTW) distance between two time series
    
    Args:
        x: First time series of shape (n, feature_dim)
        y: Second time series of shape (m, feature_dim)
        
    Returns:
        DTW distance between x and y
    """
    n, m = len(x), len(y)
    
    # Compute pairwise distances between points
    dist_matrix = cdist(x, y, metric='euclidean')
    
    # Initialize cost matrix
    cost_matrix = np.zeros((n+1, m+1))
    cost_matrix[0, 1:] = np.inf
    cost_matrix[1:, 0] = np.inf
    
    # Fill cost matrix
    for i in range(1, n+1):
        for j in range(1, m+1):
            cost_matrix[i, j] = dist_matrix[i-1, j-1] + min(
                cost_matrix[i-1, j],     # insertion
                cost_matrix[i, j-1],     # deletion
                cost_matrix[i-1, j-1]    # match
            )
    
    return cost_matrix[n, m]


class TimeSeriesRetriever:
    """Time series pattern retrieval using DTW"""
    
    def __init__(self, reference_patterns: Dict[str, np.ndarray], top_k: int = 5):
        """
        Initialize the retriever with reference patterns
        
        Args:
            reference_patterns: Dictionary mapping pattern names to time series
            top_k: Number of similar patterns to retrieve
        """
        self.reference_patterns = reference_patterns
        self.pattern_names = list(reference_patterns.keys())
        self.top_k = top_k
        
    def retrieve_similar_patterns(self, query: np.ndarray) -> Tuple[List[str], List[np.ndarray], np.ndarray]:
        """
        Retrieve similar patterns to the query
        
        Args:
            query: Query time series of shape (n, feature_dim)
            
        Returns:
            Tuple of (pattern_names, pattern_series, similarity_scores)
        """
        scores = []
        
        # Compute DTW distance to each reference pattern
        for pattern_name in self.pattern_names:
            reference = self.reference_patterns[pattern_name]
            distance = dynamic_time_warping(query, reference)
            scores.append((pattern_name, distance))
        
        # Sort by distance (ascending)
        scores.sort(key=lambda x: x[1])
        
        # Get top-k most similar patterns
        top_patterns = scores[:self.top_k]
        pattern_names = [p[0] for p in top_patterns]
        pattern_series = [self.reference_patterns[name] for name in pattern_names]
        similarity_scores = np.array([1.0 / (1.0 + p[1]) for p in top_patterns])  # Convert distance to similarity
        
        # Normalize similarity scores to sum to 1
        similarity_scores = similarity_scores / np.sum(similarity_scores)
        
        return pattern_names, pattern_series, similarity_scores
    
    def get_retrieval_tensor(self, query: np.ndarray, hidden_dim: int) -> torch.Tensor:
        """
        Get weighted embedding of similar patterns for model enhancement
        
        Args:
            query: Query time series of shape (n, feature_dim)
            hidden_dim: Dimension of hidden representation
            
        Returns:
            Tensor of shape (batch_size, hidden_dim) with weighted pattern representation
        """
        _, pattern_series, similarity_scores = self.retrieve_similar_patterns(query)
        
        # Simple projection from patterns to hidden dim (can be replaced with learned embedding)
        pattern_embeds = []
        for pattern in pattern_series:
            # Mean pooling as a simple feature
            embed = np.mean(pattern, axis=0)
            # Project to hidden_dim with simple linear mapping
            if embed.shape[0] < hidden_dim:
                # Pad if needed
                embed = np.pad(embed, (0, hidden_dim - embed.shape[0]), 'constant')
            elif embed.shape[0] > hidden_dim:
                # Truncate if needed
                embed = embed[:hidden_dim]
            pattern_embeds.append(embed)
        
        # Stack and convert to tensor
        pattern_embeds = np.stack(pattern_embeds)  # (top_k, hidden_dim)
        
        # Weight by similarity
        weighted_embed = np.sum(pattern_embeds * similarity_scores[:, np.newaxis], axis=0)
        
        return torch.tensor(weighted_embed, dtype=torch.float32).unsqueeze(0)  # Add batch dimension


class TimeSeriesDataset(Dataset):
    """Dataset for financial time series data"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 feature_columns: List[str],
                 target_columns: Optional[List[str]] = None,
                 sequence_length: int = 60,
                 forecast_horizon: int = 5,
                 train: bool = True,
                 scale_data: bool = True):
        """
        Initialize the dataset
        
        Args:
            data: DataFrame containing time series data
            feature_columns: List of column names to use as features
            target_columns: List of column names to use as targets (if None, same as feature_columns)
            sequence_length: Length of input sequences
            forecast_horizon: Number of steps to forecast
            train: Whether this is training data (affects scaling)
            scale_data: Whether to scale the data
        """
        self.data = data.copy()
        self.feature_columns = feature_columns
        self.target_columns = target_columns if target_columns is not None else feature_columns
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.train = train
        self.scale_data = scale_data
        
        # Extract features and targets
        self.features = self.data[feature_columns].values
        
        # Scale features if needed
        if scale_data:
            self.scaler = StandardScaler()
            if train:
                self.features = self.scaler.fit_transform(self.features)
            else:
                self.scaler = None  # Will be set by the caller for validation/test data
                
        # Create sequences
        self.X, self.y = self._create_sequences()
        
    def _create_sequences(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create input/output sequences from the data"""
        X, y = [], []
        
        for i in range(len(self.features) - self.sequence_length - self.forecast_horizon + 1):
            # Input sequence
            X.append(self.features[i:i+self.sequence_length])
            
            # Target sequence (future values)
            target_indices = range(i+self.sequence_length, i+self.sequence_length+self.forecast_horizon)
            target_seq = self.features[target_indices]
            y.append(target_seq)
            
        return np.array(X), np.array(y)
    
    def __len__(self) -> int:
        """Get dataset length"""
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a sample"""
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        
        return x, y
    
    def get_retrieval_vectors(self, retriever: TimeSeriesRetriever, hidden_dim: int) -> List[torch.Tensor]:
        """
        Get retrieval vectors for each sequence in the dataset
        
        Args:
            retriever: TimeSeriesRetriever instance
            hidden_dim: Hidden dimension size
            
        Returns:
            List of retrieval vectors for each sequence
        """
        retrieval_vectors = []
        
        for i in range(len(self.X)):
            sequence = self.X[i]
            retrieval_vector = retriever.get_retrieval_tensor(sequence, hidden_dim)
            retrieval_vectors.append(retrieval_vector)
            
        return retrieval_vectors


def load_financial_data(filepath: str, date_column: str = 'date') -> pd.DataFrame:
    """
    Load financial data from CSV file
    
    Args:
        filepath: Path to CSV file
        date_column: Name of date column
        
    Returns:
        DataFrame with parsed dates
    """
    df = pd.read_csv(filepath)
    
    # Parse dates
    if date_column in df.columns:
        df[date_column] = pd.to_datetime(df[date_column])
        df.set_index(date_column, inplace=True)
        
    return df


def extract_market_patterns(data: pd.DataFrame, 
                           column: str, 
                           window_size: int = 30,
                           step_size: int = 15,
                           min_patterns: int = 50) -> Dict[str, np.ndarray]:
    """
    Extract market patterns from time series data
    
    Args:
        data: DataFrame with time series data
        column: Column name to extract patterns from
        window_size: Size of pattern window
        step_size: Step size between patterns
        min_patterns: Minimum number of patterns to extract
        
    Returns:
        Dictionary of pattern name to pattern data
    """
    series = data[column].values
    patterns = {}
    
    # Extract patterns
    for i in range(0, len(series) - window_size, step_size):
        pattern = series[i:i+window_size]
        pattern_name = f"pattern_{i}"
        patterns[pattern_name] = pattern.reshape(-1, 1)
        
        if len(patterns) >= min_patterns:
            break
            
    return patterns


def plot_prediction_with_uncertainty(x_input: np.ndarray, 
                                   y_true: np.ndarray, 
                                   prediction: Dict[str, np.ndarray],
                                   feature_idx: int = 0,
                                   title: str = "Time Series Prediction") -> plt.Figure:
    """
    Plot time series prediction with uncertainty
    
    Args:
        x_input: Input time series
        y_true: Ground truth future values
        prediction: Dictionary with prediction results
        feature_idx: Index of feature to plot
        title: Plot title
        
    Returns:
        Matplotlib figure
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Time axis
    time_past = np.arange(0, len(x_input))
    time_future = np.arange(len(x_input), len(x_input) + len(y_true))
    time_full = np.arange(0, len(x_input) + len(y_true))
    
    # Plot historical data
    ax.plot(time_past, x_input[:, feature_idx], 'b-', label='Historical Data')
    
    # Plot ground truth
    ax.plot(time_future, y_true[:, feature_idx], 'k-', label='Ground Truth')
    
    # Plot prediction and uncertainty
    ax.plot(time_future, prediction['mean_prediction'][:, feature_idx], 'r-', label='Prediction')
    ax.fill_between(time_future, 
                    prediction['lower_bound'][:, feature_idx], 
                    prediction['upper_bound'][:, feature_idx], 
                    color='r', alpha=0.2, label='90% Confidence Interval')
    
    # Add labels and legend
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    
    return fig 