import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Dict, Optional, Union


class TimeGPTLayer(nn.Module):
    """TimeGPT base layer with sliding window mechanism for zero-shot time series forecasting"""
    
    def __init__(self, input_dim: int, hidden_dim: int, window_size: int = 64):
        super().__init__()
        self.window_size = window_size
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.position_encoding = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Apply sliding window if sequence is longer than window size
        if seq_len > self.window_size:
            # Use the most recent window_size time steps
            x = x[:, -self.window_size:, :]
            seq_len = self.window_size
            
        # Project input to hidden dimension
        x = self.input_projection(x)
        
        # Add positional encoding
        pos_enc = self.position_encoding[:, :seq_len, :]
        x = x + pos_enc
        
        # Self-attention with residual connection and normalization
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x)
        x = x + residual
        
        # Feed-forward network with residual connection
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = x + residual
        
        return x


class LSTMTransformerLayer(nn.Module):
    """LSTM-Transformer hybrid architecture with improved attention mechanism"""
    
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
            dropout=0.1 if num_layers > 1 else 0
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # DTW retrieval enhancement
        self.dtw_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor, 
                retrieval_vectors: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x shape: [batch_size, seq_len, input_dim]
        
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Self-attention with residual connection
        residual = lstm_out
        attn_out = self.norm1(lstm_out)
        attn_out, _ = self.attention(attn_out, attn_out, attn_out)
        lstm_attn_out = residual + attn_out
        
        # DTW retrieval enhancement if provided
        if retrieval_vectors is not None:
            # Project retrieval vectors
            retrieval_proj = self.dtw_projection(retrieval_vectors)
            # Add to the output
            lstm_attn_out = lstm_attn_out + retrieval_proj
            
        # Feed-forward network with residual connection
        residual = lstm_attn_out
        ffn_out = self.norm2(lstm_attn_out)
        ffn_out = self.ffn(ffn_out)
        out = residual + ffn_out
        
        return out


class ARIMALayer(nn.Module):
    """Neural ARIMA implementation with conformal prediction methods"""
    
    def __init__(self, input_dim: int, hidden_dim: int, p: int = 5, d: int = 1, q: int = 1):
        super().__init__()
        self.p = p  # Autoregressive order
        self.d = d  # Integrated/differencing order
        self.q = q  # Moving average order
        
        # AR parameters (autoregressive)
        self.ar_weights = nn.Parameter(torch.rand(p, input_dim, hidden_dim) * 0.1)
        
        # MA parameters (moving average)
        self.ma_weights = nn.Parameter(torch.rand(q, input_dim, hidden_dim) * 0.1)
        
        # Integration layer
        self.integration = nn.Linear(hidden_dim, hidden_dim)
        
        # Conformal prediction layer
        self.conformal_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2)  # Output mean and variance for each feature
        )
        
    def forward(self, x: torch.Tensor, 
                prev_errors: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, input_dim = x.shape
        
        # Apply differencing if d > 0
        if self.d > 0:
            # Simple differencing - can be extended to higher orders
            diff_x = x[:, 1:, :] - x[:, :-1, :]
            for _ in range(1, self.d):
                diff_x = diff_x[:, 1:, :] - diff_x[:, :-1, :]
        else:
            diff_x = x
            
        # Get the most recent p time steps for AR component
        if seq_len >= self.p:
            ar_input = diff_x[:, -self.p:, :]
        else:
            # Pad with zeros if not enough time steps
            padding = torch.zeros(batch_size, self.p - seq_len, input_dim, device=x.device)
            ar_input = torch.cat([padding, diff_x], dim=1)
            
        # Apply AR weights
        ar_output = torch.zeros(batch_size, input_dim, device=x.device)
        for i in range(self.p):
            ar_term = torch.matmul(ar_input[:, i, :].unsqueeze(1), self.ar_weights[i])
            ar_output = ar_output + ar_term.squeeze(1)
            
        # Apply MA weights if previous errors are provided
        ma_output = torch.zeros(batch_size, input_dim, device=x.device)
        if prev_errors is not None and self.q > 0:
            if prev_errors.size(1) >= self.q:
                ma_input = prev_errors[:, -self.q:, :]
            else:
                # Pad with zeros if not enough previous errors
                padding = torch.zeros(batch_size, self.q - prev_errors.size(1), input_dim, device=x.device)
                ma_input = torch.cat([padding, prev_errors], dim=1)
                
            for i in range(self.q):
                ma_term = torch.matmul(ma_input[:, i, :].unsqueeze(1), self.ma_weights[i])
                ma_output = ma_output + ma_term.squeeze(1)
        
        # Combine AR and MA components
        arima_output = ar_output + ma_output
        
        # Apply integration layer
        integrated_output = self.integration(arima_output.unsqueeze(1)).squeeze(1)
        
        # Apply conformal prediction to get distribution parameters
        conformal_params = self.conformal_net(integrated_output)
        mean, log_var = torch.split(conformal_params, integrated_output.size(-1), dim=-1)
        
        # Ensure positive variance
        variance = F.softplus(log_var)
        
        return mean, variance


class DynamicWeightModule(nn.Module):
    """Dynamic weight allocation using reinforcement learning strategy"""
    
    def __init__(self, input_dim: int, num_models: int = 3):
        super().__init__()
        self.num_models = num_models
        
        # Weight generation network
        self.weight_net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_models),
            nn.Softmax(dim=-1)
        )
        
        # Gradient isolation gate for GRs update strategy
        self.gradient_gates = nn.Parameter(torch.ones(num_models))
        
    def forward(self, x: torch.Tensor, model_outputs: List[torch.Tensor]) -> torch.Tensor:
        # x: market context features [batch_size, input_dim]
        # model_outputs: list of tensor outputs from each model [batch_size, output_dim]
        
        # Generate dynamic weights based on market context
        weights = self.weight_net(x)  # [batch_size, num_models]
        
        # Apply gradient isolation gates during training
        if self.training:
            weights = weights * self.gradient_gates
            
        # Normalize weights to sum to 1
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Combine model outputs using dynamic weights
        combined_output = torch.zeros_like(model_outputs[0])
        for i, model_output in enumerate(model_outputs):
            combined_output += model_output * weights[:, i].unsqueeze(-1)
            
        return combined_output


class HybridModel(nn.Module):
    """
    Hybrid model that combines TimeGPT, LSTM-Transformer and ARIMA
    with dynamic weight adjustment module
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, forecast_horizon: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.forecast_horizon = forecast_horizon
        
        # TimeGPT base layer
        self.timegpt_layer = TimeGPTLayer(input_dim, hidden_dim)
        
        # LSTM-Transformer layer
        self.lstm_transformer = LSTMTransformerLayer(input_dim, hidden_dim)
        
        # ARIMA layer
        self.arima_layer = ARIMALayer(input_dim, hidden_dim)
        
        # Prediction heads for each model
        self.timegpt_head = nn.Linear(hidden_dim, forecast_horizon * input_dim)
        self.lstm_head = nn.Linear(hidden_dim, forecast_horizon * input_dim)
        self.arima_head = nn.Linear(hidden_dim, forecast_horizon * input_dim)
        
        # Dynamic weight allocation module
        self.weight_module = DynamicWeightModule(input_dim + hidden_dim * 3, num_models=3)
        
        # Final projection layer
        self.final_projection = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor, 
                retrieval_vectors: Optional[torch.Tensor] = None,
                prev_errors: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # x shape: [batch_size, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Get last time step features for context
        context_features = x[:, -1, :]
        
        # TimeGPT prediction
        timegpt_features = self.timegpt_layer(x)  # [batch_size, seq_len, hidden_dim]
        timegpt_last = timegpt_features[:, -1, :]  # Take last time step
        timegpt_pred = self.timegpt_head(timegpt_last)
        timegpt_pred = timegpt_pred.reshape(batch_size, self.forecast_horizon, self.input_dim)
        
        # LSTM-Transformer prediction
        lstm_features = self.lstm_transformer(x, retrieval_vectors)  # [batch_size, seq_len, hidden_dim]
        lstm_last = lstm_features[:, -1, :]  # Take last time step
        lstm_pred = self.lstm_head(lstm_last)
        lstm_pred = lstm_pred.reshape(batch_size, self.forecast_horizon, self.input_dim)
        
        # ARIMA prediction
        arima_mean, arima_var = self.arima_layer(x, prev_errors)  # [batch_size, input_dim]
        arima_pred = self.arima_head(arima_mean.unsqueeze(1))
        arima_pred = arima_pred.reshape(batch_size, self.forecast_horizon, self.input_dim)
        
        # Concatenate context with model features for weight allocation
        combined_context = torch.cat([
            context_features,
            timegpt_last,
            lstm_last,
            arima_mean
        ], dim=-1)
        
        # Get dynamic weights and combine predictions
        model_outputs = [timegpt_pred, lstm_pred, arima_pred]
        weighted_pred = self.weight_module(combined_context, model_outputs)
        
        # Final projection
        final_pred = self.final_projection(weighted_pred)
        
        return {
            "prediction": final_pred,
            "timegpt_pred": timegpt_pred,
            "lstm_pred": lstm_pred,
            "arima_pred": arima_pred,
            "arima_variance": arima_var
        }
    
    def predict_with_uncertainty(self, x: torch.Tensor, 
                               retrieval_vectors: Optional[torch.Tensor] = None,
                               prev_errors: Optional[torch.Tensor] = None,
                               num_samples: int = 100) -> Dict[str, torch.Tensor]:
        """Generate predictions with uncertainty estimates using Monte Carlo sampling"""
        
        outputs = self.forward(x, retrieval_vectors, prev_errors)
        
        # Get base prediction and ARIMA variance
        base_pred = outputs["prediction"]
        arima_var = outputs["arima_variance"].unsqueeze(1).expand(-1, self.forecast_horizon, -1)
        
        # Generate samples from distribution
        samples = []
        for _ in range(num_samples):
            # Sample from normal distribution
            noise = torch.randn_like(base_pred) * torch.sqrt(arima_var)
            sample = base_pred + noise
            samples.append(sample)
            
        # Stack samples
        all_samples = torch.stack(samples, dim=1)  # [batch_size, num_samples, forecast_horizon, input_dim]
        
        # Calculate mean and quantiles
        mean_pred = all_samples.mean(dim=1)
        lower_quantile = torch.quantile(all_samples, 0.1, dim=1)
        upper_quantile = torch.quantile(all_samples, 0.9, dim=1)
        
        return {
            "mean_prediction": mean_pred,
            "lower_bound": lower_quantile,
            "upper_bound": upper_quantile,
            "samples": all_samples
        } 