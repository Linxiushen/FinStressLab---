import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from torch.utils.data import DataLoader
import os
import time
import json
from tqdm import tqdm

from model import HybridModel
from data_utils import (
    TimeSeriesDataset, 
    TimeSeriesRetriever, 
    extract_market_patterns,
    load_financial_data,
    plot_prediction_with_uncertainty
)


class FeedbackMemory:
    """Memory for storing trader feedback for reinforcement learning"""
    
    def __init__(self, capacity: int = 1000):
        """
        Initialize feedback memory
        
        Args:
            capacity: Maximum number of feedback instances to store
        """
        self.capacity = capacity
        self.memory = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray):
        """
        Store feedback in memory
        
        Args:
            state: Input state (market features)
            action: Action taken (model index)
            reward: Reward received
            next_state: Next state
        """
        if len(self.memory) < self.capacity:
            self.memory.append(None)
            
        self.memory[self.position] = (state, action, reward, next_state)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> List[Tuple]:
        """Sample a batch of feedback from memory"""
        return np.random.choice(self.memory, batch_size)
    
    def __len__(self) -> int:
        """Get memory size"""
        return len(self.memory)


class DynamicWeightTrainer:
    """Trainer for HybridModel with dynamic weight adjustment"""
    
    def __init__(self, 
                 model: HybridModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                 learning_rate: float = 1e-4,
                 feedback_memory_capacity: int = 1000):
        """
        Initialize the trainer
        
        Args:
            model: HybridModel instance
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            device: Device to use for training
            learning_rate: Learning rate for optimizer
            feedback_memory_capacity: Capacity of feedback memory
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.learning_rate = learning_rate
        
        # Move model to device
        self.model.to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Initialize loss functions
        self.mse_loss = nn.MSELoss()
        self.quantile_loss = self._quantile_loss
        
        # Initialize feedback memory
        self.feedback_memory = FeedbackMemory(capacity=feedback_memory_capacity)
        
        # Initialize training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'model_weights': []
        }
        
    def _quantile_loss(self, preds: torch.Tensor, targets: torch.Tensor, q: float = 0.5) -> torch.Tensor:
        """
        Quantile loss function for quantile regression
        
        Args:
            preds: Predictions
            targets: Target values
            q: Quantile level
            
        Returns:
            Quantile loss
        """
        errors = targets - preds
        return torch.max((q - 1) * errors, q * errors).mean()
    
    def train_epoch(self, retriever: Optional[TimeSeriesRetriever] = None, hidden_dim: int = 128) -> float:
        """
        Train for one epoch
        
        Args:
            retriever: Optional TimeSeriesRetriever for enhancing predictions
            hidden_dim: Hidden dimension size for retrieval vectors
            
        Returns:
            Average loss for the epoch
        """
        self.model.train()
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (x, y) in enumerate(pbar):
            # Move data to device
            x = x.to(self.device)
            y = y.to(self.device)
            
            # Get retrieval vectors if retriever is provided
            retrieval_vectors = None
            if retriever is not None:
                # For simplicity, we're using the first item in the batch
                # In a real implementation, you'd do this for each item
                sample_x = x[0].cpu().numpy()
                retrieval_vectors = retriever.get_retrieval_tensor(sample_x, hidden_dim).to(self.device)
                retrieval_vectors = retrieval_vectors.expand(x.size(0), -1)  # Expand to batch size
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(x, retrieval_vectors)
            
            # Calculate loss
            prediction = outputs['prediction']
            loss = self.mse_loss(prediction, y)
            
            # Backward pass and optimize
            loss.backward()
            self.optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            
        # Calculate average loss
        avg_loss = epoch_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        
        # Store model weights history
        if hasattr(self.model, 'weight_module'):
            weights = self.model.weight_module.weight_net[-2].weight.detach().cpu().numpy()
            self.history['model_weights'].append(weights)
            
        return avg_loss
    
    def validate(self, retriever: Optional[TimeSeriesRetriever] = None, hidden_dim: int = 128) -> float:
        """
        Validate the model
        
        Args:
            retriever: Optional TimeSeriesRetriever for enhancing predictions
            hidden_dim: Hidden dimension size for retrieval vectors
            
        Returns:
            Validation loss
        """
        if self.val_loader is None:
            return 0.0
            
        self.model.eval()
        val_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for x, y in tqdm(self.val_loader, desc="Validation"):
                # Move data to device
                x = x.to(self.device)
                y = y.to(self.device)
                
                # Get retrieval vectors if retriever is provided
                retrieval_vectors = None
                if retriever is not None:
                    sample_x = x[0].cpu().numpy()
                    retrieval_vectors = retriever.get_retrieval_tensor(sample_x, hidden_dim).to(self.device)
                    retrieval_vectors = retrieval_vectors.expand(x.size(0), -1)
                
                # Forward pass
                outputs = self.model(x, retrieval_vectors)
                
                # Calculate loss
                prediction = outputs['prediction']
                loss = self.mse_loss(prediction, y)
                
                # Update statistics
                val_loss += loss.item()
                num_batches += 1
                
        # Calculate average loss
        avg_loss = val_loss / num_batches
        self.history['val_loss'].append(avg_loss)
        
        return avg_loss
    
    def train(self, 
              num_epochs: int, 
              retriever: Optional[TimeSeriesRetriever] = None,
              hidden_dim: int = 128,
              save_dir: str = 'checkpoints',
              save_freq: int = 5) -> Dict:
        """
        Train the model for multiple epochs
        
        Args:
            num_epochs: Number of epochs to train
            retriever: Optional TimeSeriesRetriever for enhancing predictions
            hidden_dim: Hidden dimension size for retrieval vectors
            save_dir: Directory to save checkpoints
            save_freq: Frequency of saving checkpoints (in epochs)
            
        Returns:
            Training history
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Start training
        start_time = time.time()
        
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train one epoch
            train_loss = self.train_epoch(retriever, hidden_dim)
            
            # Validate
            val_loss = self.validate(retriever, hidden_dim)
            
            # Print epoch results
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % save_freq == 0:
                self.save_checkpoint(os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt"))
                
        # Save final model
        self.save_checkpoint(os.path.join(save_dir, "final_model.pt"))
        
        # Calculate training time
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        return self.history
    
    def update_from_feedback(self, feedback_data: List[Tuple], gamma: float = 0.99, batch_size: int = 32) -> float:
        """
        Update model weights based on trader feedback using reinforcement learning
        
        Args:
            feedback_data: List of (state, action, reward, next_state) tuples
            gamma: Discount factor
            batch_size: Batch size for updates
            
        Returns:
            Loss after update
        """
        # Add feedback to memory
        for feedback in feedback_data:
            self.feedback_memory.push(*feedback)
            
        # Skip if not enough data
        if len(self.feedback_memory) < batch_size:
            return 0.0
            
        # Sample batch from memory
        batch = self.feedback_memory.sample(batch_size)
        states, actions, rewards, next_states = zip(*batch)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.long).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        
        # Get model weights for states
        with torch.no_grad():
            # For simplicity, we're assuming states contain the necessary context
            # In a real implementation, you'd extract the appropriate features
            next_weights = self.model.weight_module.weight_net(next_states)
            max_next_weights = next_weights.max(1)[0]
            
        # Calculate target weights with Q-learning
        target_weights = rewards + gamma * max_next_weights
        
        # Get current weights
        current_weights = self.model.weight_module.weight_net(states)
        current_weights_for_actions = current_weights.gather(1, actions.unsqueeze(1))
        
        # Calculate loss
        loss = self.mse_loss(current_weights_for_actions, target_weights.unsqueeze(1))
        
        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_checkpoint(self, filepath: str) -> None:
        """
        Save model checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str) -> None:
        """
        Load model checkpoint
        
        Args:
            filepath: Path to checkpoint
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")
    
    def predict(self, 
                x: torch.Tensor, 
                retrieval_vectors: Optional[torch.Tensor] = None,
                prev_errors: Optional[torch.Tensor] = None,
                with_uncertainty: bool = True,
                num_samples: int = 100) -> Dict[str, np.ndarray]:
        """
        Make predictions with the model
        
        Args:
            x: Input tensor
            retrieval_vectors: Optional retrieval vectors
            prev_errors: Optional previous errors for ARIMA
            with_uncertainty: Whether to include uncertainty estimates
            num_samples: Number of samples for uncertainty estimation
            
        Returns:
            Dictionary with prediction results
        """
        self.model.eval()
        
        # Move data to device
        x = x.to(self.device)
        if retrieval_vectors is not None:
            retrieval_vectors = retrieval_vectors.to(self.device)
        if prev_errors is not None:
            prev_errors = prev_errors.to(self.device)
        
        # Make prediction
        with torch.no_grad():
            if with_uncertainty:
                outputs = self.model.predict_with_uncertainty(
                    x, retrieval_vectors, prev_errors, num_samples)
            else:
                outputs = self.model(x, retrieval_vectors, prev_errors)
        
        # Convert to numpy
        result = {}
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                result[key] = value.cpu().numpy()
            else:
                result[key] = value
                
        return result


def main():
    """Main training function"""
    # Parameters
    input_dim = 5  # Number of features (e.g. OHLCV for stocks)
    hidden_dim = 128
    sequence_length = 60  # 60 days of historical data
    forecast_horizon = 10  # Predict 10 days ahead
    batch_size = 32
    num_epochs = 50
    learning_rate = 1e-4
    
    # Load data
    # For demo purposes, we'll create synthetic data
    # In a real scenario, you'd load actual market data
    dates = pd.date_range(start='2018-01-01', periods=1000, freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(100, 10, 1000).cumsum(),
        'high': np.random.normal(100, 10, 1000).cumsum() + 5,
        'low': np.random.normal(100, 10, 1000).cumsum() - 5,
        'close': np.random.normal(100, 10, 1000).cumsum() + 2,
        'volume': np.random.normal(1000000, 100000, 1000).astype(int)
    }, index=dates)
    
    # Split data
    train_data = data.iloc[:800]
    val_data = data.iloc[800:]
    
    # Create datasets
    feature_columns = ['open', 'high', 'low', 'close', 'volume']
    train_dataset = TimeSeriesDataset(
        train_data, 
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        train=True,
        scale_data=True
    )
    
    val_dataset = TimeSeriesDataset(
        val_data,
        feature_columns=feature_columns,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        train=False,
        scale_data=True
    )
    val_dataset.scaler = train_dataset.scaler  # Use same scaler as training
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Extract market patterns for retrieval
    patterns = extract_market_patterns(train_data, 'close', window_size=30, min_patterns=50)
    retriever = TimeSeriesRetriever(patterns, top_k=5)
    
    # Create model
    model = HybridModel(input_dim=input_dim, hidden_dim=hidden_dim, forecast_horizon=forecast_horizon)
    
    # Create trainer
    trainer = DynamicWeightTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=learning_rate
    )
    
    # Train model
    history = trainer.train(
        num_epochs=num_epochs,
        retriever=retriever,
        hidden_dim=hidden_dim,
        save_dir='checkpoints'
    )
    
    # Plot training history
    plt.figure(figsize=(12, 6))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig('training_history.png')
    
    # Make predictions on validation set
    val_sample = next(iter(val_loader))
    x_val, y_val = val_sample
    predictions = trainer.predict(
        x_val,
        with_uncertainty=True,
        num_samples=100
    )
    
    # Plot predictions with uncertainty
    sample_idx = 0
    x_sample = x_val[sample_idx].cpu().numpy()
    y_sample = y_val[sample_idx].cpu().numpy()
    
    # Extract predictions for the sample
    pred_sample = {
        'mean_prediction': predictions['mean_prediction'][sample_idx],
        'lower_bound': predictions['lower_bound'][sample_idx],
        'upper_bound': predictions['upper_bound'][sample_idx]
    }
    
    # Plot
    fig = plot_prediction_with_uncertainty(
        x_sample,
        y_sample,
        pred_sample,
        feature_idx=3,  # Close price
        title="Stock Price Prediction with Uncertainty"
    )
    fig.savefig('prediction_example.png')


if __name__ == "__main__":
    main() 