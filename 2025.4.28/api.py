import torch
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
import json
import os
import time
from datetime import datetime, timedelta
import asyncio
from sqlalchemy import create_engine, Column, Integer, Float, String, JSON, DateTime, Boolean, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from model import HybridModel
from data_utils import TimeSeriesRetriever, extract_market_patterns, TimeSeriesDataset
from train import DynamicWeightTrainer


# Initialize FastAPI app
app = FastAPI(
    title="TimeSeriesGPT API",
    description="Financial time series forecasting API combining ARIMA/LSTM with LLM interpretations",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup TimescaleDB connection
# For local development, you can use SQLite instead
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./timeseries_gpt.db")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# Define database models
class MarketData(Base):
    """Market data table for storing time series data"""
    __tablename__ = "market_data"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    volume = Column(Integer)
    
    class Config:
        orm_mode = True


class Prediction(Base):
    """Prediction table for storing model predictions"""
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    timestamp = Column(DateTime, index=True)
    prediction_timestamp = Column(DateTime, index=True)
    forecast_horizon = Column(Integer)
    prediction_data = Column(JSON)  # Store full prediction including uncertainty
    model_weights = Column(JSON)  # Store model weights for explainability
    
    class Config:
        orm_mode = True


class Feedback(Base):
    """Feedback table for storing trader feedback"""
    __tablename__ = "feedback"
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"))
    timestamp = Column(DateTime, index=True)
    action_taken = Column(String)  # e.g., "buy", "sell", "hold"
    actual_return = Column(Float)
    expected_return = Column(Float)
    notes = Column(String, nullable=True)
    
    class Config:
        orm_mode = True


# Create tables
Base.metadata.create_all(bind=engine)


# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Load model and data (singleton pattern)
class ModelManager:
    """Singleton class to manage model and data"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
            cls._instance.model = None
            cls._instance.trainer = None
            cls._instance.retriever = None
            cls._instance.hidden_dim = 128
            cls._instance.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            cls._instance.symbols = {}  # Dictionary to store symbols and their data
            cls._instance.is_loaded = False
        return cls._instance
    
    def load_model(self, model_path: str, input_dim: int = 5, hidden_dim: int = 128, forecast_horizon: int = 10):
        """Load model from checkpoint"""
        # Create model
        self.model = HybridModel(input_dim=input_dim, hidden_dim=hidden_dim, forecast_horizon=forecast_horizon)
        self.model.to(self.device)
        
        # Create trainer
        self.trainer = DynamicWeightTrainer(
            model=self.model,
            train_loader=None,  # Not needed for inference
            device=str(self.device)
        )
        
        # Load model checkpoint
        self.trainer.load_checkpoint(model_path)
        self.hidden_dim = hidden_dim
        self.is_loaded = True
        
        print(f"Model loaded from {model_path}")
        
    def add_symbol_data(self, symbol: str, data: pd.DataFrame, extract_patterns: bool = True):
        """Add data for a new symbol"""
        self.symbols[symbol] = {
            "data": data,
            "patterns": None,
            "retriever": None
        }
        
        # Extract patterns for retrieval if specified
        if extract_patterns:
            patterns = extract_market_patterns(data, 'close', window_size=30, min_patterns=50)
            retriever = TimeSeriesRetriever(patterns, top_k=5)
            self.symbols[symbol]["patterns"] = patterns
            self.symbols[symbol]["retriever"] = retriever
            
        print(f"Added data for symbol {symbol}")
        
    def get_retriever(self, symbol: str) -> Optional[TimeSeriesRetriever]:
        """Get retriever for a symbol"""
        if symbol in self.symbols and self.symbols[symbol]["retriever"] is not None:
            return self.symbols[symbol]["retriever"]
        return None
    
    def update_from_feedback(self, feedback_data: List):
        """Update model from trader feedback"""
        return self.trainer.update_from_feedback(feedback_data)


# Initialize model manager
model_manager = ModelManager()


# Pydantic models for API
class MarketDataInput(BaseModel):
    """Input model for market data"""
    symbol: str
    data: List[Dict[str, Any]]
    date_format: str = "%Y-%m-%d"


class PredictionRequest(BaseModel):
    """Input model for prediction request"""
    symbol: str
    sequence_length: int = 60
    forecast_horizon: int = 10
    include_uncertainty: bool = True
    uncertainty_samples: int = 100


class FeedbackInput(BaseModel):
    """Input model for trader feedback"""
    prediction_id: int
    action_taken: str
    actual_return: float
    expected_return: float
    notes: Optional[str] = None


# API endpoints
@app.post("/load-model")
async def load_model(
    model_path: str,
    input_dim: int = 5,
    hidden_dim: int = 128,
    forecast_horizon: int = 10
):
    """Load model from checkpoint"""
    try:
        model_manager.load_model(model_path, input_dim, hidden_dim, forecast_horizon)
        return {"status": "success", "message": "Model loaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")


@app.post("/add-market-data")
async def add_market_data(
    data_input: MarketDataInput,
    db: Session = Depends(get_db)
):
    """Add market data for a symbol"""
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(data_input.data)
        
        # Parse dates
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], format=data_input.date_format)
            df.set_index('date', inplace=True)
        
        # Add data to model manager
        model_manager.add_symbol_data(data_input.symbol, df)
        
        # Store data in database
        for _, row in df.reset_index().iterrows():
            db_record = MarketData(
                symbol=data_input.symbol,
                timestamp=row['date'],
                open=row.get('open', 0.0),
                high=row.get('high', 0.0),
                low=row.get('low', 0.0),
                close=row.get('close', 0.0),
                volume=row.get('volume', 0)
            )
            db.add(db_record)
        
        db.commit()
        
        return {
            "status": "success", 
            "message": f"Added {len(df)} records for {data_input.symbol}",
            "data_shape": df.shape
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error adding market data: {str(e)}")


@app.post("/predict")
async def predict(
    request: PredictionRequest,
    db: Session = Depends(get_db)
):
    """Generate prediction for a symbol"""
    try:
        # Check if model is loaded
        if not model_manager.is_loaded:
            raise HTTPException(status_code=400, detail="Model not loaded. Call /load-model first.")
        
        # Check if symbol data exists
        if request.symbol not in model_manager.symbols:
            raise HTTPException(status_code=404, detail=f"No data found for symbol {request.symbol}")
        
        # Get data for symbol
        symbol_data = model_manager.symbols[request.symbol]["data"]
        
        # Prepare input data
        feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Create dataset
        dataset = TimeSeriesDataset(
            symbol_data,
            feature_columns=feature_columns,
            sequence_length=request.sequence_length,
            forecast_horizon=request.forecast_horizon,
            train=False,
            scale_data=True
        )
        
        # Get last sequence
        if len(dataset) == 0:
            raise HTTPException(status_code=400, detail="Not enough data to create prediction")
        
        x, _ = dataset[-1]
        x = x.unsqueeze(0)  # Add batch dimension
        
        # Get retriever
        retriever = model_manager.get_retriever(request.symbol)
        retrieval_vectors = None
        if retriever is not None:
            sample_x = x[0].numpy()
            retrieval_vectors = retriever.get_retrieval_tensor(sample_x, model_manager.hidden_dim)
            retrieval_vectors = retrieval_vectors.unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        predictions = model_manager.trainer.predict(
            x,
            retrieval_vectors=retrieval_vectors,
            with_uncertainty=request.include_uncertainty,
            num_samples=request.uncertainty_samples
        )
        
        # Get model weights
        weight_module = model_manager.model.weight_module
        context = torch.cat([
            x[0, -1, :],  # Last timestep features
            torch.zeros(model_manager.hidden_dim * 3)  # Placeholder for model features
        ]).unsqueeze(0)
        
        weights = weight_module.weight_net(context).detach().cpu().numpy()[0]
        
        # Prepare prediction timestamp
        last_timestamp = symbol_data.index[-1]
        prediction_timestamps = [
            (last_timestamp + timedelta(days=i+1)).strftime("%Y-%m-%d")
            for i in range(request.forecast_horizon)
        ]
        
        # Convert prediction to original scale (inverse transform)
        # In a real implementation, you'd apply the inverse transform here
        
        # Format output
        formatted_output = {}
        
        # Basic prediction
        formatted_output["prediction"] = {
            "timestamps": prediction_timestamps,
            "values": {
                "open": predictions["mean_prediction"][0, :, 0].tolist(),
                "high": predictions["mean_prediction"][0, :, 1].tolist(),
                "low": predictions["mean_prediction"][0, :, 2].tolist(),
                "close": predictions["mean_prediction"][0, :, 3].tolist(),
                "volume": predictions["mean_prediction"][0, :, 4].tolist()
            }
        }
        
        # Uncertainty bounds if requested
        if request.include_uncertainty:
            formatted_output["uncertainty"] = {
                "lower_bound": {
                    "open": predictions["lower_bound"][0, :, 0].tolist(),
                    "high": predictions["lower_bound"][0, :, 1].tolist(),
                    "low": predictions["lower_bound"][0, :, 2].tolist(),
                    "close": predictions["lower_bound"][0, :, 3].tolist(),
                    "volume": predictions["lower_bound"][0, :, 4].tolist()
                },
                "upper_bound": {
                    "open": predictions["upper_bound"][0, :, 0].tolist(),
                    "high": predictions["upper_bound"][0, :, 1].tolist(),
                    "low": predictions["upper_bound"][0, :, 2].tolist(),
                    "close": predictions["upper_bound"][0, :, 3].tolist(),
                    "volume": predictions["upper_bound"][0, :, 4].tolist()
                }
            }
        
        # Model weights
        formatted_output["model_weights"] = {
            "timegpt": float(weights[0]),
            "lstm_transformer": float(weights[1]),
            "arima": float(weights[2])
        }
        
        # Store prediction in database
        db_prediction = Prediction(
            symbol=request.symbol,
            timestamp=datetime.now(),
            prediction_timestamp=last_timestamp,
            forecast_horizon=request.forecast_horizon,
            prediction_data=formatted_output,
            model_weights=formatted_output["model_weights"]
        )
        db.add(db_prediction)
        db.commit()
        
        # Add prediction ID to output
        formatted_output["prediction_id"] = db_prediction.id
        
        return formatted_output
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error generating prediction: {str(e)}")


@app.post("/feedback")
async def add_feedback(
    feedback: FeedbackInput,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """Add trader feedback for model improvement"""
    try:
        # Get prediction from database
        prediction = db.query(Prediction).filter(Prediction.id == feedback.prediction_id).first()
        if prediction is None:
            raise HTTPException(status_code=404, detail=f"Prediction with ID {feedback.prediction_id} not found")
        
        # Store feedback in database
        db_feedback = Feedback(
            prediction_id=feedback.prediction_id,
            timestamp=datetime.now(),
            action_taken=feedback.action_taken,
            actual_return=feedback.actual_return,
            expected_return=feedback.expected_return,
            notes=feedback.notes
        )
        db.add(db_feedback)
        db.commit()
        
        # Calculate reward (difference between actual and expected return)
        reward = feedback.actual_return - feedback.expected_return
        
        # Get model weights
        weights = prediction.model_weights
        
        # Determine which model had the highest weight
        weight_values = [weights["timegpt"], weights["lstm_transformer"], weights["arima"]]
        action = weight_values.index(max(weight_values))
        
        # Create state from prediction data
        # For simplicity, we'll use the last day of the prediction
        state = np.array([
            prediction.prediction_data["prediction"]["values"]["open"][-1],
            prediction.prediction_data["prediction"]["values"]["high"][-1],
            prediction.prediction_data["prediction"]["values"]["low"][-1],
            prediction.prediction_data["prediction"]["values"]["close"][-1],
            prediction.prediction_data["prediction"]["values"]["volume"][-1]
        ])
        
        # For next state, we'd typically use the actual values observed
        # For now, we'll just use the same state as a placeholder
        next_state = state.copy()
        
        # Prepare feedback data for model update
        feedback_data = [(state, action, reward, next_state)]
        
        # Update model in background task to avoid blocking
        background_tasks.add_task(model_manager.update_from_feedback, feedback_data)
        
        return {
            "status": "success", 
            "message": "Feedback recorded and model update scheduled",
            "feedback_id": db_feedback.id
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")


@app.get("/symbols")
async def get_symbols():
    """Get list of available symbols"""
    symbols = list(model_manager.symbols.keys())
    return {"symbols": symbols}


@app.get("/model-status")
async def get_model_status():
    """Check if model is loaded"""
    return {
        "model_loaded": model_manager.is_loaded,
        "device": str(model_manager.device),
        "symbols_count": len(model_manager.symbols)
    }


@app.get("/predictions/{symbol}")
async def get_predictions(
    symbol: str,
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get recent predictions for a symbol"""
    predictions = db.query(Prediction).filter(
        Prediction.symbol == symbol
    ).order_by(Prediction.timestamp.desc()).limit(limit).all()
    
    return [
        {
            "id": p.id,
            "timestamp": p.timestamp,
            "prediction_timestamp": p.prediction_timestamp,
            "forecast_horizon": p.forecast_horizon,
            "prediction_data": p.prediction_data,
            "model_weights": p.model_weights
        }
        for p in predictions
    ]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True) 