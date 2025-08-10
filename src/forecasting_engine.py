import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

class ForecastingEngine:
    def __init__(self, models_dir: str = "models", demo_mode: bool = False):
        self.models_dir = models_dir
        self.volume_model = None
        self.demo_mode = demo_mode
        self.load_models()
    
    def load_models(self):
        """Load the pre-trained forecasting models"""
        try:
            # Load the stock volume model
            model_path = os.path.join(self.models_dir, "stock_volume_model.pkl")
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    self.volume_model = pickle.load(f)
                print("✅ Stock volume model loaded successfully")
            else:
                print("⚠️ Stock volume model not found, will train new model if needed")
        except Exception as e:
            print(f"❌ Error loading models: {e}")
    
    def predict_stock_volume(self, ticker: str, days_ahead: int = 5) -> Dict[str, Any]:
        """
        Predict stock volume for the next N days using the pre-trained model
        """
        try:
            if self.demo_mode:
                return self._get_demo_volume_prediction(ticker, days_ahead)
            
            # Download recent data for the ticker
            data = yf.download(ticker, period="60d", interval="1d", auto_adjust=True)
            data.reset_index(inplace=True)
            
            if data.empty:
                return self._get_demo_volume_prediction(ticker, days_ahead)
            
            # Prepare features for prediction
            features_df = self._prepare_features(data, ticker)
            
            if features_df.empty:
                return self._get_demo_volume_prediction(ticker, days_ahead)
            
            # Get the latest data point for prediction
            latest_features = features_df.iloc[-1:].copy()
            
            # Make prediction
            if self.volume_model is not None:
                predicted_volume = self.volume_model.predict(latest_features)[0]
                
                # Generate predictions for multiple days
                predictions = []
                current_data = latest_features.copy()
                
                for day in range(1, days_ahead + 1):
                    # Update features for next day prediction
                    next_day_features = self._update_features_for_next_day(current_data, data)
                    
                    if next_day_features is not None:
                        pred_volume = self.volume_model.predict(next_day_features)[0]
                        predictions.append({
                            "day": day,
                            "predicted_volume": int(pred_volume),
                            "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
                        })
                        current_data = next_day_features
                
                return {
                    "ticker": ticker,
                    "current_volume": int(data['Volume'].iloc[-1]),
                    "predictions": predictions,
                    "model_confidence": self._calculate_model_confidence(data),
                    "last_updated": datetime.now().isoformat()
                }
            else:
                return self._get_demo_volume_prediction(ticker, days_ahead)
                
        except Exception as e:
            return self._get_demo_volume_prediction(ticker, days_ahead)
    
    def _get_demo_volume_prediction(self, ticker: str, days_ahead: int) -> Dict[str, Any]:
        """Generate demo volume predictions for testing"""
        base_volume = 50000000  # 50M base volume
        
        predictions = []
        for day in range(1, days_ahead + 1):
            # Add some randomness to predictions
            volume_change = np.random.normal(0, 0.1)  # ±10% change
            predicted_volume = int(base_volume * (1 + volume_change))
            predictions.append({
                "day": day,
                "predicted_volume": predicted_volume,
                "date": (datetime.now() + timedelta(days=day)).strftime("%Y-%m-%d")
            })
        
        return {
            "ticker": ticker,
            "current_volume": base_volume,
            "predictions": predictions,
            "model_confidence": 0.75,
            "last_updated": datetime.now().isoformat(),
            "demo_mode": True
        }
    
    def predict_stock_price_trend(self, ticker: str, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Predict stock price trend using technical analysis and volume predictions
        """
        try:
            if self.demo_mode:
                return self._get_demo_price_trend(ticker, days_ahead)
            
            # Get volume predictions
            volume_pred = self.predict_stock_volume(ticker, days_ahead)
            
            if "error" in volume_pred:
                return self._get_demo_price_trend(ticker, days_ahead)
            
            # Download price data
            data = yf.download(ticker, period="60d", interval="1d", auto_adjust=True)
            
            if data.empty:
                return self._get_demo_price_trend(ticker, days_ahead)
            
            # Calculate technical indicators
            data['SMA_20'] = data['Close'].rolling(window=20).mean()
            data['SMA_50'] = data['Close'].rolling(window=50).mean()
            data['RSI'] = self._calculate_rsi(data['Close'])
            
            # Get current indicators
            current_price = data['Close'].iloc[-1]
            current_sma_20 = data['SMA_20'].iloc[-1]
            current_sma_50 = data['SMA_50'].iloc[-1]
            current_rsi = data['RSI'].iloc[-1]
            
            # Simple trend prediction based on technical indicators
            trend_signals = []
            
            if current_price > current_sma_20:
                trend_signals.append("Bullish short-term")
            else:
                trend_signals.append("Bearish short-term")
            
            if current_sma_20 > current_sma_50:
                trend_signals.append("Bullish medium-term")
            else:
                trend_signals.append("Bearish medium-term")
            
            if current_rsi < 30:
                trend_signals.append("Oversold")
            elif current_rsi > 70:
                trend_signals.append("Overbought")
            else:
                trend_signals.append("Neutral RSI")
            
            # Volume trend analysis
            avg_volume = data['Volume'].mean()
            current_volume = data['Volume'].iloc[-1]
            
            if current_volume > avg_volume * 1.5:
                trend_signals.append("High volume")
            elif current_volume < avg_volume * 0.5:
                trend_signals.append("Low volume")
            else:
                trend_signals.append("Normal volume")
            
            # Overall trend assessment
            bullish_signals = sum(1 for signal in trend_signals if "Bullish" in signal or "Oversold" in signal)
            bearish_signals = sum(1 for signal in trend_signals if "Bearish" in signal or "Overbought" in signal)
            
            if bullish_signals > bearish_signals:
                overall_trend = "Bullish"
                confidence = bullish_signals / len(trend_signals)
            elif bearish_signals > bullish_signals:
                overall_trend = "Bearish"
                confidence = bearish_signals / len(trend_signals)
            else:
                overall_trend = "Neutral"
                confidence = 0.5
            
            return {
                "ticker": ticker,
                "current_price": current_price,
                "overall_trend": overall_trend,
                "trend_confidence": confidence,
                "trend_signals": trend_signals,
                "technical_indicators": {
                    "sma_20": current_sma_20,
                    "sma_50": current_sma_50,
                    "rsi": current_rsi,
                    "volume_ratio": current_volume / avg_volume
                },
                "volume_predictions": volume_pred,
                "prediction_horizon": f"{days_ahead} days",
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._get_demo_price_trend(ticker, days_ahead)
    
    def _get_demo_price_trend(self, ticker: str, days_ahead: int) -> Dict[str, Any]:
        """Generate demo price trend predictions for testing"""
        base_price = 150.0  # Base price
        
        # Random trend determination
        trend_options = ["Bullish", "Bearish", "Neutral"]
        overall_trend = np.random.choice(trend_options, p=[0.4, 0.3, 0.3])
        
        # Generate trend signals
        trend_signals = []
        if overall_trend == "Bullish":
            trend_signals = ["Bullish short-term", "Bullish medium-term", "Neutral RSI", "High volume"]
            confidence = 0.7
        elif overall_trend == "Bearish":
            trend_signals = ["Bearish short-term", "Bearish medium-term", "Overbought", "Low volume"]
            confidence = 0.6
        else:
            trend_signals = ["Neutral short-term", "Neutral medium-term", "Neutral RSI", "Normal volume"]
            confidence = 0.5
        
        return {
            "ticker": ticker,
            "current_price": base_price,
            "overall_trend": overall_trend,
            "trend_confidence": confidence,
            "trend_signals": trend_signals,
            "technical_indicators": {
                "sma_20": base_price * 0.98,
                "sma_50": base_price * 0.95,
                "rsi": np.random.uniform(30, 70),
                "volume_ratio": np.random.uniform(0.8, 1.2)
            },
            "volume_predictions": self._get_demo_volume_prediction(ticker, days_ahead),
            "prediction_horizon": f"{days_ahead} days",
            "last_updated": datetime.now().isoformat(),
            "demo_mode": True
        }
    
    def get_forecasting_summary(self, ticker: str) -> Dict[str, Any]:
        """Get comprehensive forecasting summary for a ticker"""
        try:
            # Get both volume and price predictions
            volume_pred = self.predict_stock_volume(ticker, 5)
            price_trend = self.predict_stock_price_trend(ticker, 30)
            
            if "error" in volume_pred or "error" in price_trend:
                return self._get_demo_forecasting_summary(ticker)
            
            return {
                "ticker": ticker,
                "volume_forecast": volume_pred,
                "price_trend_forecast": price_trend,
                "summary": {
                    "trend": price_trend["overall_trend"],
                    "confidence": price_trend["trend_confidence"],
                    "volume_trend": "Increasing" if volume_pred["predictions"][0]["predicted_volume"] > volume_pred["current_volume"] else "Decreasing",
                    "key_signals": price_trend["trend_signals"][:3]  # Top 3 signals
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return self._get_demo_forecasting_summary(ticker)
    
    def _get_demo_forecasting_summary(self, ticker: str) -> Dict[str, Any]:
        """Generate demo forecasting summary for testing"""
        volume_pred = self._get_demo_volume_prediction(ticker, 5)
        price_trend = self._get_demo_price_trend(ticker, 30)
        
        return {
            "ticker": ticker,
            "volume_forecast": volume_pred,
            "price_trend_forecast": price_trend,
            "summary": {
                "trend": price_trend["overall_trend"],
                "confidence": price_trend["trend_confidence"],
                "volume_trend": "Increasing" if volume_pred["predictions"][0]["predicted_volume"] > volume_pred["current_volume"] else "Decreasing",
                "key_signals": price_trend["trend_signals"][:3]
            },
            "last_updated": datetime.now().isoformat(),
            "demo_mode": True
        }
    
    def _prepare_features(self, data: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Prepare features for the model"""
        try:
            # Basic features
            features_df = data.copy()
            features_df['Day'] = features_df['Date'].dt.day
            features_df['Year'] = features_df['Date'].dt.year
            features_df['IsWeekend'] = features_df['Date'].dt.dayofweek >= 5
            
            # Volume lags
            features_df['Volume_lag1'] = features_df['Volume'].shift(1)
            features_df['Volume_lag2'] = features_df['Volume'].shift(2)
            features_df['Volume_lag3'] = features_df['Volume'].shift(3)
            features_df['Volume_lag5'] = features_df['Volume'].shift(5)
            
            # Rolling stats
            features_df['Volume_roll_mean_5'] = features_df['Volume'].rolling(window=5).mean()
            features_df['Volume_roll_std_5'] = features_df['Volume'].rolling(window=5).std()
            
            # Price range & change %
            features_df['Price_Range'] = features_df['High'] - features_df['Low']
            features_df['Price_Change_Pct'] = (features_df['Close'] - features_df['Open']) / features_df['Open'] * 100
            
            # Cyclical month encoding
            features_df['Month_sin'] = np.sin(2 * np.pi * features_df['Date'].dt.month / 12)
            features_df['Month_cos'] = np.cos(2 * np.pi * features_df['Date'].dt.month / 12)
            
            # Cyclical day-of-week encoding
            features_df['DayOfWeek_sin'] = np.sin(2 * np.pi * features_df['Date'].dt.dayofweek / 7)
            features_df['DayOfWeek_cos'] = np.cos(2 * np.pi * features_df['Date'].dt.dayofweek / 7)
            
            # Dummy ticker columns (set based on actual ticker)
            features_df['Ticker_AMZN'] = 1 if ticker == 'AMZN' else 0
            features_df['Ticker_MSFT'] = 1 if ticker == 'MSFT' else 0
            features_df['Ticker_SPY'] = 1 if ticker == 'SPY' else 0
            
            # Define feature columns
            feature_cols = ['Open', 'High', 'Low', 'Close', 'Day', 'Year', 'IsWeekend',
                          'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag5',
                          'Volume_roll_mean_5', 'Volume_roll_std_5', 'Price_Range',
                          'Price_Change_Pct', 'Month_sin', 'Month_cos', 'DayOfWeek_sin',
                          'DayOfWeek_cos', 'Ticker_AMZN', 'Ticker_MSFT', 'Ticker_SPY']
            
            # Select features and drop NaN
            features_df = features_df[feature_cols].dropna()
            
            return features_df
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def _update_features_for_next_day(self, current_features: pd.DataFrame, 
                                    historical_data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Update features for next day prediction"""
        try:
            # This is a simplified approach - in practice, you'd need more sophisticated
            # feature updating logic based on the model's requirements
            
            next_day = current_features.copy()
            
            # Update date-related features
            next_date = datetime.now() + timedelta(days=1)
            next_day['Day'] = next_date.day
            next_day['Year'] = next_date.year
            next_day['IsWeekend'] = next_date.weekday() >= 5
            
            # Update cyclical features
            next_day['Month_sin'] = np.sin(2 * np.pi * next_date.month / 12)
            next_day['Month_cos'] = np.cos(2 * np.pi * next_date.month / 12)
            next_day['DayOfWeek_sin'] = np.sin(2 * np.pi * next_date.weekday() / 7)
            next_day['DayOfWeek_cos'] = np.cos(2 * np.pi * next_date.weekday() / 7)
            
            # For simplicity, keep other features the same
            # In a real implementation, you'd update these based on predicted values
            
            return next_day
            
        except Exception as e:
            print(f"Error updating features: {e}")
            return None
    
    def _calculate_model_confidence(self, data: pd.DataFrame) -> float:
        """Calculate model confidence based on data quality and recent performance"""
        try:
            # Simple confidence calculation based on data availability
            data_quality = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Volume volatility (lower volatility = higher confidence)
            volume_volatility = data['Volume'].std() / data['Volume'].mean()
            volatility_confidence = max(0, 1 - volume_volatility)
            
            # Recent data availability
            recent_data_ratio = len(data) / 60  # Assuming we want 60 days of data
            
            confidence = (data_quality + volatility_confidence + recent_data_ratio) / 3
            return min(1.0, max(0.0, confidence))
            
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return 0.5
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI (Relative Strength Index)"""
        try:
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi
        except Exception as e:
            print(f"Error calculating RSI: {e}")
            return pd.Series([50] * len(prices), index=prices.index)
    
    def train_new_model(self, ticker: str, period: str = "2y") -> Dict[str, Any]:
        """Train a new forecasting model for a specific ticker"""
        try:
            print(f"Training new model for {ticker}...")
            
            # Download data
            data = yf.download(ticker, period=period, interval="1d", auto_adjust=True)
            data.reset_index(inplace=True)
            
            if data.empty:
                return {"error": f"No data available for {ticker}"}
            
            # Prepare features
            features_df = self._prepare_features(data, ticker)
            
            if features_df.empty:
                return {"error": "Insufficient data for training"}
            
            # Define features and target - use only available columns
            available_cols = features_df.columns.tolist()
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if we have the minimum required columns
            missing_cols = [col for col in required_cols if col not in available_cols]
            if missing_cols:
                return {"error": f"Missing required columns: {missing_cols}"}
            
            # Use available feature columns (exclude Volume as it's our target)
            feature_cols = [col for col in available_cols if col != 'Volume' and col != 'Date']
            
            if len(feature_cols) < 3:
                return {"error": "Insufficient features for training"}
            
            X = features_df[feature_cols]
            y = features_df['Volume']
            
            # Train model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Evaluate model
            y_pred = model.predict(X)
            mse = mean_squared_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            
            # Save model
            model_path = os.path.join(self.models_dir, f"{ticker}_volume_model.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            return {
                "ticker": ticker,
                "model_saved": True,
                "model_path": model_path,
                "performance": {
                    "mse": mse,
                    "r2_score": r2,
                    "training_samples": len(X)
                },
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Model training failed: {str(e)}"} 