# Create the main AI trading system code
ai_trading_system_code = '''
"""
AI Stock Market Trading System with Zerodha API
Author: AI Assistant
Version: 1.0
Description: Automated stock trading system with ML predictions, technical analysis, and daily retraining
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Kite Connect API imports
try:
    from kiteconnect import KiteConnect
except ImportError:
    print("Please install kiteconnect: pip install kiteconnect")

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Deep Learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. LSTM models will be disabled.")

# Technical Analysis imports
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("TA-Lib not available. Using pandas for technical indicators.")

import logging
import json
import time
import schedule
from typing import Dict, List, Tuple, Optional


class ZerodhaAPIHandler:
    """Handle Zerodha Kite API operations"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        self.api_key = api_key
        self.api_secret = api_secret
        self.access_token = access_token
        self.kite = None
        
        if access_token:
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
    
    def get_login_url(self) -> str:
        """Get login URL for authorization"""
        kite = KiteConnect(api_key=self.api_key)
        return kite.login_url()
    
    def generate_session(self, request_token: str) -> Dict:
        """Generate access token from request token"""
        kite = KiteConnect(api_key=self.api_key)
        data = kite.generate_session(request_token, api_secret=self.api_secret)
        self.access_token = data["access_token"]
        self.kite = kite
        self.kite.set_access_token(self.access_token)
        return data
    
    def get_instruments(self, exchange: str = "NSE") -> pd.DataFrame:
        """Get list of tradeable instruments"""
        instruments = self.kite.instruments(exchange)
        return pd.DataFrame(instruments)
    
    def get_historical_data(self, instrument_token: str, from_date: str, 
                          to_date: str, interval: str = "day") -> pd.DataFrame:
        """Get historical candle data"""
        data = self.kite.historical_data(
            instrument_token=instrument_token,
            from_date=from_date,
            to_date=to_date,
            interval=interval
        )
        return pd.DataFrame(data)
    
    def get_quote(self, symbols: List[str]) -> Dict:
        """Get real-time quotes"""
        return self.kite.quote(symbols)
    
    def place_order(self, tradingsymbol: str, transaction_type: str, 
                   quantity: int, order_type: str = "MARKET", 
                   product: str = "CNC", price: float = None) -> str:
        """Place buy/sell order"""
        return self.kite.place_order(
            tradingsymbol=tradingsymbol,
            exchange=self.kite.EXCHANGE_NSE,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=product,
            variety=self.kite.VARIETY_REGULAR,
            price=price
        )
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.kite.positions()
    
    def get_holdings(self) -> List:
        """Get holdings"""
        return self.kite.holdings()


class TechnicalIndicators:
    """Calculate technical indicators for stock analysis"""
    
    @staticmethod
    def sma(data: pd.Series, period: int) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=period).mean()
    
    @staticmethod
    def ema(data: pd.Series, period: int) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=period).mean()
    
    @staticmethod
    def rsi(data: pd.Series, period: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple:
        """MACD Indicator"""
        ema_fast = TechnicalIndicators.ema(data, fast)
        ema_slow = TechnicalIndicators.ema(data, slow)
        macd_line = ema_fast - ema_slow
        signal_line = TechnicalIndicators.ema(macd_line, signal)
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    @staticmethod
    def bollinger_bands(data: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple:
        """Bollinger Bands"""
        sma = TechnicalIndicators.sma(data, period)
        std = data.rolling(window=period).std()
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        return upper_band, sma, lower_band
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Average True Range"""
        high_low = high - low
        high_close = np.abs(high - close.shift())
        low_close = np.abs(low - close.shift())
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_period: int = 14, d_period: int = 3) -> Tuple:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        return k_percent, d_percent


class FeatureEngineering:
    """Create features for machine learning models"""
    
    def __init__(self):
        self.technical_indicators = TechnicalIndicators()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive feature set"""
        features_df = df.copy()
        
        # Price-based features
        features_df['price_change'] = df['close'].diff()
        features_df['price_change_pct'] = df['close'].pct_change()
        features_df['volatility'] = df['close'].rolling(window=20).std()
        features_df['volume_sma'] = self.technical_indicators.sma(df['volume'], 20)
        features_df['volume_ratio'] = df['volume'] / features_df['volume_sma']
        
        # Technical indicators
        features_df['sma_5'] = self.technical_indicators.sma(df['close'], 5)
        features_df['sma_10'] = self.technical_indicators.sma(df['close'], 10)
        features_df['sma_20'] = self.technical_indicators.sma(df['close'], 20)
        features_df['sma_50'] = self.technical_indicators.sma(df['close'], 50)
        
        features_df['ema_12'] = self.technical_indicators.ema(df['close'], 12)
        features_df['ema_26'] = self.technical_indicators.ema(df['close'], 26)
        features_df['ema_50'] = self.technical_indicators.ema(df['close'], 50)
        
        features_df['rsi'] = self.technical_indicators.rsi(df['close'])
        
        macd, macd_signal, macd_hist = self.technical_indicators.macd(df['close'])
        features_df['macd'] = macd
        features_df['macd_signal'] = macd_signal
        features_df['macd_histogram'] = macd_hist
        
        bb_upper, bb_middle, bb_lower = self.technical_indicators.bollinger_bands(df['close'])
        features_df['bb_upper'] = bb_upper
        features_df['bb_middle'] = bb_middle
        features_df['bb_lower'] = bb_lower
        features_df['bb_width'] = (bb_upper - bb_lower) / bb_middle
        
        features_df['atr'] = self.technical_indicators.atr(df['high'], df['low'], df['close'])
        
        stoch_k, stoch_d = self.technical_indicators.stochastic(
            df['high'], df['low'], df['close']
        )
        features_df['stoch_k'] = stoch_k
        features_df['stoch_d'] = stoch_d
        
        # Price position features
        features_df['price_vs_sma20'] = df['close'] / features_df['sma_20']
        features_df['price_vs_ema20'] = df['close'] / features_df['ema_12']
        
        # Time-based features
        if 'date' in df.columns or df.index.name == 'date':
            date_col = df.index if df.index.name == 'date' else df['date']
            features_df['hour'] = date_col.hour if hasattr(date_col, 'hour') else 0
            features_df['day_of_week'] = date_col.dayofweek if hasattr(date_col, 'dayofweek') else 0
            features_df['month'] = date_col.month if hasattr(date_col, 'month') else 0
            features_df['quarter'] = date_col.quarter if hasattr(date_col, 'quarter') else 0
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            features_df[f'close_lag_{lag}'] = df['close'].shift(lag)
            features_df[f'volume_lag_{lag}'] = df['volume'].shift(lag)
            features_df[f'rsi_lag_{lag}'] = features_df['rsi'].shift(lag)
        
        return features_df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for prediction"""
        target_df = df.copy()
        
        # Next day price prediction
        target_df['next_close'] = df['close'].shift(-1)
        target_df['price_direction'] = (target_df['next_close'] > df['close']).astype(int)
        target_df['price_change_next'] = target_df['next_close'] - df['close']
        target_df['price_change_pct_next'] = target_df['price_change_next'] / df['close']
        
        return target_df


class MLModels:
    """Machine Learning models for stock prediction"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.feature_columns = None
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, 
                    test_size: float = 0.2) -> Tuple:
        """Prepare data for ML training"""
        # Remove rows with NaN values
        df_clean = df.dropna()
        
        # Select feature columns (exclude target and date columns)
        exclude_cols = ['date', 'next_close', 'price_direction', 
                       'price_change_next', 'price_change_pct_next']
        feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df_clean[feature_cols]
        y = df_clean[target_column]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers[target_column] = scaler
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_random_forest_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
        """Train Random Forest classifier"""
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['rf_classifier'] = model
        return model
    
    def train_random_forest_regressor(self, X_train: np.ndarray, y_train: np.ndarray) -> RandomForestRegressor:
        """Train Random Forest regressor"""
        model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['rf_regressor'] = model
        return model
    
    def train_xgboost_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBClassifier:
        """Train XGBoost classifier"""
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        self.models['xgb_classifier'] = model
        return model
    
    def train_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> SVC:
        """Train SVM classifier"""
        model = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models['svm_classifier'] = model
        return model
    
    def train_lstm_model(self, X_train: np.ndarray, y_train: np.ndarray, 
                        sequence_length: int = 60) -> Optional[Sequential]:
        """Train LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot train LSTM model.")
            return None
        
        # Reshape data for LSTM
        X_train_lstm = []
        y_train_lstm = []
        
        for i in range(sequence_length, len(X_train)):
            X_train_lstm.append(X_train[i-sequence_length:i])
            y_train_lstm.append(y_train.iloc[i] if hasattr(y_train, 'iloc') else y_train[i])
        
        X_train_lstm = np.array(X_train_lstm)
        y_train_lstm = np.array(y_train_lstm)
        
        # Build LSTM model
        model = Sequential([
            LSTM(units=50, return_sequences=True, input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])),
            Dropout(0.2),
            LSTM(units=50, return_sequences=True),
            Dropout(0.2),
            LSTM(units=50),
            Dropout(0.2),
            Dense(units=1, activation='sigmoid' if len(np.unique(y_train_lstm)) == 2 else 'linear')
        ])
        
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy' if len(np.unique(y_train_lstm)) == 2 else 'mse',
            metrics=['accuracy'] if len(np.unique(y_train_lstm)) == 2 else ['mae']
        )
        
        model.fit(
            X_train_lstm, y_train_lstm,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            verbose=0
        )
        
        self.models['lstm'] = model
        return model
    
    def predict(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Make predictions using specified model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if model_name == 'lstm':
            # Handle LSTM prediction differently
            # This is a simplified version - in practice you'd need to maintain sequence
            return model.predict(X)
        else:
            return model.predict(X)
    
    def predict_proba(self, X: np.ndarray, model_name: str) -> np.ndarray:
        """Get prediction probabilities"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X)
        else:
            # For models without predict_proba, return predictions as probabilities
            predictions = model.predict(X)
            return np.column_stack([1 - predictions, predictions])


class SignalGenerator:
    """Generate buy/sell signals based on ML predictions and technical analysis"""
    
    def __init__(self, models: MLModels):
        self.models = models
        self.signals = []
    
    def generate_signals(self, df: pd.DataFrame, 
                        confidence_threshold: float = 0.7) -> pd.DataFrame:
        """Generate trading signals"""
        signals_df = df.copy()
        
        # Prepare features for prediction
        if self.models.feature_columns is None:
            raise ValueError("Models must be trained before generating signals")
        
        # Get latest data for prediction
        latest_features = df[self.models.feature_columns].iloc[-1:].values
        
        # Scale features
        if 'price_direction' in self.models.scalers:
            scaler = self.models.scalers['price_direction']
            latest_features_scaled = scaler.transform(latest_features)
        else:
            latest_features_scaled = latest_features
        
        # Get predictions from all models
        predictions = {}
        
        for model_name in self.models.models:
            try:
                if 'classifier' in model_name:
                    pred_proba = self.models.predict_proba(latest_features_scaled, model_name)
                    predictions[model_name] = pred_proba[0][1]  # Probability of class 1 (price up)
                else:
                    pred = self.models.predict(latest_features_scaled, model_name)
                    predictions[model_name] = pred[0]
            except Exception as e:
                print(f"Error predicting with {model_name}: {e}")
                continue
        
        # Ensemble prediction (average of all model predictions)
        if predictions:
            ensemble_prediction = np.mean(list(predictions.values()))
        else:
            ensemble_prediction = 0.5  # Neutral if no predictions
        
        # Technical analysis signals
        latest_data = df.iloc[-1]
        
        # RSI signals
        rsi_signal = 0
        if 'rsi' in df.columns and not pd.isna(latest_data['rsi']):
            if latest_data['rsi'] < 30:  # Oversold - buy signal
                rsi_signal = 1
            elif latest_data['rsi'] > 70:  # Overbought - sell signal
                rsi_signal = -1
        
        # MACD signals
        macd_signal = 0
        if all(col in df.columns for col in ['macd', 'macd_signal']):
            if (latest_data['macd'] > latest_data['macd_signal'] and 
                df['macd'].iloc[-2] <= df['macd_signal'].iloc[-2]):  # MACD crossover up
                macd_signal = 1
            elif (latest_data['macd'] < latest_data['macd_signal'] and 
                  df['macd'].iloc[-2] >= df['macd_signal'].iloc[-2]):  # MACD crossover down
                macd_signal = -1
        
        # Moving average signals
        ma_signal = 0
        if all(col in df.columns for col in ['close', 'sma_20', 'sma_50']):
            if (latest_data['close'] > latest_data['sma_20'] > latest_data['sma_50']):
                ma_signal = 1
            elif (latest_data['close'] < latest_data['sma_20'] < latest_data['sma_50']):
                ma_signal = -1
        
        # Volume signal
        volume_signal = 0
        if 'volume_ratio' in df.columns:
            if latest_data['volume_ratio'] > 1.5:  # High volume
                volume_signal = 1
        
        # Combine signals
        technical_score = (rsi_signal + macd_signal + ma_signal + volume_signal) / 4
        
        # Final signal generation
        final_signal = 0
        signal_strength = 0
        
        if ensemble_prediction > confidence_threshold and technical_score >= 0:
            final_signal = 1  # Buy signal
            signal_strength = (ensemble_prediction + technical_score) / 2
        elif ensemble_prediction < (1 - confidence_threshold) and technical_score <= 0:
            final_signal = -1  # Sell signal
            signal_strength = (1 - ensemble_prediction - technical_score) / 2
        
        return {
            'timestamp': datetime.now(),
            'ml_prediction': ensemble_prediction,
            'technical_score': technical_score,
            'final_signal': final_signal,
            'signal_strength': signal_strength,
            'rsi_signal': rsi_signal,
            'macd_signal': macd_signal,
            'ma_signal': ma_signal,
            'volume_signal': volume_signal,
            'individual_predictions': predictions
        }


class RiskManager:
    """Manage trading risk and position sizing"""
    
    def __init__(self, max_portfolio_risk: float = 0.02, max_position_size: float = 0.1):
        self.max_portfolio_risk = max_portfolio_risk  # Max 2% portfolio risk per trade
        self.max_position_size = max_position_size    # Max 10% in any single position
        self.current_positions = {}
        
    def calculate_position_size(self, portfolio_value: float, entry_price: float, 
                              stop_loss_price: float, signal_strength: float) -> int:
        """Calculate optimal position size"""
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss_price)
        
        # Maximum risk amount
        max_risk_amount = portfolio_value * self.max_portfolio_risk
        
        # Position size based on risk
        shares_by_risk = int(max_risk_amount / risk_per_share)
        
        # Position size based on signal strength
        adjusted_shares = int(shares_by_risk * signal_strength)
        
        # Maximum position value
        max_position_value = portfolio_value * self.max_position_size
        max_shares_by_value = int(max_position_value / entry_price)
        
        # Take minimum of all constraints
        final_shares = min(adjusted_shares, max_shares_by_value)
        
        return max(final_shares, 0)  # Ensure non-negative
    
    def calculate_stop_loss(self, entry_price: float, position_type: str, 
                          atr: float) -> float:
        """Calculate stop loss price"""
        if position_type == 'long':
            # Stop loss below entry price
            stop_loss = entry_price - (2 * atr)  # 2 ATR stop loss
        else:  # short position
            # Stop loss above entry price
            stop_loss = entry_price + (2 * atr)
        
        return stop_loss
    
    def calculate_take_profit(self, entry_price: float, position_type: str, 
                            atr: float, risk_reward_ratio: float = 2.0) -> float:
        """Calculate take profit price"""
        stop_loss = self.calculate_stop_loss(entry_price, position_type, atr)
        risk = abs(entry_price - stop_loss)
        
        if position_type == 'long':
            take_profit = entry_price + (risk * risk_reward_ratio)
        else:  # short position
            take_profit = entry_price - (risk * risk_reward_ratio)
        
        return take_profit


class AITradingSystem:
    """Main AI Trading System"""
    
    def __init__(self, api_key: str, api_secret: str, access_token: str = None):
        # Initialize components
        self.zerodha_api = ZerodhaAPIHandler(api_key, api_secret, access_token)
        self.feature_engineer = FeatureEngineering()
        self.ml_models = MLModels()
        self.signal_generator = None
        self.risk_manager = RiskManager()
        
        # Configuration
        self.config = {
            'symbols': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK'],  # Default symbols
            'retraining_frequency': 'daily',
            'confidence_threshold': 0.7,
            'paper_trading': True,  # Start with paper trading
            'max_positions': 5,
            'lookback_days': 252  # 1 year of data for training
        }
        
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('ai_trading_system.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def initialize_system(self):
        """Initialize the trading system"""
        try:
            self.logger.info("Initializing AI Trading System...")
            
            # Test API connection
            if self.zerodha_api.kite:
                profile = self.zerodha_api.kite.profile()
                self.logger.info(f"Connected to Zerodha API. User: {profile['user_name']}")
            else:
                self.logger.warning("Zerodha API not connected. Using demo mode.")
            
            # Load historical data and train initial models
            self.initial_training()
            
            self.logger.info("AI Trading System initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            raise
    
    def get_historical_data(self, symbol: str, days: int = 252) -> pd.DataFrame:
        """Get historical data for a symbol"""
        try:
            # Try to get data from Zerodha API first
            if self.zerodha_api.kite:
                # This would require instrument token lookup
                # For now, using yfinance as fallback
                pass
            
            # Fallback to yfinance for demo
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Add .NS suffix for NSE stocks
            yf_symbol = f"{symbol}.NS" if not symbol.endswith('.NS') else symbol
            
            data = yf.download(yf_symbol, start=start_date, end=end_date)
            
            if data.empty:
                self.logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Rename columns to lowercase
            data.columns = [col.lower() for col in data.columns]
            data.reset_index(inplace=True)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def initial_training(self):
        """Train initial ML models"""
        self.logger.info("Starting initial model training...")
        
        # Collect training data for all symbols
        all_data = []
        
        for symbol in self.config['symbols']:
            self.logger.info(f"Fetching data for {symbol}")
            data = self.get_historical_data(symbol, self.config['lookback_days'])
            
            if not data.empty:
                # Add symbol column
                data['symbol'] = symbol
                all_data.append(data)
            
        if not all_data:
            self.logger.error("No training data available")
            return
        
        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Feature engineering
        self.logger.info("Creating features...")
        features_data = self.feature_engineer.create_features(combined_data)
        target_data = self.feature_engineer.create_target_variables(features_data)
        
        # Train models
        self.logger.info("Training ML models...")
        
        # Prepare data for classification (price direction)
        X_train, X_test, y_train, y_test = self.ml_models.prepare_data(
            target_data, 'price_direction'
        )
        
        # Train multiple models
        models_trained = []
        
        try:
            self.ml_models.train_random_forest_classifier(X_train, y_train)
            models_trained.append('Random Forest Classifier')
        except Exception as e:
            self.logger.error(f"Error training Random Forest: {e}")
        
        try:
            self.ml_models.train_xgboost_classifier(X_train, y_train)
            models_trained.append('XGBoost Classifier')
        except Exception as e:
            self.logger.error(f"Error training XGBoost: {e}")
        
        try:
            # SVM might be slow for large datasets
            if len(X_train) < 10000:
                self.ml_models.train_svm_classifier(X_train, y_train)
                models_trained.append('SVM Classifier')
        except Exception as e:
            self.logger.error(f"Error training SVM: {e}")
        
        # Initialize signal generator
        self.signal_generator = SignalGenerator(self.ml_models)
        
        self.logger.info(f"Trained models: {', '.join(models_trained)}")
    
    def retrain_models(self):
        """Retrain models with latest data"""
        self.logger.info("Retraining models with latest data...")
        
        # This would be called daily after market close
        self.initial_training()  # For now, same as initial training
        
        self.logger.info("Model retraining completed")
    
    def analyze_symbol(self, symbol: str) -> Dict:
        """Analyze a single symbol and generate signals"""
        try:
            # Get latest data
            data = self.get_historical_data(symbol, days=100)  # Last 100 days for analysis
            
            if data.empty:
                return {'error': f'No data available for {symbol}'}
            
            # Feature engineering
            features_data = self.feature_engineer.create_features(data)
            
            # Generate signals
            if self.signal_generator:
                signal_info = self.signal_generator.generate_signals(features_data)
                signal_info['symbol'] = symbol
                return signal_info
            else:
                return {'error': 'Signal generator not initialized'}
                
        except Exception as e:
            self.logger.error(f"Error analyzing {symbol}: {e}")
            return {'error': str(e)}
    
    def execute_trade(self, symbol: str, signal: int, signal_strength: float, 
                     current_price: float) -> Dict:
        """Execute trade based on signal"""
        try:
            if self.config['paper_trading']:
                self.logger.info(f"PAPER TRADE - {symbol}: Signal={signal}, Strength={signal_strength:.2f}, Price={current_price}")
                return {'status': 'paper_trade', 'message': 'Trade logged in paper trading mode'}
            
            if not self.zerodha_api.kite:
                return {'status': 'error', 'message': 'API not connected'}
            
            # Get portfolio value (simplified - you'd get this from API)
            portfolio_value = 100000  # Assume 1 lakh portfolio
            
            # Calculate position size
            # Estimate ATR for risk management (simplified)
            atr = current_price * 0.02  # 2% of current price as rough ATR
            
            if signal == 1:  # Buy signal
                stop_loss = self.risk_manager.calculate_stop_loss(current_price, 'long', atr)
                position_size = self.risk_manager.calculate_position_size(
                    portfolio_value, current_price, stop_loss, signal_strength
                )
                
                if position_size > 0:
                    # Place buy order
                    order_id = self.zerodha_api.place_order(
                        tradingsymbol=symbol,
                        transaction_type="BUY",
                        quantity=position_size,
                        order_type="MARKET"
                    )
                    
                    self.logger.info(f"Buy order placed for {symbol}: {position_size} shares, Order ID: {order_id}")
                    return {'status': 'success', 'order_id': order_id, 'action': 'BUY', 'quantity': position_size}
            
            elif signal == -1:  # Sell signal
                # Check if we have positions to sell
                positions = self.zerodha_api.get_positions()
                # Implementation would check for existing positions and place sell orders
                
                self.logger.info(f"Sell signal for {symbol} - checking existing positions")
                return {'status': 'success', 'action': 'SELL_SIGNAL', 'message': 'Checking positions for sell'}
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def run_analysis_cycle(self):
        """Run one complete analysis and trading cycle"""
        self.logger.info("Starting analysis cycle...")
        
        results = []
        
        for symbol in self.config['symbols']:
            self.logger.info(f"Analyzing {symbol}...")
            
            # Analyze symbol
            analysis = self.analyze_symbol(symbol)
            
            if 'error' not in analysis:
                # Execute trade if signal is strong enough
                if abs(analysis['final_signal']) > 0 and analysis['signal_strength'] > self.config['confidence_threshold']:
                    
                    # Get current price (simplified - you'd get real-time price from API)
                    data = self.get_historical_data(symbol, days=5)
                    if not data.empty:
                        current_price = data['close'].iloc[-1]
                        
                        trade_result = self.execute_trade(
                            symbol, 
                            analysis['final_signal'], 
                            analysis['signal_strength'], 
                            current_price
                        )
                        
                        analysis['trade_result'] = trade_result
                
                results.append(analysis)
            else:
                self.logger.warning(f"Analysis failed for {symbol}: {analysis['error']}")
        
        self.logger.info(f"Analysis cycle completed. Processed {len(results)} symbols")
        return results
    
    def schedule_daily_retraining(self):
        """Schedule daily model retraining"""
        # Schedule retraining after market close (3:30 PM IST)
        schedule.every().day.at("15:45").do(self.retrain_models)
        
        self.logger.info("Daily retraining scheduled for 3:45 PM")
    
    def start_trading_loop(self):
        """Start the main trading loop"""
        self.logger.info("Starting AI Trading System...")
        
        # Schedule daily retraining
        self.schedule_daily_retraining()
        
        # Schedule regular analysis during trading hours
        # Run analysis every 15 minutes during market hours (9:15 AM to 3:30 PM)
        schedule.every(15).minutes.do(self.run_analysis_cycle)
        
        self.logger.info("Trading loop started. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("Trading loop stopped by user")
        except Exception as e:
            self.logger.error(f"Trading loop error: {e}")


# Example usage and configuration
def main():
    """Main function to run the AI Trading System"""
    
    # API credentials (replace with your actual credentials)
    API_KEY = "your_api_key_here"
    API_SECRET = "your_api_secret_here"
    ACCESS_TOKEN = "your_access_token_here"  # Optional, can be generated during runtime
    
    # Create and initialize the trading system
    trading_system = AITradingSystem(API_KEY, API_SECRET, ACCESS_TOKEN)
    
    try:
        # Initialize the system
        trading_system.initialize_system()
        
        # Run a single analysis cycle for testing
        results = trading_system.run_analysis_cycle()
        
        print("\\n=== ANALYSIS RESULTS ===")
        for result in results:
            if 'error' not in result:
                print(f"\\nSymbol: {result['symbol']}")
                print(f"ML Prediction: {result['ml_prediction']:.3f}")
                print(f"Technical Score: {result['technical_score']:.3f}")
                print(f"Final Signal: {result['final_signal']}")
                print(f"Signal Strength: {result['signal_strength']:.3f}")
                
                if 'trade_result' in result:
                    print(f"Trade Result: {result['trade_result']}")
        
        # Uncomment to start continuous trading
        # trading_system.start_trading_loop()
        
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
'''

# Save the code to a file
with open('ai_trading_system.py', 'w') as f:
    f.write(ai_trading_system_code)

print("AI Trading System code has been created and saved to 'ai_trading_system.py'")
print("\nFile size:", len(ai_trading_system_code), "characters")
print("Lines of code:", ai_trading_system_code.count('\n'))