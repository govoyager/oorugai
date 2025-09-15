
"""
Simple AI Stock Analysis Example
This is a simplified version to demonstrate the core concepts
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class SimpleStockAnalyzer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_columns = None

    def get_stock_data(self, symbol, period="1y"):
        """Get stock data from Yahoo Finance"""
        try:
            # Add .NS for Indian stocks
            if not symbol.endswith('.NS'):
                symbol += '.NS'

            stock = yf.Ticker(symbol)
            data = stock.history(period=period)
            return data
        except Exception as e:
            print(f"Error fetching data for {symbol}: {e}")
            return None

    def calculate_indicators(self, data):
        """Calculate technical indicators"""
        df = data.copy()

        # Moving averages
        df['SMA_10'] = df['Close'].rolling(window=10).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # Price change
        df['Price_Change'] = df['Close'].pct_change()
        df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(window=20).mean()

        # Target: Next day price direction (1 if up, 0 if down)
        df['Next_Close'] = df['Close'].shift(-1)
        df['Target'] = (df['Next_Close'] > df['Close']).astype(int)

        return df

    def prepare_features(self, df):
        """Prepare features for machine learning"""
        # Select feature columns
        feature_cols = ['SMA_10', 'SMA_20', 'EMA_12', 'RSI', 'Price_Change', 'Volume_Ratio']

        # Remove rows with NaN values
        df_clean = df[feature_cols + ['Target']].dropna()

        if len(df_clean) < 50:
            print("Not enough data for training")
            return None, None

        X = df_clean[feature_cols]
        y = df_clean['Target']

        self.feature_columns = feature_cols
        return X, y

    def train_model(self, X, y):
        """Train the machine learning model"""
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train Random Forest model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_scaled, y)

        # Calculate accuracy
        accuracy = self.model.score(X_scaled, y)
        print(f"Model training accuracy: {accuracy:.2f}")

        return accuracy

    def predict_direction(self, df):
        """Predict price direction for the latest data"""
        if self.model is None or self.scaler is None:
            return None

        # Get latest features
        latest_features = df[self.feature_columns].iloc[-1:].values

        # Scale features
        latest_features_scaled = self.scaler.transform(latest_features)

        # Make prediction
        prediction = self.model.predict(latest_features_scaled)[0]
        probability = self.model.predict_proba(latest_features_scaled)[0]

        return {
            'prediction': prediction,
            'probability_down': probability[0],
            'probability_up': probability[1],
            'confidence': max(probability)
        }

    def get_trading_signal(self, df):
        """Generate simple trading signal"""
        latest = df.iloc[-1]

        # Technical analysis signals
        rsi_signal = 0
        if latest['RSI'] < 30:  # Oversold - buy signal
            rsi_signal = 1
        elif latest['RSI'] > 70:  # Overbought - sell signal
            rsi_signal = -1

        # Moving average signal
        ma_signal = 0
        if latest['Close'] > latest['SMA_20']:  # Price above MA - bullish
            ma_signal = 1
        elif latest['Close'] < latest['SMA_20']:  # Price below MA - bearish
            ma_signal = -1

        # Volume signal
        volume_signal = 1 if latest['Volume_Ratio'] > 1.5 else 0

        # ML prediction
        ml_result = self.predict_direction(df)
        ml_signal = 1 if ml_result and ml_result['prediction'] == 1 else 0

        # Combine signals
        total_signal = rsi_signal + ma_signal + volume_signal + ml_signal

        return {
            'rsi_signal': rsi_signal,
            'ma_signal': ma_signal,
            'volume_signal': volume_signal,
            'ml_signal': ml_signal,
            'total_signal': total_signal,
            'ml_confidence': ml_result['confidence'] if ml_result else 0,
            'recommendation': 'BUY' if total_signal >= 2 else 'SELL' if total_signal <= -1 else 'HOLD'
        }

    def analyze_stock(self, symbol):
        """Complete analysis of a stock"""
        print(f"\n=== Analyzing {symbol} ===")

        # Get data
        data = self.get_stock_data(symbol)
        if data is None or len(data) < 100:
            print("Insufficient data for analysis")
            return None

        # Calculate indicators
        df = self.calculate_indicators(data)

        # Prepare features and train model
        X, y = self.prepare_features(df)
        if X is None:
            print("Could not prepare features")
            return None

        # Train model
        accuracy = self.train_model(X, y)

        # Get trading signal
        signal = self.get_trading_signal(df)

        # Latest stock info
        latest = df.iloc[-1]

        result = {
            'symbol': symbol,
            'current_price': latest['Close'],
            'rsi': latest['RSI'],
            'sma_20': latest['SMA_20'],
            'model_accuracy': accuracy,
            'signal': signal
        }

        return result


def main():
    """Example usage"""
    analyzer = SimpleStockAnalyzer()

    # List of Indian stocks to analyze
    stocks = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']

    print("Simple AI Stock Analysis")
    print("=" * 40)

    for stock in stocks:
        try:
            result = analyzer.analyze_stock(stock)

            if result:
                print(f"\nStock: {result['symbol']}")
                print(f"Current Price: ₹{result['current_price']:.2f}")
                print(f"RSI: {result['rsi']:.2f}")
                print(f"20-day SMA: ₹{result['sma_20']:.2f}")
                print(f"Model Accuracy: {result['model_accuracy']:.2%}")
                print(f"\nSignals:")
                print(f"  RSI Signal: {result['signal']['rsi_signal']}")
                print(f"  MA Signal: {result['signal']['ma_signal']}")
                print(f"  Volume Signal: {result['signal']['volume_signal']}")
                print(f"  ML Signal: {result['signal']['ml_signal']}")
                print(f"  Total Score: {result['signal']['total_signal']}")
                print(f"  ML Confidence: {result['signal']['ml_confidence']:.2%}")
                print(f"\n🎯 RECOMMENDATION: {result['signal']['recommendation']}")
                print("-" * 50)

        except Exception as e:
            print(f"Error analyzing {stock}: {e}")

    print("\n⚠️  DISCLAIMER: This is for educational purposes only.")
    print("Always do your own research before making investment decisions!")


if __name__ == "__main__":
    main()
