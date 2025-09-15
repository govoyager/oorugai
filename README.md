# AI Stock Trading System Setup Guide

## Prerequisites

1. **Python 3.8 or higher**
2. **Zerodha Demat Account** with Kite Connect API access
3. **API Subscription** (₹500/month for live data and historical data)

## Installation Steps

### 1. Clone or Download the Code
```bash
# If using git
git clone <repository_url>
cd ai-trading-system

# Or download the files directly
```

### 2. Create Virtual Environment
```bash
python -m venv ai_trading_env
source ai_trading_env/bin/activate  # Linux/Mac
# or
ai_trading_env\Scripts\activate  # Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Install TA-Lib (Optional but Recommended)
```bash
# On Ubuntu/Debian
sudo apt-get install build-essential
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
tar -xzf ta-lib-0.4.0-src.tar.gz
cd ta-lib/
./configure --prefix=/usr
make
sudo make install
pip install TA-Lib

# On Windows
# Download TA-Lib wheel from https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# pip install TA_Lib-0.4.XX-cpXX-cpXXm-win_amd64.whl

# On macOS
brew install ta-lib
pip install TA-Lib
```

## Zerodha API Setup

### 1. Create Kite Connect App
1. Go to https://developers.kite.trade/
2. Sign up with your Zerodha credentials
3. Create a new app
4. Note down your API Key and API Secret

### 2. Subscribe to API Plan
1. Choose between:
   - **Personal (Free)**: Order management only, no live data
   - **Connect (₹500/month)**: Full API access with live and historical data

### 3. Get API Credentials
```python
# In your code, you'll need:
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
ACCESS_TOKEN = "generated_during_login_flow"
```

## Configuration

### 1. Environment Variables (Recommended)
Create a `.env` file:
```
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token
PAPER_TRADING=True
```

### 2. Update Configuration in Code
Edit the `ai_trading_system.py` file:
```python
# Replace these with your actual credentials
API_KEY = "your_api_key_here"
API_SECRET = "your_api_secret_here"
ACCESS_TOKEN = "your_access_token_here"  # Optional

# Configure your trading symbols
'symbols': ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK']
```

## Running the System

### 1. Initial Setup and Testing
```bash
python ai_trading_system.py
```

### 2. Paper Trading Mode (Recommended for Testing)
The system starts in paper trading mode by default. No real trades are executed.

### 3. Live Trading Mode
**WARNING: Only enable after thorough testing**
```python
# In the config section
'paper_trading': False  # Set to False for live trading
```

## Features Overview

### 1. Data Fetching
- Real-time stock data from Zerodha API
- Historical data for backtesting
- Multiple stock symbols support

### 2. Technical Analysis
- 15+ technical indicators
- Moving averages (SMA, EMA)
- RSI, MACD, Bollinger Bands
- Volume analysis

### 3. Machine Learning Models
- **Random Forest**: Ensemble method for classification
- **XGBoost**: Gradient boosting for high performance
- **SVM**: Support Vector Machines for pattern recognition
- **LSTM**: Deep learning for time series (optional)

### 4. Signal Generation
- ML-based predictions
- Technical analysis signals
- Ensemble approach combining multiple signals
- Confidence thresholds

### 5. Risk Management
- Position sizing based on portfolio risk
- Stop-loss and take-profit calculation
- Maximum position limits
- ATR-based risk calculation

### 6. Automated Trading
- Real-time order execution
- Portfolio monitoring
- Daily model retraining
- Scheduled analysis cycles

## Usage Examples

### 1. Basic Analysis
```python
from ai_trading_system import AITradingSystem

# Initialize system
system = AITradingSystem(API_KEY, API_SECRET, ACCESS_TOKEN)
system.initialize_system()

# Analyze single symbol
result = system.analyze_symbol('RELIANCE')
print(result)
```

### 2. Run Analysis Cycle
```python
# Run analysis for all configured symbols
results = system.run_analysis_cycle()
for result in results:
    print(f"{result['symbol']}: Signal={result['final_signal']}")
```

### 3. Start Automated Trading
```python
# Start continuous trading loop (use with caution)
system.start_trading_loop()
```

## Customization

### 1. Add New Symbols
```python
self.config['symbols'] = ['RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'YOUR_SYMBOL']
```

### 2. Adjust Model Parameters
```python
# In the MLModels class
model = RandomForestClassifier(
    n_estimators=200,  # Increase for better accuracy
    max_depth=15,      # Adjust based on data complexity
    random_state=42
)
```

### 3. Modify Technical Indicators
```python
# In TechnicalIndicators class, add new indicators
@staticmethod
def custom_indicator(data, period):
    # Your custom logic here
    return result
```

### 4. Change Trading Schedule
```python
# In start_trading_loop method
schedule.every(30).minutes.do(self.run_analysis_cycle)  # Every 30 minutes
schedule.every().day.at("16:00").do(self.retrain_models)  # 4 PM retraining
```

## Risk Management and Safety

### 1. Start with Paper Trading
- Always test extensively before live trading
- Verify signal accuracy over multiple market conditions
- Monitor performance metrics

### 2. Set Appropriate Limits
```python
self.risk_manager = RiskManager(
    max_portfolio_risk=0.01,  # 1% max risk per trade
    max_position_size=0.05    # 5% max position size
)
```

### 3. Monitor System Performance
- Check logs regularly
- Monitor API rate limits
- Verify model predictions against actual outcomes

### 4. Backup and Recovery
- Save model states regularly
- Keep configuration backups
- Monitor API connectivity

## Troubleshooting

### Common Issues

1. **API Connection Errors**
   - Verify API credentials
   - Check API subscription status
   - Ensure proper internet connectivity

2. **Data Fetching Issues**
   - Verify symbol names (use NSE format)
   - Check market hours
   - Verify API rate limits

3. **Model Training Errors**
   - Ensure sufficient historical data
   - Check for missing values in data
   - Verify feature column consistency

4. **Import Errors**
   - Install all required packages
   - Check Python version compatibility
   - Verify virtual environment activation

### Getting Help

1. Check the log files (`ai_trading_system.log`)
2. Verify all dependencies are installed
3. Test with paper trading first
4. Review Zerodha API documentation: https://kite.trade/docs/

## Disclaimers

⚠️ **Important Warnings:**

1. **Financial Risk**: Trading involves substantial risk and may not be suitable for all investors
2. **No Guarantees**: Past performance does not guarantee future results
3. **Testing Required**: Thoroughly test before using real money
4. **Regulatory Compliance**: Ensure compliance with local trading regulations
5. **API Limits**: Be aware of API rate limits and costs

## License and Support

This is an educational/demonstration system. Use at your own risk.
For production use, consider additional testing, risk management, and compliance measures.

---

**Remember**: Always start with paper trading and never risk more than you can afford to lose!
