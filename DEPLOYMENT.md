
# AI Stock Trading System - Advanced Deployment Guide

## Cloud Deployment Options

### 1. AWS EC2 Deployment

#### Step 1: Launch EC2 Instance
```bash
# Launch Ubuntu 20.04 LTS instance
# Recommended: t3.medium (2 vCPU, 4GB RAM) or higher
# Security group: Allow SSH (22) and custom ports if needed
```

#### Step 2: Setup Environment
```bash
# Connect to instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Python and dependencies
sudo apt install python3 python3-pip python3-venv git -y

# Clone your repository
git clone <your-repo-url>
cd ai-trading-system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

#### Step 3: Configure as Service
```bash
# Create systemd service file
sudo nano /etc/systemd/system/ai-trading.service
```

```ini
[Unit]
Description=AI Trading System
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/ai-trading-system
Environment=PATH=/home/ubuntu/ai-trading-system/venv/bin
ExecStart=/home/ubuntu/ai-trading-system/venv/bin/python ai_trading_system.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ai-trading.service
sudo systemctl start ai-trading.service

# Check status
sudo systemctl status ai-trading.service
```

### 2. Google Cloud Platform (GCP)

#### Using Compute Engine
```bash
# Create VM instance
gcloud compute instances create ai-trading-vm \
    --machine-type=e2-medium \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --boot-disk-size=20GB

# SSH into instance
gcloud compute ssh ai-trading-vm

# Follow similar setup as EC2
```

#### Using Cloud Run (Containerized)
```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "ai_trading_system.py"]
```

```bash
# Build and deploy
docker build -t ai-trading .
gcloud run deploy ai-trading --image ai-trading --platform managed
```

### 3. Digital Ocean Droplet

```bash
# Create droplet (4GB RAM, 2 vCPUs recommended)
# Follow similar setup as EC2

# Use screen or tmux for persistent sessions
sudo apt install screen
screen -S trading
# Run your application
# Detach: Ctrl+A, D
# Reattach: screen -r trading
```

## Production Configuration

### 1. Environment Variables
```bash
# Create .env file
cat > .env << EOF
ZERODHA_API_KEY=your_api_key
ZERODHA_API_SECRET=your_api_secret
ZERODHA_ACCESS_TOKEN=your_access_token

# Database settings
DATABASE_URL=postgresql://user:pass@localhost/trading_db

# Redis for caching
REDIS_URL=redis://localhost:6379

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/ai-trading/system.log

# Trading settings
PAPER_TRADING=false
MAX_POSITIONS=10
PORTFOLIO_SIZE=1000000
RISK_PER_TRADE=0.02

# Monitoring
DISCORD_WEBHOOK_URL=your_webhook_url
EMAIL_NOTIFICATIONS=true
SMTP_SERVER=smtp.gmail.com
EMAIL_USER=your_email
EMAIL_PASS=your_app_password
EOF
```

### 2. Database Integration
```python
# Add to ai_trading_system.py

import psycopg2
from sqlalchemy import create_engine
import redis

class DatabaseManager:
    def __init__(self, db_url):
        self.engine = create_engine(db_url)
        self.redis_client = redis.Redis.from_url(os.getenv('REDIS_URL'))

    def save_prediction(self, symbol, prediction, timestamp):
        """Save ML predictions to database"""
        query = """
        INSERT INTO predictions (symbol, prediction, confidence, timestamp)
        VALUES (%s, %s, %s, %s)
        """
        # Implementation here

    def save_trade(self, trade_data):
        """Save trade execution data"""
        # Implementation here

    def get_performance_metrics(self):
        """Get trading performance metrics"""
        # Implementation here
```

### 3. Monitoring and Alerts

#### Discord Notifications
```python
import requests

class NotificationManager:
    def __init__(self):
        self.discord_webhook = os.getenv('DISCORD_WEBHOOK_URL')

    def send_discord_alert(self, message):
        """Send alert to Discord channel"""
        if self.discord_webhook:
            data = {"content": message}
            requests.post(self.discord_webhook, json=data)

    def send_trade_alert(self, symbol, action, price, quantity):
        """Send trade execution alert"""
        message = f"🔔 TRADE EXECUTED\n"
        message += f"Symbol: {symbol}\n"
        message += f"Action: {action}\n"
        message += f"Price: ₹{price}\n"
        message += f"Quantity: {quantity}"
        self.send_discord_alert(message)
```

#### Email Notifications
```python
import smtplib
from email.mime.text import MIMEText

class EmailNotifier:
    def __init__(self):
        self.smtp_server = os.getenv('SMTP_SERVER')
        self.email_user = os.getenv('EMAIL_USER')
        self.email_pass = os.getenv('EMAIL_PASS')

    def send_email(self, subject, body, to_email):
        """Send email notification"""
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = self.email_user
        msg['To'] = to_email

        with smtplib.SMTP(self.smtp_server, 587) as server:
            server.starttls()
            server.login(self.email_user, self.email_pass)
            server.send_message(msg)
```

### 4. Performance Monitoring
```python
import psutil
import time

class SystemMonitor:
    def __init__(self):
        self.start_time = time.time()

    def get_system_stats(self):
        """Get system performance statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'uptime': time.time() - self.start_time
        }

    def check_system_health(self):
        """Check if system is healthy"""
        stats = self.get_system_stats()

        alerts = []
        if stats['cpu_percent'] > 80:
            alerts.append("High CPU usage")
        if stats['memory_percent'] > 85:
            alerts.append("High memory usage")
        if stats['disk_usage'] > 90:
            alerts.append("High disk usage")

        return alerts
```

## Security Best Practices

### 1. API Key Security
```python
import os
from cryptography.fernet import Fernet

class SecureConfig:
    def __init__(self):
        self.key = os.getenv('ENCRYPTION_KEY') or Fernet.generate_key()
        self.cipher = Fernet(self.key)

    def encrypt_api_key(self, api_key):
        """Encrypt API key for storage"""
        return self.cipher.encrypt(api_key.encode()).decode()

    def decrypt_api_key(self, encrypted_key):
        """Decrypt API key for use"""
        return self.cipher.decrypt(encrypted_key.encode()).decode()
```

### 2. Rate Limiting
```python
import time
from collections import defaultdict

class RateLimiter:
    def __init__(self, max_calls=100, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = defaultdict(list)

    def can_make_call(self, endpoint):
        """Check if we can make API call"""
        now = time.time()
        # Clean old calls
        self.calls[endpoint] = [
            call_time for call_time in self.calls[endpoint]
            if now - call_time < self.time_window
        ]

        if len(self.calls[endpoint]) < self.max_calls:
            self.calls[endpoint].append(now)
            return True
        return False
```

### 3. Backup and Recovery
```bash
#!/bin/bash
# backup_system.sh

# Create backup directory
mkdir -p /backup/$(date +%Y%m%d)

# Backup configuration
cp -r /home/ubuntu/ai-trading-system/config /backup/$(date +%Y%m%d)/

# Backup models
cp -r /home/ubuntu/ai-trading-system/models /backup/$(date +%Y%m%d)/

# Backup database
pg_dump trading_db > /backup/$(date +%Y%m%d)/database.sql

# Upload to cloud storage (optional)
aws s3 cp /backup/$(date +%Y%m%d) s3://your-backup-bucket/ --recursive

# Keep only last 30 days of backups
find /backup -type d -mtime +30 -exec rm -rf {} +
```

## Scaling and Optimization

### 1. Multi-Symbol Processing
```python
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

class ScalableAnalyzer:
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def analyze_multiple_symbols(self, symbols):
        """Analyze multiple symbols in parallel"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self.analyze_symbol, symbol): symbol 
                for symbol in symbols
            }

            results = {}
            for future in futures:
                symbol = futures[future]
                try:
                    results[symbol] = future.result()
                except Exception as e:
                    print(f"Error analyzing {symbol}: {e}")

            return results
```

### 2. Caching Strategy
```python
import redis
import pickle
import hashlib

class CacheManager:
    def __init__(self, redis_url):
        self.redis_client = redis.Redis.from_url(redis_url)
        self.default_ttl = 300  # 5 minutes

    def cache_key(self, symbol, analysis_type):
        """Generate cache key"""
        key_string = f"{symbol}:{analysis_type}"
        return hashlib.md5(key_string.encode()).hexdigest()

    def get_cached_analysis(self, symbol, analysis_type):
        """Get cached analysis result"""
        key = self.cache_key(symbol, analysis_type)
        cached = self.redis_client.get(key)

        if cached:
            return pickle.loads(cached)
        return None

    def cache_analysis(self, symbol, analysis_type, result, ttl=None):
        """Cache analysis result"""
        key = self.cache_key(symbol, analysis_type)
        ttl = ttl or self.default_ttl

        self.redis_client.setex(
            key, 
            ttl, 
            pickle.dumps(result)
        )
```

### 3. Load Balancing
```python
# nginx.conf for load balancing multiple instances

upstream ai_trading {
    server localhost:8001;
    server localhost:8002;
    server localhost:8003;
}

server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://ai_trading;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Maintenance and Updates

### 1. Automated Updates
```bash
#!/bin/bash
# update_system.sh

cd /home/ubuntu/ai-trading-system

# Pull latest code
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/

# Restart service if tests pass
if [ $? -eq 0 ]; then
    sudo systemctl restart ai-trading.service
    echo "System updated successfully"
else
    echo "Tests failed, update aborted"
    exit 1
fi
```

### 2. Health Checks
```python
class HealthChecker:
    def __init__(self, trading_system):
        self.system = trading_system

    def check_api_connectivity(self):
        """Check if APIs are accessible"""
        try:
            # Test Zerodha API
            if self.system.zerodha_api.kite:
                profile = self.system.zerodha_api.kite.profile()
                return True
        except:
            return False

    def check_model_performance(self):
        """Check if models are performing well"""
        # Implementation to check model metrics
        pass

    def run_health_check(self):
        """Run complete health check"""
        checks = {
            'api_connectivity': self.check_api_connectivity(),
            'model_performance': self.check_model_performance(),
            'system_resources': self.check_system_resources()
        }

        return all(checks.values()), checks
```

## Troubleshooting Common Issues

### 1. Memory Issues
```bash
# Monitor memory usage
free -h
top -o %MEM

# Add swap space if needed
sudo dd if=/dev/zero of=/swapfile bs=1G count=2
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### 2. API Rate Limits
```python
# Implement exponential backoff
import time
import random

def api_call_with_backoff(func, max_retries=3):
    for attempt in range(max_retries):
        try:
            return func()
        except RateLimitError:
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise
```

### 3. Database Connection Issues
```python
# Connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    database_url,
    poolclass=QueuePool,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True
)
```

## Cost Optimization

### 1. AWS Cost Management
- Use spot instances for non-critical workloads
- Set up billing alerts
- Use reserved instances for predictable workloads
- Monitor with AWS Cost Explorer

### 2. API Cost Management
- Cache API responses when possible
- Use batch API calls when available
- Monitor API usage quotas
- Optimize data fetching frequency

### 3. Resource Optimization
```python
# Optimize pandas operations
import pandas as pd

# Use appropriate data types
df['price'] = df['price'].astype('float32')  # Instead of float64
df['volume'] = df['volume'].astype('int32')   # Instead of int64

# Use chunking for large datasets
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    process_chunk(chunk)
```

---

This deployment guide provides a comprehensive framework for taking your AI trading system from development to production. Always test thoroughly and start with paper trading before deploying with real money.
