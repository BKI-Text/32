# 🚀 Beverly Knits AI Supply Chain Planner - Deployment Guide

**Version:** 1.0.0  
**Last Updated:** January 2025  
**Status:** Production Ready  
**Document Type:** Deployment & Operations Guide

---

## 📋 Overview

This guide provides comprehensive instructions for deploying the Beverly Knits AI Supply Chain Optimization Planner in various environments, from local development to production cloud deployment.

## 🎯 Deployment Options

### 1. Local Development
- **Use Case**: Development and testing
- **Requirements**: Python 3.12+, Git
- **Setup Time**: 5-10 minutes
- **Scalability**: Single user

### 2. Docker Deployment
- **Use Case**: Containerized deployment
- **Requirements**: Docker, Docker Compose
- **Setup Time**: 10-15 minutes
- **Scalability**: Multiple users

### 3. Cloud Deployment
- **Use Case**: Production environment
- **Requirements**: Cloud provider account
- **Setup Time**: 30-60 minutes
- **Scalability**: Highly scalable

---

## 🔧 Local Development Setup

### Prerequisites
- **Python**: 3.12 or higher
- **Git**: Latest version
- **Storage**: 1GB free space
- **Memory**: 4GB RAM minimum

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd beverly-knits-ai-planner
```

### Step 2: Virtual Environment Setup
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/Mac:
source venv/bin/activate
# Windows:
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
# Install requirements
pip install -r requirements.txt

# Verify installation
python -c "import streamlit; print('Streamlit installed successfully')"
```

### Step 4: Configuration
```bash
# Copy sample configuration
cp config/app_config.json.sample config/app_config.json

# Edit configuration as needed
nano config/app_config.json
```

### Step 5: Run Application
```bash
# Start Streamlit application
streamlit run main.py

# Access application
# Open browser to: http://localhost:8501
```

### Step 6: Load Sample Data
```bash
# Generate sample data
python -m src.utils.sample_data_generator

# Or copy your data files
cp your-data/*.csv data/live/
```

---

## 🐳 Docker Deployment

### Prerequisites
- **Docker**: 20.10+ 
- **Docker Compose**: 2.0+
- **Storage**: 2GB free space
- **Memory**: 8GB RAM recommended

### Step 1: Build Docker Image
```bash
# Build application image
docker build -t beverly-knits-planner:latest .

# Verify build
docker images | grep beverly-knits-planner
```

### Step 2: Docker Compose Setup
```yaml
# docker-compose.yml
version: '3.8'

services:
  app:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
      - ./config:/app/config
    environment:
      - ENVIRONMENT=production
      - DEBUG=false
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

volumes:
  redis_data:
```

### Step 3: Start Services
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f app
```

### Step 4: Access Application
```bash
# Application URL
echo "Application available at: http://localhost:8501"

# Health check
curl -f http://localhost:8501/_stcore/health
```

---

## ☁️ Cloud Deployment

### Option 1: Streamlit Cloud

#### Prerequisites
- GitHub account
- Streamlit Cloud account
- Repository access

#### Deployment Steps
1. **Fork Repository** to your GitHub account
2. **Connect Streamlit Cloud** to your GitHub account
3. **Deploy Application**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select repository and branch
   - Set main file path: `main.py`
   - Click "Deploy"

#### Configuration
```toml
# .streamlit/config.toml
[server]
port = 8501
headless = true
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#2E86AB"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
```

### Option 2: AWS ECS Deployment

#### Prerequisites
- AWS account
- AWS CLI configured
- ECS cluster setup

#### Step 1: Create Task Definition
```json
{
  "family": "beverly-knits-planner",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "512",
  "memory": "1024",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "app",
      "image": "your-account.dkr.ecr.region.amazonaws.com/beverly-knits-planner:latest",
      "portMappings": [
        {
          "containerPort": 8501,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "ENVIRONMENT",
          "value": "production"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/beverly-knits-planner",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

#### Step 2: Deploy to ECS
```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster beverly-knits-cluster \
  --service-name beverly-knits-service \
  --task-definition beverly-knits-planner:1 \
  --desired-count 2 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-12345],securityGroups=[sg-12345],assignPublicIp=ENABLED}"
```

### Option 3: Google Cloud Run

#### Prerequisites
- Google Cloud account
- `gcloud` CLI installed
- Container Registry access

#### Step 1: Build and Push Image
```bash
# Configure Docker for GCR
gcloud auth configure-docker

# Build and tag image
docker build -t gcr.io/your-project/beverly-knits-planner:latest .

# Push to Container Registry
docker push gcr.io/your-project/beverly-knits-planner:latest
```

#### Step 2: Deploy to Cloud Run
```bash
# Deploy to Cloud Run
gcloud run deploy beverly-knits-planner \
  --image gcr.io/your-project/beverly-knits-planner:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8501 \
  --memory 2Gi \
  --cpu 1 \
  --set-env-vars ENVIRONMENT=production
```

---

## 🔒 Security Configuration

### Environment Variables
```bash
# Required environment variables
export ENVIRONMENT=production
export DEBUG=false
export SECRET_KEY=your-secret-key-here
export DATABASE_URL=postgresql://user:pass@host:5432/db
export REDIS_URL=redis://host:6379/0
```

### SSL/TLS Configuration
```bash
# Generate SSL certificate
openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes

# Configure Streamlit for HTTPS
streamlit run main.py --server.enableXsrfProtection=true --server.enableCORS=false
```

### Firewall Rules
```bash
# Allow only necessary ports
ufw allow 8501/tcp  # Streamlit application
ufw allow 22/tcp    # SSH access
ufw deny 5432/tcp   # Deny direct database access
```

---

## 📊 Monitoring and Logging

### Application Monitoring
```python
# Add to requirements.txt
prometheus-client==0.19.0
structlog==23.2.0

# Monitoring configuration
MONITORING_CONFIG = {
    "prometheus": {
        "port": 9090,
        "metrics_path": "/metrics"
    },
    "logging": {
        "level": "INFO",
        "format": "json",
        "file": "/var/log/beverly-knits.log"
    }
}
```

### Health Checks
```python
# src/utils/health_check.py
def health_check():
    """Application health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "database": check_database_connection(),
        "redis": check_redis_connection()
    }
```

### Log Aggregation
```yaml
# docker-compose.logging.yml
version: '3.8'

services:
  app:
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
  
  logstash:
    image: logstash:8.5.0
    volumes:
      - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf
    ports:
      - "5044:5044"
```

---

## 🔧 Backup and Recovery

### Data Backup Strategy
```bash
# Database backup
pg_dump -h localhost -U postgres beverly_knits > backup_$(date +%Y%m%d).sql

# File system backup
tar -czf data_backup_$(date +%Y%m%d).tar.gz data/

# Automated backup script
#!/bin/bash
# backup.sh
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump -h $DB_HOST -U $DB_USER $DB_NAME > /backups/db_backup_$DATE.sql
tar -czf /backups/data_backup_$DATE.tar.gz /app/data/
```

### Disaster Recovery
```bash
# Database restore
psql -h localhost -U postgres beverly_knits < backup_20250115.sql

# File system restore
tar -xzf data_backup_20250115.tar.gz -C /app/
```

---

## 🚨 Troubleshooting

### Common Issues

#### 1. Application Won't Start
```bash
# Check Python version
python --version

# Check dependencies
pip list | grep streamlit

# Check port availability
netstat -tlnp | grep 8501

# Check logs
tail -f logs/beverly_knits.log
```

#### 2. Data Loading Issues
```bash
# Check file permissions
ls -la data/live/

# Check file format
head -5 data/live/Style_BOM.csv

# Check data quality
python -m src.utils.data_quality_fixer
```

#### 3. Performance Issues
```bash
# Check memory usage
free -h

# Check CPU usage
top -p $(pgrep -f streamlit)

# Check disk space
df -h

# Optimize configuration
sed -i 's/debug=true/debug=false/g' config/app_config.json
```

### Performance Optimization
```python
# Streamlit configuration
# .streamlit/config.toml
[server]
maxUploadSize = 200
maxMessageSize = 200
enableWebsocketCompression = true

[global]
dataFrameSerialization = "arrow"
```

---

## 📈 Scaling Considerations

### Horizontal Scaling
- **Load Balancer**: Nginx or AWS ALB
- **Multiple Instances**: Docker Swarm or Kubernetes
- **Session Storage**: Redis for session management
- **Database**: Read replicas for analytics

### Vertical Scaling
- **CPU**: 2-4 cores for typical workloads
- **Memory**: 8-16GB for large datasets
- **Storage**: SSD for better I/O performance
- **Network**: High bandwidth for data processing

---

## 📚 Maintenance

### Regular Tasks
```bash
# Weekly maintenance script
#!/bin/bash
# maintenance.sh

# Update dependencies
pip install --upgrade -r requirements.txt

# Clear cache
rm -rf __pycache__/
rm -rf .streamlit/cache/

# Backup data
./backup.sh

# Check disk space
df -h

# Restart services
docker-compose restart
```

### Updates and Patches
```bash
# Update application
git pull origin main
pip install --upgrade -r requirements.txt
docker-compose build --no-cache
docker-compose up -d
```

---

## 📞 Support

### Documentation
- **User Guide**: [docs/user-guide/](../user-guide/)
- **Technical Docs**: [docs/technical/](../technical/)
- **API Reference**: [docs/api/](../api/)

### Getting Help
- **Issues**: Create GitHub issue
- **Discussions**: GitHub discussions
- **Email**: support@beverlyknits.com

---

*This deployment guide is regularly updated. Check for the latest version before deployment.*