# Docker-based ML Model Deployment Pipeline

This directory contains the Docker-based deployment pipeline for the Beverly Knits AI Supply Chain Planner, including ML model training, inference, and production deployment.

## 🚀 Quick Start

### Development Environment
```bash
# Start development environment
./scripts/deploy.sh start development

# Access the application
open http://localhost:8000

# View logs
./scripts/deploy.sh logs development
```

### Production Environment
```bash
# Start production environment
./scripts/deploy.sh start production

# Access the application (with SSL)
open https://localhost

# Monitor services
docker service ls
```

## 📁 Directory Structure

```
docker/
├── Dockerfile                 # Multi-stage Docker build
├── docker-compose.yml         # Main compose configuration
├── docker-compose.dev.yml     # Development configuration
├── docker-compose.prod.yml    # Production configuration
├── nginx.conf                 # Nginx reverse proxy config
├── prometheus.yml             # Prometheus monitoring config
├── scripts/
│   ├── deploy.sh             # Main deployment script
│   ├── train_models.sh       # Model training script
│   └── backup.sh             # Database backup script
└── README.md                 # This file
```

## 🐳 Docker Services

### Application Services
- **app**: Main FastAPI application
- **ml-inference**: ML model inference service
- **ml-training**: Model training service (on-demand)

### Infrastructure Services
- **postgres**: PostgreSQL database
- **redis**: Redis cache and message broker
- **nginx**: Reverse proxy and load balancer

### Monitoring Services
- **prometheus**: Metrics collection
- **grafana**: Visualization dashboards
- **mlflow**: ML experiment tracking

## 🔧 Configuration

### Environment Variables
Create `.env` files for different environments:

```bash
# .env.development
ENVIRONMENT=development
DEBUG=true
ENABLE_AI_INTEGRATION=true
DB_HOST=postgres-dev
DB_PASSWORD=dev_password

# .env.production
ENVIRONMENT=production
DEBUG=false
ENABLE_AI_INTEGRATION=true
DB_HOST=postgres
DB_PASSWORD_FILE=/run/secrets/db_password
```

### Docker Secrets (Production)
```bash
# Create secrets for production
echo "secure_db_password" | docker secret create db_password -
echo "secure_api_key" | docker secret create api_key -
echo "grafana_admin_password" | docker secret create grafana_password -
```

## 🚀 Deployment Commands

### Basic Operations
```bash
# Start services
./scripts/deploy.sh start [environment]

# Stop services
./scripts/deploy.sh stop [environment]

# Restart services
./scripts/deploy.sh restart [environment]

# View logs
./scripts/deploy.sh logs [environment]

# Health check
curl http://localhost:8000/health
```

### Model Training
```bash
# Train all models
./scripts/train_models.sh train development all

# Train specific model
./scripts/train_models.sh train development prophet

# Run full training pipeline
./scripts/train_models.sh pipeline production ensemble live

# Evaluate models
./scripts/train_models.sh evaluate

# Deploy trained models
./scripts/train_models.sh deploy
```

### Data Management
```bash
# Backup data
./scripts/deploy.sh backup

# Restore from backup
./scripts/deploy.sh restore backups/20231201_120000

# Clean up unused resources
./scripts/deploy.sh cleanup
```

## 🏗️ Multi-Stage Docker Build

### Build Stages
1. **base**: Common dependencies and setup
2. **development**: Development tools and hot-reload
3. **production**: Optimized production build
4. **ml-training**: ML training environment
5. **ml-inference**: Optimized inference service

### Building Images
```bash
# Build development image
docker build -t beverly-knits:dev --target development .

# Build production image
docker build -t beverly-knits:prod --target production .

# Build ML training image
docker build -t beverly-knits:ml-training --target ml-training .
```

## 🔄 CI/CD Pipeline

### GitHub Actions Integration
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to production
        run: |
          ./docker/scripts/deploy.sh start production
```

### Automated Model Training
```yaml
# .github/workflows/train-models.yml
name: Train ML Models
on:
  schedule:
    - cron: '0 2 * * *'  # Daily at 2 AM
jobs:
  train:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Train models
        run: |
          ./docker/scripts/train_models.sh pipeline production
```

## 📊 Monitoring and Observability

### Prometheus Metrics
- Application metrics: http://localhost:9090
- Custom ML model metrics
- Supply chain specific metrics
- Infrastructure metrics

### Grafana Dashboards
- Application performance: http://localhost:3000
- ML model performance tracking
- Supply chain KPIs
- Infrastructure monitoring

### MLflow Experiment Tracking
- Model experiments: http://localhost:5000
- Model versioning and deployment
- Parameter and metric tracking

## 🔒 Security

### Production Security Features
- Non-root user containers
- Secret management with Docker secrets
- SSL/TLS encryption
- Network segmentation
- Security headers (HSTS, CSP, etc.)
- Rate limiting

### Development Security
- Isolated development environment
- Separate databases and networks
- Development-only debug modes

## 🎯 Performance Optimization

### Production Optimizations
- Multi-worker FastAPI deployment
- Database connection pooling
- Redis caching
- Nginx load balancing
- Image optimization and caching

### Resource Management
- CPU and memory limits
- Health checks and auto-restart
- Graceful shutdown handling
- Log rotation and management

## 🧪 Testing

### Testing in Docker
```bash
# Run unit tests
docker run --rm beverly-knits:dev python -m pytest tests/

# Run integration tests
docker run --rm --network beverly-knits-network beverly-knits:dev python -m pytest tests/integration/

# Run ML model tests
docker run --rm beverly-knits:ml-training python -m pytest tests/ml/
```

### Load Testing
```bash
# Install load testing tools
pip install locust

# Run load tests
locust -f tests/load/locustfile.py --host http://localhost:8000
```

## 📈 Scaling

### Horizontal Scaling
```bash
# Scale application service
docker service scale beverly-knits_app=5

# Scale ML inference service
docker service scale beverly-knits_ml-inference=3
```

### Database Scaling
- Read replicas for query load distribution
- Connection pooling
- Query optimization

## 🛠️ Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Check port usage
   netstat -tulpn | grep :8000
   
   # Use different ports
   export APP_PORT=8001
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   docker stats
   
   # Increase memory limits in compose file
   ```

3. **Database Connection Issues**
   ```bash
   # Check database logs
   docker logs beverly-knits-postgres
   
   # Test connection
   docker exec -it beverly-knits-postgres psql -U postgres
   ```

### Debugging
```bash
# Enter running container
docker exec -it beverly-knits-app bash

# Check application logs
docker logs -f beverly-knits-app

# Check service health
docker service ps beverly-knits_app
```

## 🔄 Updates and Maintenance

### Rolling Updates
```bash
# Update application
docker service update --image beverly-knits:v1.1.0 beverly-knits_app

# Update with zero downtime
docker service update --update-parallelism 1 --update-delay 10s beverly-knits_app
```

### Maintenance Tasks
```bash
# Update dependencies
docker build --no-cache -t beverly-knits:latest .

# Clean up old images
docker image prune -a

# Backup before updates
./scripts/deploy.sh backup
```

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Docker Compose Documentation](https://docs.docker.com/compose/)
- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [ML Model Deployment Best Practices](https://ml-ops.org/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Test your changes with Docker
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.