#!/bin/bash

# Beverly Knits AI Supply Chain Planner - Model Training Script

set -e

# Configuration
ENVIRONMENT=${1:-development}
MODEL_TYPE=${2:-all}
DATA_SOURCE=${3:-synthetic}

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        error "Docker is not running. Please start Docker first."
    fi
    log "Docker is running"
}

# Train models using Docker
train_models() {
    log "Starting model training..."
    log "Environment: $ENVIRONMENT"
    log "Model type: $MODEL_TYPE"
    log "Data source: $DATA_SOURCE"
    
    # Create training container
    local container_name="beverly-knits-training-$(date +%s)"
    
    # Prepare training command
    local training_cmd=""
    case $MODEL_TYPE in
        arima)
            training_cmd="python train_arima_model.py"
            ;;
        prophet)
            training_cmd="python train_prophet_model.py"
            ;;
        lstm)
            training_cmd="python train_lstm_model.py"
            ;;
        xgboost)
            training_cmd="python train_xgboost_model.py"
            ;;
        ensemble)
            training_cmd="python train_ensemble_models.py"
            ;;
        all)
            training_cmd="python train_all_models.py"
            ;;
        *)
            error "Unknown model type: $MODEL_TYPE"
            ;;
    esac
    
    # Run training container
    docker run --name "$container_name" \
        --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        -e MODEL_TYPE="$MODEL_TYPE" \
        -e DATA_SOURCE="$DATA_SOURCE" \
        --network "beverly-knits-network" \
        beverly-knits-ml-training:latest \
        bash -c "$training_cmd"
    
    success "Model training completed successfully"
}

# Train specific model with custom parameters
train_custom_model() {
    local model_script=$1
    local model_params=$2
    
    log "Training custom model: $model_script"
    
    if [[ ! -f "$model_script" ]]; then
        error "Model script not found: $model_script"
    fi
    
    local container_name="beverly-knits-custom-training-$(date +%s)"
    
    docker run --name "$container_name" \
        --rm \
        -v "$(pwd):/app" \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        --network "beverly-knits-network" \
        beverly-knits-ml-training:latest \
        bash -c "python $model_script $model_params"
    
    success "Custom model training completed"
}

# Evaluate trained models
evaluate_models() {
    log "Evaluating trained models..."
    
    local container_name="beverly-knits-evaluation-$(date +%s)"
    
    docker run --name "$container_name" \
        --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/logs:/app/logs" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        --network "beverly-knits-network" \
        beverly-knits-ml-training:latest \
        bash -c "python evaluate_models.py"
    
    success "Model evaluation completed"
}

# Deploy trained models
deploy_models() {
    log "Deploying trained models..."
    
    # Copy models to inference service
    if [[ $ENVIRONMENT == "production" ]]; then
        # For production, use Docker volumes
        docker run --name "model-deployment-$(date +%s)" \
            --rm \
            -v "beverly-knits_ml_models:/app/models/trained" \
            -v "$(pwd)/models:/app/models/source" \
            alpine:latest \
            cp -r /app/models/source/* /app/models/trained/
    else
        # For development, copy directly
        cp -r models/* data/models/
    fi
    
    # Restart ML inference service
    if [[ $ENVIRONMENT == "production" ]]; then
        docker service update --force beverly-knits_ml-inference
    else
        docker-compose -f docker/docker-compose.dev.yml restart ml-inference
    fi
    
    success "Models deployed successfully"
}

# Monitor training progress
monitor_training() {
    log "Monitoring training progress..."
    
    # Show training logs
    docker logs -f beverly-knits-training 2>/dev/null || true
    
    # Show MLflow UI (if available)
    if curl -f -s http://localhost:5000 > /dev/null 2>&1; then
        log "MLflow UI available at http://localhost:5000"
    fi
}

# Cleanup training artifacts
cleanup_training() {
    log "Cleaning up training artifacts..."
    
    # Remove temporary files
    rm -rf temp_training_*
    
    # Remove old model versions (keep last 5)
    find models/ -name "*.pkl" -o -name "*.h5" -o -name "*.joblib" | \
        sort -t'_' -k2 -nr | \
        tail -n +6 | \
        xargs rm -f
    
    success "Training cleanup completed"
}

# Generate training report
generate_report() {
    log "Generating training report..."
    
    local report_file="reports/training_report_$(date +%Y%m%d_%H%M%S).html"
    mkdir -p reports
    
    docker run --name "report-generation-$(date +%s)" \
        --rm \
        -v "$(pwd)/data:/app/data" \
        -v "$(pwd)/models:/app/models" \
        -v "$(pwd)/reports:/app/reports" \
        -e ENVIRONMENT="$ENVIRONMENT" \
        --network "beverly-knits-network" \
        beverly-knits-ml-training:latest \
        bash -c "python generate_training_report.py --output /app/reports/$(basename $report_file)"
    
    success "Training report generated: $report_file"
}

# Main execution
main() {
    log "Starting Beverly Knits AI model training..."
    
    case "$1" in
        train)
            check_docker
            train_models
            ;;
        custom)
            check_docker
            train_custom_model "$2" "$3"
            ;;
        evaluate)
            check_docker
            evaluate_models
            ;;
        deploy)
            deploy_models
            ;;
        monitor)
            monitor_training
            ;;
        cleanup)
            cleanup_training
            ;;
        report)
            generate_report
            ;;
        pipeline)
            # Full training pipeline
            check_docker
            train_models
            evaluate_models
            deploy_models
            generate_report
            cleanup_training
            success "Full training pipeline completed!"
            ;;
        *)
            echo "Usage: $0 {train|custom|evaluate|deploy|monitor|cleanup|report|pipeline} [environment] [model_type] [data_source]"
            echo ""
            echo "Commands:"
            echo "  train      - Train models"
            echo "  custom     - Train custom model script"
            echo "  evaluate   - Evaluate trained models"
            echo "  deploy     - Deploy models to inference service"
            echo "  monitor    - Monitor training progress"
            echo "  cleanup    - Clean up training artifacts"
            echo "  report     - Generate training report"
            echo "  pipeline   - Run full training pipeline"
            echo ""
            echo "Model types: arima, prophet, lstm, xgboost, ensemble, all"
            echo "Data sources: synthetic, live, historical"
            echo ""
            echo "Examples:"
            echo "  $0 train development all synthetic"
            echo "  $0 custom train_custom_model.py '--epochs 100 --batch_size 32'"
            echo "  $0 pipeline production ensemble live"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"