#!/bin/bash

# Beverly Knits AI Supply Chain Planner - Deployment Script

set -e

# Configuration
ENVIRONMENT=${1:-development}
VERSION=${2:-latest}
COMPOSE_FILE=""

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

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed. Please install Docker first."
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        error "Docker Compose is not installed. Please install Docker Compose first."
    fi
    
    log "Docker and Docker Compose are available"
}

# Set compose file based on environment
set_compose_file() {
    case $ENVIRONMENT in
        development|dev)
            COMPOSE_FILE="docker-compose.dev.yml"
            log "Using development configuration"
            ;;
        production|prod)
            COMPOSE_FILE="docker-compose.prod.yml"
            log "Using production configuration"
            ;;
        *)
            COMPOSE_FILE="docker-compose.yml"
            log "Using default configuration"
            ;;
    esac
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    
    mkdir -p data/models
    mkdir -p data/training
    mkdir -p logs
    mkdir -p backups
    
    # Create SSL directory for production
    if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
        mkdir -p ssl
        if [[ ! -f ssl/server.crt || ! -f ssl/server.key ]]; then
            warning "SSL certificates not found. Generating self-signed certificates..."
            generate_ssl_certificates
        fi
    fi
    
    success "Directories created successfully"
}

# Generate SSL certificates
generate_ssl_certificates() {
    log "Generating self-signed SSL certificates..."
    
    openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
        -keyout ssl/server.key \
        -out ssl/server.crt \
        -subj "/C=US/ST=State/L=City/O=Beverly Knits/CN=localhost"
    
    success "SSL certificates generated"
}

# Pull Docker images
pull_images() {
    log "Pulling Docker images..."
    docker-compose -f docker/$COMPOSE_FILE pull
    success "Images pulled successfully"
}

# Build application images
build_images() {
    log "Building application images..."
    docker-compose -f docker/$COMPOSE_FILE build --no-cache
    success "Images built successfully"
}

# Start services
start_services() {
    log "Starting services..."
    
    if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
        # For production, use Docker Swarm
        docker stack deploy -c docker/$COMPOSE_FILE beverly-knits
    else
        # For development/testing, use docker-compose
        docker-compose -f docker/$COMPOSE_FILE up -d
    fi
    
    success "Services started successfully"
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
        docker stack rm beverly-knits
    else
        docker-compose -f docker/$COMPOSE_FILE down
    fi
    
    success "Services stopped successfully"
}

# Health check
health_check() {
    log "Performing health check..."
    
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi
        
        log "Health check attempt $attempt/$max_attempts failed, retrying in 10 seconds..."
        sleep 10
        ((attempt++))
    done
    
    error "Health check failed after $max_attempts attempts"
}

# Show logs
show_logs() {
    log "Showing application logs..."
    
    if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
        docker service logs beverly-knits_app
    else
        docker-compose -f docker/$COMPOSE_FILE logs -f
    fi
}

# Clean up
cleanup() {
    log "Cleaning up..."
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful in production)
    if [[ $ENVIRONMENT != "production" && $ENVIRONMENT != "prod" ]]; then
        docker volume prune -f
    fi
    
    success "Cleanup completed"
}

# Backup data
backup_data() {
    log "Backing up data..."
    
    local backup_dir="backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # Backup database
    if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
        docker exec beverly-knits_postgres pg_dump -U postgres beverly_knits_prod > "$backup_dir/database.sql"
    else
        docker exec beverly-knits-postgres-dev pg_dump -U postgres beverly_knits_dev > "$backup_dir/database.sql"
    fi
    
    # Backup models
    cp -r data/models "$backup_dir/"
    
    success "Data backed up to $backup_dir"
}

# Restore data
restore_data() {
    local backup_dir=$1
    
    if [[ -z "$backup_dir" ]]; then
        error "Please specify backup directory"
    fi
    
    if [[ ! -d "$backup_dir" ]]; then
        error "Backup directory does not exist: $backup_dir"
    fi
    
    log "Restoring data from $backup_dir..."
    
    # Restore database
    if [[ -f "$backup_dir/database.sql" ]]; then
        if [[ $ENVIRONMENT == "production" || $ENVIRONMENT == "prod" ]]; then
            docker exec -i beverly-knits_postgres psql -U postgres beverly_knits_prod < "$backup_dir/database.sql"
        else
            docker exec -i beverly-knits-postgres-dev psql -U postgres beverly_knits_dev < "$backup_dir/database.sql"
        fi
    fi
    
    # Restore models
    if [[ -d "$backup_dir/models" ]]; then
        cp -r "$backup_dir/models"/* data/models/
    fi
    
    success "Data restored successfully"
}

# Main execution
main() {
    log "Starting Beverly Knits AI Supply Chain Planner deployment..."
    log "Environment: $ENVIRONMENT"
    log "Version: $VERSION"
    
    case "$1" in
        start)
            check_docker
            set_compose_file
            create_directories
            pull_images
            build_images
            start_services
            health_check
            success "Deployment completed successfully!"
            ;;
        stop)
            set_compose_file
            stop_services
            ;;
        restart)
            set_compose_file
            stop_services
            sleep 5
            start_services
            health_check
            ;;
        logs)
            set_compose_file
            show_logs
            ;;
        backup)
            backup_data
            ;;
        restore)
            restore_data "$2"
            ;;
        cleanup)
            cleanup
            ;;
        *)
            echo "Usage: $0 {start|stop|restart|logs|backup|restore|cleanup} [environment] [version]"
            echo "Environments: development, production"
            echo "Examples:"
            echo "  $0 start development"
            echo "  $0 start production v1.0.0"
            echo "  $0 backup"
            echo "  $0 restore backups/20231201_120000"
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"