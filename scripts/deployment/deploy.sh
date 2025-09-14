#!/bin/bash
# deploy.sh - Production deployment script

set -euo pipefail

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"
BACKUP_DIR="./backups"
LOG_DIR="./logs"

# Colors for output
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
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."

    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running. Please start Docker and try again."
        exit 1
    fi

    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "docker-compose is not installed. Please install docker-compose and try again."
        exit 1
    fi

    # Check if environment file exists
    if [[ ! -f "$ENV_FILE" ]]; then
        error "Environment file $ENV_FILE not found. Please copy .env.prod.example to .env.prod and configure it."
        exit 1
    fi

    # Check if nginx config exists
    if [[ ! -f "nginx.prod.conf" ]]; then
        error "nginx.prod.conf not found. Please ensure production configuration is available."
        exit 1
    fi

    success "Prerequisites check passed"
}

# Create necessary directories
create_directories() {
    log "Creating necessary directories..."
    mkdir -p "$BACKUP_DIR" "$LOG_DIR"
    success "Directories created"
}

# Backup database before deployment
backup_database() {
    if [[ "${SKIP_BACKUP:-false}" == "true" ]]; then
        warning "Skipping database backup as requested"
        return
    fi

    log "Creating database backup..."
    BACKUP_FILE="$BACKUP_DIR/db_backup_$(date +%Y%m%d_%H%M%S).sql"

    # Check if database is running
    if docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps postgres | grep -q "Up"; then
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres \
            pg_dump -U chess -d chessdb > "$BACKUP_FILE"
        success "Database backup created: $BACKUP_FILE"
    else
        warning "Database not running, skipping backup"
    fi
}

# Build images
build_images() {
    log "Building production images..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" build \
        --parallel \
        --pull \
        --no-cache
    success "Images built successfully"
}

# Run database migrations
run_migrations() {
    log "Running database migrations..."
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres redis
    sleep 10  # Wait for services to be ready

    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" run --rm migrate
    success "Database migrations completed"
}

# Deploy services
deploy_services() {
    log "Deploying services..."

    # Deploy with zero-downtime strategy
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d \
        --remove-orphans \
        --force-recreate

    success "Services deployed"
}

# Health check
health_check() {
    log "Performing health checks..."

    local max_attempts=30
    local attempt=1
    local health_url="http://localhost/health"

    while [[ $attempt -le $max_attempts ]]; do
        if curl -f -s "$health_url" >/dev/null 2>&1; then
            success "Health check passed"
            return 0
        fi

        warning "Health check attempt $attempt/$max_attempts failed, retrying in 10s..."
        sleep 10
        ((attempt++))
    done

    error "Health check failed after $max_attempts attempts"
    return 1
}

# Show deployment status
show_status() {
    log "Deployment status:"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" ps

    log "Service logs (last 10 lines):"
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" logs --tail=10 backend
}

# Rollback function
rollback() {
    error "Deployment failed, initiating rollback..."

    # Stop current deployment
    docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" down

    # Restore from backup if available
    local latest_backup=$(ls -t "$BACKUP_DIR"/db_backup_*.sql 2>/dev/null | head -n1)
    if [[ -n "$latest_backup" ]]; then
        warning "Restoring database from backup: $latest_backup"
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" up -d postgres
        sleep 10
        docker-compose -f "$COMPOSE_FILE" --env-file "$ENV_FILE" exec -T postgres \
            psql -U chess -d chessdb < "$latest_backup"
    fi

    error "Rollback completed"
    exit 1
}

# Cleanup old images and containers
cleanup() {
    if [[ "${SKIP_CLEANUP:-false}" == "true" ]]; then
        warning "Skipping cleanup as requested"
        return
    fi

    log "Cleaning up old images and containers..."
    docker image prune -f
    docker container prune -f
    success "Cleanup completed"
}

# Main deployment function
main() {
    log "Starting production deployment..."

    # Trap rollback on error
    trap rollback ERR

    check_prerequisites
    create_directories
    backup_database
    build_images
    run_migrations
    deploy_services

    # Disable rollback trap after successful deployment
    trap - ERR

    if health_check; then
        show_status
        cleanup
        success "Deployment completed successfully!"

        log "Access your application at:"
        log "  - Main application: http://localhost"
        log "  - Jaeger UI: http://localhost:16686"
        log "  - Prometheus (if enabled): http://localhost:9090"
        log "  - Grafana (if enabled): http://localhost:3000"

    else
        rollback
    fi
}

# Script usage
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  --skip-backup    Skip database backup"
    echo "  --skip-cleanup   Skip cleanup of old images"
    echo "  --help          Show this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-backup)
            SKIP_BACKUP=true
            shift
            ;;
        --skip-cleanup)
            SKIP_CLEANUP=true
            shift
            ;;
        --help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

# Run main function
main "$@"