#!/bin/bash
# Blue-Green Deployment Script for Chess Player Analyzer
# Implements zero-downtime deployment with automated rollback

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
COMPOSE_FILE="docker-compose.prod.yml"
ENV_FILE=".env.prod"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

error() {
    echo -e "${RED}❌ $1${NC}"
}

# Default values
ENVIRONMENT="staging"
IMAGE_TAG="latest"
HEALTH_CHECK_TIMEOUT=180
ROLLBACK_ENABLED=true
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--environment)
            ENVIRONMENT="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        --timeout)
            HEALTH_CHECK_TIMEOUT="$2"
            shift 2
            ;;
        --no-rollback)
            ROLLBACK_ENABLED=false
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  -e, --environment ENV    Target environment (staging/production) [default: staging]"
            echo "  -t, --tag TAG           Docker image tag [default: latest]"
            echo "  --timeout SECONDS       Health check timeout [default: 180]"
            echo "  --no-rollback          Disable automatic rollback on failure"
            echo "  --dry-run              Show what would be done without executing"
            echo "  -h, --help             Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(staging|production)$ ]]; then
    error "Environment must be 'staging' or 'production'"
    exit 1
fi

log "Starting Blue-Green Deployment"
log "Environment: $ENVIRONMENT"
log "Image Tag: $IMAGE_TAG"
log "Health Check Timeout: ${HEALTH_CHECK_TIMEOUT}s"
log "Rollback Enabled: $ROLLBACK_ENABLED"
log "Dry Run: $DRY_RUN"

# Change to project root
cd "$PROJECT_ROOT"

# Check if required files exist
if [[ ! -f "$COMPOSE_FILE" ]]; then
    error "Docker compose file not found: $COMPOSE_FILE"
    exit 1
fi

if [[ ! -f "$ENV_FILE.example" ]] && [[ ! -f "$ENV_FILE" ]]; then
    warning "Environment file not found. Using defaults."
fi

# Function to execute commands (respecting dry-run)
execute() {
    local cmd="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        echo "[DRY RUN] Would execute: $cmd"
    else
        eval "$cmd"
    fi
}

# Function to get current active environment
get_active_environment() {
    local active_containers
    active_containers=$(docker-compose -f "$COMPOSE_FILE" ps --services --filter "status=running" 2>/dev/null || echo "")

    if echo "$active_containers" | grep -q "app-blue"; then
        echo "blue"
    elif echo "$active_containers" | grep -q "app-green"; then
        echo "green"
    else
        echo "none"
    fi
}

# Function to get target environment (opposite of current)
get_target_environment() {
    local current="$1"
    case "$current" in
        "blue") echo "green" ;;
        "green") echo "blue" ;;
        *) echo "blue" ;;  # Default to blue if none are running
    esac
}

# Function to check service health
check_health() {
    local service="$1"
    local max_attempts=$(($HEALTH_CHECK_TIMEOUT / 5))
    local attempt=1

    log "Checking health of service: $service"

    while [[ $attempt -le $max_attempts ]]; do
        if docker-compose -f "$COMPOSE_FILE" exec -T "$service" curl -sf http://localhost:8000/health >/dev/null 2>&1; then
            success "Service $service is healthy (attempt $attempt/$max_attempts)"
            return 0
        fi

        log "Health check attempt $attempt/$max_attempts failed, waiting 5 seconds..."
        sleep 5
        ((attempt++))
    done

    error "Service $service failed health check after $max_attempts attempts"
    return 1
}

# Function to run performance validation
validate_performance() {
    local service="$1"
    log "Running performance validation on $service"

    # Run performance regression tests against the new deployment
    local perf_result
    if docker-compose -f "$COMPOSE_FILE" exec -T "$service" python scripts/ci/performance_regression_test.py --output /tmp/perf_validation.json; then
        # Extract performance metrics
        local speedup
        speedup=$(docker-compose -f "$COMPOSE_FILE" exec -T "$service" jq -r '.summary.average_speedup' /tmp/perf_validation.json 2>/dev/null || echo "0")

        if (( $(echo "$speedup >= 2.0" | bc -l 2>/dev/null || echo "0") )); then
            success "Performance validation passed: ${speedup}x speedup maintained"
            return 0
        else
            error "Performance validation failed: ${speedup}x < 2.0x minimum"
            return 1
        fi
    else
        error "Performance validation script failed"
        return 1
    fi
}

# Function to perform rollback
rollback() {
    local current_env="$1"
    local failed_env="$2"

    error "Deployment failed. Initiating rollback..."

    if [[ "$ROLLBACK_ENABLED" == "true" ]] && [[ "$current_env" != "none" ]]; then
        log "Rolling back to $current_env environment"

        # Switch load balancer back to current environment
        execute "docker-compose -f $COMPOSE_FILE up -d nginx-lb"

        # Stop and remove failed environment
        execute "docker-compose -f $COMPOSE_FILE stop app-$failed_env"
        execute "docker-compose -f $COMPOSE_FILE rm -f app-$failed_env"

        # Verify rollback health
        if check_health "app-$current_env"; then
            success "Rollback completed successfully"
        else
            error "Rollback failed - manual intervention required"
            exit 1
        fi
    else
        error "Rollback disabled or no previous environment available"
        error "Manual intervention required"
        exit 1
    fi
}

# Function to update load balancer configuration
update_load_balancer() {
    local target_env="$1"
    log "Updating load balancer to point to $target_env environment"

    # Update nginx configuration to point to target environment
    local nginx_config="nginx.prod.conf"
    local temp_config="/tmp/nginx.temp.conf"

    # Create updated nginx configuration
    sed "s/app-blue:8000/app-$target_env:8000/g; s/app-green:8000/app-$target_env:8000/g" "$nginx_config" > "$temp_config"

    # Update the configuration and reload nginx
    execute "docker cp $temp_config \$(docker-compose -f $COMPOSE_FILE ps -q nginx-lb):/etc/nginx/nginx.conf"
    execute "docker-compose -f $COMPOSE_FILE exec nginx-lb nginx -s reload"

    rm -f "$temp_config"
    success "Load balancer updated"
}

# Main deployment logic
main() {
    log "=== Starting Blue-Green Deployment ==="

    # Determine current and target environments
    current_env=$(get_active_environment)
    target_env=$(get_target_environment "$current_env")

    log "Current environment: $current_env"
    log "Target environment: $target_env"

    # Pre-deployment checks
    log "=== Pre-Deployment Checks ==="

    # Check Docker and Docker Compose
    if ! command -v docker >/dev/null 2>&1; then
        error "Docker is not installed or not in PATH"
        exit 1
    fi

    if ! command -v docker-compose >/dev/null 2>&1; then
        error "Docker Compose is not installed or not in PATH"
        exit 1
    fi

    # Check if images are available
    local image_name="chess-analyzer:$IMAGE_TAG"
    if [[ "$DRY_RUN" == "false" ]] && ! docker image inspect "$image_name" >/dev/null 2>&1; then
        warning "Image $image_name not found locally. Attempting to build..."
        execute "docker-compose -f $COMPOSE_FILE build"
    fi

    success "Pre-deployment checks completed"

    # Deploy to target environment
    log "=== Deploying to Target Environment ==="

    # Set environment variables for target deployment
    export DEPLOY_ENV="$target_env"
    export IMAGE_TAG="$IMAGE_TAG"

    # Start target environment
    execute "docker-compose -f $COMPOSE_FILE up -d app-$target_env db-$target_env redis-$target_env"

    # Wait for services to be ready
    log "Waiting for services to start..."
    sleep 15

    # Health check
    if ! check_health "app-$target_env"; then
        rollback "$current_env" "$target_env"
        exit 1
    fi

    # Performance validation
    if ! validate_performance "app-$target_env"; then
        rollback "$current_env" "$target_env"
        exit 1
    fi

    # Switch traffic to target environment
    log "=== Switching Traffic ==="
    update_load_balancer "$target_env"

    # Final health check
    log "Performing final health check..."
    sleep 10
    if ! check_health "app-$target_env"; then
        rollback "$current_env" "$target_env"
        exit 1
    fi

    # Cleanup old environment
    if [[ "$current_env" != "none" ]]; then
        log "=== Cleaning Up Old Environment ==="
        execute "docker-compose -f $COMPOSE_FILE stop app-$current_env db-$current_env redis-$current_env"
        execute "docker-compose -f $COMPOSE_FILE rm -f app-$current_env db-$current_env redis-$current_env"
        success "Old environment cleaned up"
    fi

    # Deployment summary
    log "=== Deployment Summary ==="
    success "Blue-Green deployment completed successfully!"
    success "Active environment: $target_env"
    success "Image tag: $IMAGE_TAG"
    success "Environment: $ENVIRONMENT"

    # Show running services
    if [[ "$DRY_RUN" == "false" ]]; then
        log "Current running services:"
        docker-compose -f "$COMPOSE_FILE" ps
    fi

    log "=== Deployment Complete ==="
}

# Trap for cleanup on script exit
cleanup() {
    local exit_code=$?
    if [[ $exit_code -ne 0 ]]; then
        error "Deployment script exited with error code $exit_code"
        log "Check the logs above for details"
    fi
}

trap cleanup EXIT

# Run main deployment
main "$@"