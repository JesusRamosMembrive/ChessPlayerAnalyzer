#!/bin/bash
# Setup CI Environment Script
# Prepares the environment for CI/CD pipeline execution

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

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

# Check if running in CI environment
is_ci() {
    [[ "${CI:-false}" == "true" ]] || [[ -n "${GITHUB_ACTIONS:-}" ]] || [[ -n "${JENKINS_URL:-}" ]]
}

# Setup Python environment
setup_python_environment() {
    log "Setting up Python environment..."

    # Check Python version
    python_version=$(python --version 2>&1 | cut -d' ' -f2)
    log "Python version: $python_version"

    # Install dependencies
    pip install --upgrade pip setuptools wheel

    # Install requirements in order
    if [[ -f "requirements-base.txt" ]]; then
        log "Installing base requirements..."
        pip install -r requirements-base.txt
    fi

    if [[ -f "requirements.txt" ]]; then
        log "Installing main requirements..."
        pip install -r requirements.txt
    fi

    if [[ -f "requirements-dev.txt" ]]; then
        log "Installing development requirements..."
        pip install -r requirements-dev.txt
    fi

    if [[ -f "requirements-ml.txt" ]]; then
        log "Installing ML requirements..."
        pip install -r requirements-ml.txt
    fi

    success "Python environment setup complete"
}

# Setup system dependencies
setup_system_dependencies() {
    log "Setting up system dependencies..."

    if is_ci; then
        log "Detected CI environment - installing system packages"

        # Update package list
        if command -v apt-get >/dev/null 2>&1; then
            sudo apt-get update

            # Install Stockfish
            sudo apt-get install -y stockfish

            # Install additional tools
            sudo apt-get install -y curl jq bc

            # Verify Stockfish installation
            if command -v stockfish >/dev/null 2>&1; then
                stockfish_version=$(stockfish --help | head -1 || echo "Unknown")
                success "Stockfish installed: $stockfish_version"
            else
                error "Stockfish installation failed"
                return 1
            fi
        else
            warning "apt-get not available - assuming dependencies are pre-installed"
        fi
    else
        log "Local environment - assuming system dependencies are available"
    fi

    success "System dependencies setup complete"
}

# Setup test environment
setup_test_environment() {
    log "Setting up test environment..."

    # Create test directories
    mkdir -p test_results
    mkdir -p coverage_reports
    mkdir -p performance_reports

    # Setup test database (if needed)
    if [[ -n "${DATABASE_URL:-}" ]]; then
        log "Initializing test database..."
        python -m app.init_db || warning "Database initialization failed (may already exist)"
    fi

    # Verify API health
    if [[ -n "${API_URL:-}" ]]; then
        log "Checking API health..."
        if curl -sf "${API_URL}/health" >/dev/null; then
            success "API is healthy"
        else
            warning "API health check failed - will attempt to start server"
        fi
    fi

    success "Test environment setup complete"
}

# Setup performance monitoring
setup_performance_monitoring() {
    log "Setting up performance monitoring..."

    # Create monitoring directories
    mkdir -p monitoring_data
    mkdir -p benchmark_results

    # Setup baseline performance data
    if [[ -f "scripts/ci/performance_regression_test.py" ]]; then
        log "Performance regression test script available"
        # Make it executable
        chmod +x scripts/ci/performance_regression_test.py
    else
        warning "Performance regression test script not found"
    fi

    # Check if monitoring tools are available
    if command -v prometheus >/dev/null 2>&1; then
        success "Prometheus available"
    else
        log "Prometheus not available (optional for CI)"
    fi

    if command -v grafana-cli >/dev/null 2>&1; then
        success "Grafana CLI available"
    else
        log "Grafana CLI not available (optional for CI)"
    fi

    success "Performance monitoring setup complete"
}

# Setup security scanning tools
setup_security_tools() {
    log "Setting up security scanning tools..."

    # Install security tools if not available
    if ! command -v bandit >/dev/null 2>&1; then
        log "Installing bandit..."
        pip install bandit
    fi

    if ! command -v safety >/dev/null 2>&1; then
        log "Installing safety..."
        pip install safety
    fi

    # Install code quality tools
    if ! command -v ruff >/dev/null 2>&1; then
        log "Installing ruff..."
        pip install ruff
    fi

    if ! command -v black >/dev/null 2>&1; then
        log "Installing black..."
        pip install black
    fi

    if ! command -v isort >/dev/null 2>&1; then
        log "Installing isort..."
        pip install isort
    fi

    if ! command -v mypy >/dev/null 2>&1; then
        log "Installing mypy..."
        pip install mypy
    fi

    success "Security tools setup complete"
}

# Setup Docker environment (if needed)
setup_docker_environment() {
    if [[ "${SETUP_DOCKER:-false}" == "true" ]]; then
        log "Setting up Docker environment..."

        if ! command -v docker >/dev/null 2>&1; then
            error "Docker not available but required"
            return 1
        fi

        if ! command -v docker-compose >/dev/null 2>&1; then
            error "Docker Compose not available but required"
            return 1
        fi

        # Check if Docker daemon is running
        if ! docker info >/dev/null 2>&1; then
            error "Docker daemon not running"
            return 1
        fi

        # Build images if needed
        if [[ "${BUILD_IMAGES:-false}" == "true" ]]; then
            log "Building Docker images..."
            docker-compose -f docker-compose.prod.yml build
        fi

        success "Docker environment setup complete"
    else
        log "Docker setup skipped"
    fi
}

# Generate CI environment report
generate_environment_report() {
    log "Generating environment report..."

    cat > ci_environment_report.txt << EOF
# CI Environment Setup Report
Generated: $(date)

## System Information
- OS: $(uname -s) $(uname -r)
- Architecture: $(uname -m)
- Python: $(python --version)
- Pip: $(pip --version)

## Tool Versions
- Git: $(git --version 2>/dev/null || echo "Not available")
- Docker: $(docker --version 2>/dev/null || echo "Not available")
- Docker Compose: $(docker-compose --version 2>/dev/null || echo "Not available")
- Stockfish: $(stockfish --help 2>/dev/null | head -1 || echo "Not available")

## Security Tools
- Bandit: $(bandit --version 2>/dev/null || echo "Not available")
- Safety: $(safety --version 2>/dev/null || echo "Not available")
- Ruff: $(ruff --version 2>/dev/null || echo "Not available")
- Black: $(black --version 2>/dev/null || echo "Not available")
- isort: $(isort --version 2>/dev/null || echo "Not available")
- MyPy: $(mypy --version 2>/dev/null || echo "Not available")

## Environment Variables
$(env | grep -E "^(CI|GITHUB|DATABASE|REDIS|STOCKFISH|API)" | sort || echo "No relevant environment variables found")

## Package Information
- Installed packages: $(pip list | wc -l) packages
- Requirements files found:
$(find . -name "requirements*.txt" -type f | sed 's/^/  - /')

## Setup Status
- Python environment: ✅ Complete
- System dependencies: ✅ Complete
- Test environment: ✅ Complete
- Security tools: ✅ Complete
- Performance monitoring: ✅ Complete
$(if [[ "${SETUP_DOCKER:-false}" == "true" ]]; then echo "- Docker environment: ✅ Complete"; fi)

Setup completed successfully at $(date)
EOF

    success "Environment report generated: ci_environment_report.txt"
}

# Main setup function
main() {
    log "=== Starting CI Environment Setup ==="
    log "CI Environment: $(if is_ci; then echo "YES"; else echo "NO"; fi)"

    # Run setup steps
    setup_system_dependencies
    setup_python_environment
    setup_test_environment
    setup_security_tools
    setup_performance_monitoring
    setup_docker_environment
    generate_environment_report

    success "=== CI Environment Setup Complete ==="
}

# Error handling
trap 'error "Setup failed at line $LINENO"' ERR

# Run main setup
main "$@"