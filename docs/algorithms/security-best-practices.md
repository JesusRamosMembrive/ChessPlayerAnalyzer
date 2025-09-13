# Security and Best Practices

**Status**: Active
**Last Updated**: 2025-09-13
**Category**: Security & Compliance

## Overview

This document outlines the comprehensive security architecture, best practices, and compliance measures implemented in ChessPlayerAnalyzer. The application follows defense-in-depth principles with multiple layers of security controls to protect data, prevent abuse, and ensure system integrity.

## Security Architecture

### Multi-Layer Security Model

```
┌─────────────────────────────────────────────────────────────┐
│                    Internet & External APIs                 │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Network Security (CORS, Rate Limiting, HTTPS)    │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: Application Security (Input Validation, Sanitization) │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: Authentication & Authorization (Future)          │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: Data Security (Encryption, Sanitization)         │
├─────────────────────────────────────────────────────────────┤
│  Layer 5: Infrastructure Security (Container, Network Isolation) │
└─────────────────────────────────────────────────────────────┘
```

## Network Security

### 1. CORS (Cross-Origin Resource Sharing)

**Implementation**: `app/main.py:150-154`

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Production should restrict origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

#### Production Recommendations

```python
# Secure CORS configuration for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://your-frontend.vercel.app",
        "https://your-domain.com",
        "http://localhost:3000"  # Development only
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Content-Type", "Authorization", "X-Requested-With"],
    expose_headers=["X-Total-Count", "X-RateLimit-Remaining"],
)
```

### 2. Rate Limiting

**Implementation**: `app/middleware/rate_limiter.py`
**Algorithm**: Sliding window with in-memory storage

#### Configuration

```python
# Environment variables for rate limiting
RATE_LIMIT_MAX_REQUESTS=100    # Requests per window
RATE_LIMIT_WINDOW_SECONDS=60   # Time window in seconds
```

#### Security Features

- **IP-based limiting**: Prevents individual IP abuse
- **Graceful degradation**: Returns HTTP 429 with retry information
- **Configurable thresholds**: Adjustable via environment variables
- **Memory efficient**: Automatic cleanup of expired entries

```python
# Rate limit response headers
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 42
Retry-After: 60
```

### 3. HTTPS and TLS

**Development**: HTTP (container internal communication)
**Production**: HTTPS enforced via reverse proxy

#### TLS Configuration

```yaml
# Production nginx configuration
server {
    listen 443 ssl http2;
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/private.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-ECDSA-AES128-GCM-SHA256:ECDHE-RSA-AES128-GCM-SHA256;

    location / {
        proxy_pass http://backend:8000;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Application Security

### 4. Input Validation and Sanitization

**Implementation**: Multiple validation layers across the application

#### Request Validation

```python
# FastAPI automatic validation via Pydantic
class AnalysisRequest(BaseModel):
    username: str = Field(min_length=3, max_length=50, pattern="^[a-zA-Z0-9_-]+$")
    months: int = Field(ge=1, le=24, default=12)
    analysis_type: Literal["basic", "detailed", "full"] = "basic"
```

#### Data Sanitization

**File**: `app/utils_sanitize.py`

```python
def clean_json_numbers(obj):
    """
    Prevents JSON injection and handles numeric edge cases
    - Converts NaN/Inf to None
    - Handles NumPy types
    - Recursive dict/list cleaning
    """
```

#### SQL Injection Prevention

- **SQLModel/SQLAlchemy ORM**: Parameterized queries only
- **No raw SQL**: Direct SQL queries prohibited
- **Type validation**: Strong typing via Pydantic models

### 5. Error Handling and Information Disclosure

**Implementation**: `app/error_handlers.py`

#### Unified Error Response Format

```python
{
    "status": "error",
    "message": "User-friendly error message",
    "code": 400,
    "timestamp": "2025-09-13T10:30:00Z",
    "errors": [...] // Optional validation details
}
```

#### Security Features

- **No stack traces**: Production errors sanitized
- **Consistent format**: Prevents information leakage
- **Structured logging**: Detailed logs for debugging without client exposure

```python
# Error logging with security context
logger.error(
    "Database error on %s: %s",
    request.url.path,
    exc,
    exc_info=True,
    extra={"client_ip": client_ip, "user_agent": user_agent}
)
```

## Data Security

### 6. Secrets Management

#### Environment Variables

**Configuration**: All sensitive data via environment variables

```bash
# Database credentials
DATABASE_URL=postgresql+psycopg://user:password@host:5432/db
REDIS_URL=redis://redis:6379/0

# External API configuration
STOCKFISH_PATH=/usr/games/stockfish
SYZYGY_PATH=/data/syzygy

# Observability
OTEL_EXPORTER_JAEGER_AGENT_HOST=jaeger
ENABLE_TRACING=true

# Security settings
RATE_LIMIT_MAX_REQUESTS=100
RATE_LIMIT_WINDOW_SECONDS=60
```

#### Production Secrets Management

```yaml
# Docker Swarm secrets (production)
services:
  backend:
    secrets:
      - db_password
      - redis_password
      - api_keys
    environment:
      DATABASE_URL: postgresql+psycopg://chess:${db_password}@postgres/chess
```

### 7. Data Persistence Security

#### Database Security

```python
# Connection security
DATABASE_URL=postgresql+psycopg://chess:chess@postgres:5432/chessdb
```

**Security measures**:
- **Network isolation**: Database accessible only within Docker network
- **Connection pooling**: Prevents connection exhaustion attacks
- **Query logging**: All database operations logged for audit

#### Archive Security

**Local storage**: `archives/` directory with restricted access

```python
# Archive file permissions (production)
archive_dir.mkdir(parents=True, exist_ok=True, mode=0o750)
with out_path.open("w", encoding="utf-8") as fh:
    json.dump(games, fh, ensure_ascii=False, indent=2)
os.chmod(out_path, 0o640)  # Read-write owner, read group
```

### 8. Data Sanitization and Privacy

#### PII Protection

- **No personal data**: Only public chess game data processed
- **Username anonymization**: Optional username hashing for analysis
- **Data retention**: Configurable archive cleanup policies

```python
# Data anonymization (optional)
def anonymize_username(username: str, salt: str) -> str:
    return hashlib.sha256(f"{username}{salt}".encode()).hexdigest()[:8]
```

## Infrastructure Security

### 9. Container Security

#### Multi-stage Dockerfile Security

```dockerfile
# Base image with security updates
FROM python:3.12-slim AS base
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl stockfish && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN groupadd -r chessapp && useradd -r -g chessapp chessapp
USER chessapp

# Minimal runtime
FROM python:3.12-slim AS runtime
COPY --from=base /usr/local /usr/local
COPY --from=base /etc/passwd /etc/passwd
USER chessapp
```

#### Container Runtime Security

```yaml
# Security constraints in docker-compose
services:
  backend:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
```

### 10. Network Isolation

#### Docker Network Security

```yaml
networks:
  backend:
    driver: bridge
    internal: true  # No external access
  frontend:
    driver: bridge

services:
  postgres:
    networks:
      - backend  # Internal only

  backend:
    networks:
      - backend  # Database access
      - frontend # API access
```

#### Port Exposure

```yaml
# Minimal port exposure
ports:
  - "8000:8000"  # API only
  # Database and Redis ports not exposed externally
```

## Monitoring and Observability

### 11. Security Logging

#### Structured Security Logging

**Implementation**: `app/middleware/request_logger.py`

```python
logger.info(
    "%s %s - %s (%.2f ms) from %s",
    request.method,
    request.url.path,
    response.status_code,
    duration_ms,
    client_host,
    extra={
        "trace_id": trace_id,
        "span_id": span_id,
        "user_agent": request.headers.get("user-agent"),
        "referer": request.headers.get("referer")
    }
)
```

#### Security Event Detection

```python
# Suspicious activity patterns
def detect_anomalous_requests(client_ip: str, requests_per_minute: int):
    if requests_per_minute > 120:  # Rate limit: 100/min
        logger.warning(
            "Potential abuse detected from %s: %d req/min",
            client_ip,
            requests_per_minute,
            extra={"security_event": "rate_limit_exceeded"}
        )
```

### 12. Health Monitoring

#### Security-focused Health Checks

```python
@app.get("/health")
async def health_check():
    """Public health endpoint with minimal information disclosure"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": "1.0.0"
    }

@app.get("/health/detailed")
async def detailed_health():
    """Internal health check with security validation"""
    checks = {
        "database": await check_database_connection(),
        "redis": await check_redis_connection(),
        "rate_limiter": check_rate_limiter_health(),
        "external_apis": await check_external_apis()
    }

    # Log security-relevant health issues
    for service, status in checks.items():
        if not status:
            logger.warning(
                "Security-relevant service unavailable: %s",
                service,
                extra={"security_event": "service_unavailable"}
            )

    return {"services": checks}
```

## CI/CD Security

### 13. Automated Security Testing

**Implementation**: `.github/workflows/ci.yml`

#### Security Pipeline

```yaml
jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - name: Dependency vulnerability scan
        run: |
          pip install safety
          safety check -r requirements.txt

      - name: Code security analysis
        run: |
          pip install bandit
          bandit -r app/ -f json -o security-report.json

      - name: Container security scan
        run: |
          docker build -t chess-analyzer .
          docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy image chess-analyzer
```

#### Secret Detection

```yaml
      - name: Secret detection
        uses: trufflesecurity/trufflehog@main
        with:
          path: ./
          base: main
          head: HEAD
```

### 14. Deployment Security

#### Environment Separation

```yaml
# Production environment variables (secure)
environment:
  NODE_ENV: production
  DEBUG: false
  ENABLE_TRACING: false  # Disable in production
  RATE_LIMIT_MAX_REQUESTS: 50  # Stricter limits
```

#### Secure Deployment

```bash
# Production deployment with security
docker-compose -f docker-compose.prod.yml up -d
docker-compose exec backend python -c "from app.init_db import create_default_admin; create_default_admin()"
```

## Compliance and Auditing

### 15. Audit Logging

#### Comprehensive Audit Trail

```python
# Audit log format
{
    "timestamp": "2025-09-13T10:30:00Z",
    "event_type": "player_analysis_started",
    "user_ip": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "resource": "player/magnus_carlsen",
    "action": "analyze",
    "result": "success",
    "duration_ms": 1250,
    "trace_id": "abc123...",
}
```

#### Retention Policy

```python
# Log retention configuration
AUDIT_LOG_RETENTION_DAYS = 90
SECURITY_LOG_RETENTION_DAYS = 365
APPLICATION_LOG_RETENTION_DAYS = 30
```

### 16. Privacy Compliance

#### GDPR Considerations

- **Data minimization**: Only necessary chess game data collected
- **Consent**: Public data analysis (no consent required)
- **Right to erasure**: Player analysis deletion endpoints
- **Data portability**: JSON export functionality

```python
@app.delete("/api/v1/players/{username}/analysis")
async def delete_player_analysis(username: str):
    """GDPR Article 17: Right to Erasure"""
    # Delete all analysis data for username
    # Anonymize logs containing username
    # Clear caches
```

## Security Best Practices

### 17. Development Security Guidelines

#### Secure Coding Standards

1. **Input Validation**: Always validate and sanitize user input
2. **Output Encoding**: Encode data before rendering
3. **Error Handling**: Never expose internal system details
4. **Logging**: Log security events without sensitive data
5. **Dependencies**: Keep dependencies updated and scanned

#### Code Review Security Checklist

- [ ] Input validation implemented for all user inputs
- [ ] No hardcoded credentials or secrets
- [ ] Error messages don't reveal system information
- [ ] Authentication/authorization properly implemented
- [ ] SQL injection prevention via ORM
- [ ] XSS prevention via output encoding
- [ ] CSRF tokens implemented (if stateful)
- [ ] Rate limiting applied to sensitive endpoints
- [ ] Logging includes security-relevant events
- [ ] Dependencies scanned for vulnerabilities

### 18. Production Security Hardening

#### Web Server Configuration

```nginx
# Production nginx security headers
add_header X-Frame-Options "SAMEORIGIN" always;
add_header X-Content-Type-Options "nosniff" always;
add_header X-XSS-Protection "1; mode=block" always;
add_header Referrer-Policy "strict-origin-when-cross-origin" always;
add_header Content-Security-Policy "default-src 'self'" always;
```

#### Database Security

```postgresql
-- Database security configuration
ALTER ROLE chess_user SET default_transaction_isolation = 'read committed';
ALTER ROLE chess_user SET timezone TO 'UTC';
REVOKE ALL PRIVILEGES ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO chess_user;
```

## Incident Response

### 19. Security Incident Procedures

#### Detection and Response

1. **Automated Detection**: Monitoring alerts for anomalous behavior
2. **Incident Classification**: Severity levels (Low/Medium/High/Critical)
3. **Response Team**: Designated security response personnel
4. **Communication**: Internal and external notification procedures
5. **Recovery**: System restoration and lessons learned

#### Security Playbooks

```python
# Automated incident response
def handle_security_incident(incident_type: str, severity: str, details: dict):
    # Log the incident
    security_logger.critical(
        "Security incident: %s (severity: %s)",
        incident_type,
        severity,
        extra={"incident_details": details}
    )

    # Notify response team
    if severity in ["high", "critical"]:
        send_security_alert(incident_type, details)

    # Automatic mitigation
    if incident_type == "rate_limit_breach":
        escalate_rate_limits()
    elif incident_type == "suspicious_requests":
        temporary_ip_block(details["client_ip"])
```

### 20. Backup and Recovery Security

#### Secure Backup Strategy

```bash
# Encrypted database backups
pg_dump --host=postgres --username=chess --format=custom chessdb | \
  gpg --cipher-algo AES256 --compress-algo 2 --symmetric \
  --output backup_$(date +%Y%m%d_%H%M%S).sql.gpg
```

#### Recovery Testing

```python
# Regular security-focused recovery tests
def test_security_recovery():
    # Test backup integrity
    verify_backup_encryption()

    # Test access controls after recovery
    verify_user_permissions()

    # Test security configurations
    verify_security_middleware()
```

## Future Security Enhancements

### 21. Planned Security Improvements

#### Authentication and Authorization

```python
# Future JWT implementation
@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.url.path.startswith("/api/v1/secure/"):
        token = request.headers.get("Authorization")
        if not verify_jwt_token(token):
            return JSONResponse(
                status_code=401,
                content={"detail": "Authentication required"}
            )
    return await call_next(request)
```

#### Advanced Threat Detection

- **Machine Learning**: Anomaly detection for request patterns
- **Behavioral Analysis**: User behavior profiling
- **Threat Intelligence**: Integration with security feeds
- **Real-time Blocking**: Dynamic IP blocking based on threat scores

#### Security Automation

```python
# Automated security responses
class SecurityAutomation:
    async def analyze_request_patterns(self):
        """ML-based anomaly detection"""

    async def update_threat_intelligence(self):
        """Fetch latest security indicators"""

    async def auto_update_security_rules(self):
        """Dynamic security rule updates"""
```

## Security Contact and Reporting

### 22. Security Communication

#### Vulnerability Reporting

- **Email**: security@chessanalyzer.com
- **PGP Key**: Available at /security-pgp-key
- **Response Time**: 48 hours for acknowledgment
- **Disclosure**: Coordinated disclosure preferred

#### Security Updates

- **Security Notifications**: Critical security updates
- **Patch Management**: Regular security patch deployment
- **Version Control**: Security-focused release notes

---

## See Also

- [External Integrations](external-integrations.md) - External API security considerations
- [Deployment Guide](../guides/deployment.md) - Production security configuration
- [Performance Analysis](performance.md) - Security performance impact analysis
- [Troubleshooting Guide](../guides/troubleshooting.md) - Security-related debugging