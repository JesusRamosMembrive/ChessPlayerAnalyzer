#!/bin/bash
set -e

echo "🧪 Starting Full Flow Test..."

# =============================================================================
# PHASE 1A: Factory Patterns (implemented: 2024-09-24)
# =============================================================================
echo "📋 Testing Phase 1A: Factory Patterns..."
python3 tests/refactor/phases/fase1a/test_factories.py
echo "✅ Phase 1A passed"

# =============================================================================
# PHASE 1B: RedisService (implemented: 2024-09-24)
# =============================================================================
echo "📋 Testing Phase 1B: RedisService..."
python3 tests/refactor/phases/fase1b/test_redis_service.py
echo "✅ Phase 1B passed"

# =============================================================================
# PHASE 1C: AnalysisLockService (implemented: 2024-09-24)
# =============================================================================
echo "📋 Testing Phase 1C: AnalysisLockService..."
python3 tests/refactor/phases/fase1c/test_fase_1c.py
echo "✅ Phase 1C passed"

# =============================================================================
# PHASE 1D: HttpClient (implemented: 2024-09-28)
# =============================================================================
echo "📋 Testing Phase 1D: HttpClient..."
python3 tests/refactor/phases/fase1d/test_http_client.py
echo "✅ Phase 1D passed"

# =============================================================================
# PHASE 1E: Clean Imports (implemented: 2024-09-28)
# =============================================================================
echo "📋 Testing Phase 1E: Clean Imports (Legacy Cleanup)..."
python3 tests/refactor/phases/fase1e/test_clean_imports.py
echo "✅ Phase 1E passed"

# =============================================================================
# CLEANUP
# =============================================================================
echo "🧹 Cleaning up test data..."
find . -name "*.tmp" -delete 2>/dev/null || true
echo "🎉 All tests passed! FASE 1 REFACTOR COMPLETE!"