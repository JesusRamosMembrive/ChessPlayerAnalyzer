#!/usr/bin/env python3
"""
Quick status check for refactor phases.

This provides a fast overview of which phases are working
without triggering database connections.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_import(module_name: str, description: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        print(f"✅ {description}")
        return True
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False


def check_file_content(file_path: Path, pattern: str, description: str) -> bool:
    """Check if a file contains a specific pattern."""
    try:
        if not file_path.exists():
            print(f"❌ {description} - File not found")
            return False

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if pattern in content:
            print(f"✅ {description}")
            return True
        else:
            print(f"❌ {description} - Pattern not found")
            return False
    except Exception as e:
        print(f"❌ {description} - Error: {e}")
        return False


def main():
    """Quick status check."""
    print("🚀 Refactor Status Check")
    print("=" * 50)

    total_checks = 0
    passed_checks = 0

    # FASE 1A checks
    print("\n📦 FASE 1A - Factory Patterns:")
    checks = [
        (lambda: check_file_content(
            project_root / "app" / "factories.py",
            "def create_app()",
            "Factory functions exist"
        )),
        (lambda: check_file_content(
            project_root / "app" / "main.py",
            "from app.factories import create_app",
            "main.py uses factories"
        ))
    ]

    for check in checks:
        total_checks += 1
        if check():
            passed_checks += 1

    # FASE 1B checks
    print("\n🔄 FASE 1B - RedisService:")
    checks = [
        (lambda: check_file_content(
            project_root / "app" / "infrastructure" / "redis_service.py",
            "class RedisService:",
            "RedisService class exists"
        )),
        (lambda: check_file_content(
            project_root / "app" / "utils.py",
            "from app.infrastructure.redis_service import",
            "utils.py uses RedisService"
        ))
    ]

    for check in checks:
        total_checks += 1
        if check():
            passed_checks += 1

    # FASE 1C checks
    print("\n🔐 FASE 1C - AnalysisLockService:")
    checks = [
        (lambda: check_file_content(
            project_root / "app" / "services" / "analysis_lock.py",
            "class AnalysisLockService:",
            "AnalysisLockService class exists"
        )),
        (lambda: check_file_content(
            project_root / "app" / "main.py",
            "from app.services.analysis_lock import",
            "main.py uses AnalysisLockService"
        )),
        (lambda: check_file_content(
            project_root / "app" / "api" / "v1" / "endpoints" / "players.py",
            "check_analysis_preconditions",
            "players.py uses unified preconditions"
        ))
    ]

    for check in checks:
        total_checks += 1
        if check():
            passed_checks += 1

    # Test structure checks
    print("\n🧪 Test Structure:")
    test_files = [
        ("tests/refactor/phases/fase1a/test_factories.py", "FASE 1A test exists"),
        ("tests/refactor/phases/fase1b/test_redis_service.py", "FASE 1B test exists"),
        ("tests/refactor/phases/fase1c/test_fase_1c.py", "FASE 1C test exists"),
        ("tests/refactor/fixtures/real_data_set.json", "Test fixtures available")
    ]

    for file_path, description in test_files:
        total_checks += 1
        if (project_root / file_path).exists():
            print(f"✅ {description}")
            passed_checks += 1
        else:
            print(f"❌ {description}")

    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Status: {passed_checks}/{total_checks} checks passed ({passed_checks/total_checks*100:.1f}%)")

    if passed_checks >= total_checks * 0.8:  # 80% pass rate
        print("🎉 Refactor is in good shape!")
        return 0
    else:
        print("⚠️  Refactor needs attention")
        return 1


if __name__ == "__main__":
    sys.exit(main())