#!/usr/bin/env python3
"""
Test runner for all refactor phases.

This script runs validation tests for all completed phases of the refactor,
providing a comprehensive view of the refactor progress.
"""
import sys
import subprocess
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def run_phase_test(phase_name: str, test_script: str) -> bool:
    """Run a specific phase test."""
    print(f"\n{'='*20} {phase_name} {'='*20}")

    try:
        # Run the test script
        result = subprocess.run(
            [sys.executable, test_script],
            capture_output=True,
            text=True,
            timeout=60
        )

        if result.returncode == 0:
            print(f"✅ {phase_name} passed!")
            if result.stdout:
                print(result.stdout)
            return True
        else:
            print(f"❌ {phase_name} failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"⏰ {phase_name} timed out!")
        return False
    except Exception as e:
        print(f"💥 Error running {phase_name}: {e}")
        return False


def main():
    """Run all phase tests."""
    print("🚀 Chess Player Analyzer - Refactor Phase Validation")
    print("=" * 60)

    phases_dir = Path(__file__).parent

    # Define phases and their test scripts
    phases = [
        ("FASE 1A - Factory Patterns", phases_dir / "fase1a" / "test_factories.py"),
        ("FASE 1B - RedisService", phases_dir / "fase1b" / "test_redis_service.py"),
        ("FASE 1C - AnalysisLockService", phases_dir / "fase1c" / "test_fase_1c.py"),
    ]

    results = []

    for phase_name, test_script in phases:
        if test_script.exists():
            success = run_phase_test(phase_name, str(test_script))
            results.append((phase_name, success))
        else:
            print(f"\n⚠️  {phase_name}: Test script not found ({test_script})")
            results.append((phase_name, False))

    # Summary
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE FASES")
    print("=" * 60)

    passed_count = 0
    for phase_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{phase_name}: {status}")
        if success:
            passed_count += 1

    print(f"\n🎯 Progreso: {passed_count}/{len(results)} fases completadas")

    if passed_count == len(results):
        print("🎉 ¡Todas las fases del refactor están funcionando correctamente!")
        return 0
    else:
        print("💔 Algunas fases necesitan atención")
        return 1


if __name__ == "__main__":
    sys.exit(main())