#!/usr/bin/env python3
"""
Runner para toda la suite de pruebas exhaustivas.
Ejecuta todas las pruebas por capas y genera reporte completo.
"""
import sys
import asyncio
import subprocess
from datetime import datetime
from typing import List, Dict, Any

# Agregar path para imports
sys.path.append('.')


class TestResult:
    """Resultado de ejecución de un test."""
    def __init__(self, name: str, success: bool, duration: float, output: str = ""):
        self.name = name
        self.success = success
        self.duration = duration
        self.output = output


class ExhaustiveTestRunner:
    """Runner completo para todas las pruebas exhaustivas."""

    def __init__(self):
        self.results: List[TestResult] = []
        self.start_time = None
        self.end_time = None

    def run_test_file(self, test_file: str, description: str) -> TestResult:
        """Ejecuta un archivo de test y captura el resultado."""
        print(f"\n🚀 Running {description}...")
        print(f"   File: {test_file}")

        start_time = datetime.now()

        try:
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minutos máximo por test
            )

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            success = result.returncode == 0
            output = result.stdout + result.stderr

            if success:
                print(f"   ✅ PASSED in {duration:.2f}s")
            else:
                print(f"   ❌ FAILED in {duration:.2f}s")
                print(f"   Error output: {result.stderr[:200]}...")

            return TestResult(description, success, duration, output)

        except subprocess.TimeoutExpired:
            duration = 300.0
            print(f"   ⏰ TIMEOUT after {duration}s")
            return TestResult(description, False, duration, "Test timeout")

        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            print(f"   💥 ERROR in {duration:.2f}s: {e}")
            return TestResult(description, False, duration, str(e))

    def run_all_tests(self) -> Dict[str, Any]:
        """Ejecuta toda la suite de pruebas exhaustivas."""
        self.start_time = datetime.now()

        print("🧪 STARTING EXHAUSTIVE TEST SUITE")
        print("=" * 60)
        print(f"Start time: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        # Definir todos los tests a ejecutar
        tests = [
            ("tests_exhaustive/test_value_objects_direct.py", "Value Objects Layer"),
            ("tests_exhaustive/test_domain_services.py", "Domain Services Layer"),
            ("tests_exhaustive/test_application_layer.py", "Application Layer (CQRS)"),
            ("tests_exhaustive/test_infrastructure.py", "Infrastructure Layer"),
            ("tests_exhaustive/test_end_to_end.py", "End-to-End Integration"),
        ]

        # Ejecutar cada test
        for test_file, description in tests:
            result = self.run_test_file(test_file, description)
            self.results.append(result)

        self.end_time = datetime.now()

        # Generar reporte
        return self.generate_report()

    def generate_report(self) -> Dict[str, Any]:
        """Genera reporte completo de resultados."""
        total_duration = (self.end_time - self.start_time).total_seconds()
        passed_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]

        success_rate = len(passed_tests) / len(self.results) * 100 if self.results else 0

        report = {
            "summary": {
                "total_tests": len(self.results),
                "passed": len(passed_tests),
                "failed": len(failed_tests),
                "success_rate": success_rate,
                "total_duration": total_duration,
                "overall_success": len(failed_tests) == 0
            },
            "results": self.results,
            "start_time": self.start_time,
            "end_time": self.end_time
        }

        return report

    def print_final_report(self, report: Dict[str, Any]):
        """Imprime reporte final formateado."""
        print("\n" + "=" * 60)
        print("🎯 EXHAUSTIVE TEST SUITE RESULTS")
        print("=" * 60)

        summary = report["summary"]

        print(f"⏱️  Total Duration: {summary['total_duration']:.2f} seconds")
        print(f"📊 Tests Executed: {summary['total_tests']}")
        print(f"✅ Passed: {summary['passed']}")
        print(f"❌ Failed: {summary['failed']}")
        print(f"📈 Success Rate: {summary['success_rate']:.1f}%")

        print("\n📋 DETAILED RESULTS:")
        print("-" * 60)

        for result in self.results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"{status} | {result.name:<30} | {result.duration:>6.2f}s")

        print("\n🏗️ ARCHITECTURE VALIDATION:")
        print("-" * 60)

        layer_results = {
            "Value Objects": any("Value Objects" in r.name for r in self.results if r.success),
            "Domain Services": any("Domain Services" in r.name for r in self.results if r.success),
            "Application Layer": any("Application Layer" in r.name for r in self.results if r.success),
            "Infrastructure": any("Infrastructure" in r.name for r in self.results if r.success),
            "End-to-End": any("End-to-End" in r.name for r in self.results if r.success),
        }

        for layer, passed in layer_results.items():
            status = "✅" if passed else "❌"
            print(f"{status} {layer:<20} - {'VALIDATED' if passed else 'FAILED'}")

        print("\n🎯 CLEAN ARCHITECTURE COMPLIANCE:")
        print("-" * 60)

        compliance_checks = [
            ("Domain Layer Isolation", layer_results["Value Objects"] and layer_results["Domain Services"]),
            ("Application Layer CQRS", layer_results["Application Layer"]),
            ("Infrastructure Separation", layer_results["Infrastructure"]),
            ("End-to-End Integration", layer_results["End-to-End"]),
            ("Overall System Health", summary["overall_success"])
        ]

        for check_name, passed in compliance_checks:
            status = "✅" if passed else "❌"
            print(f"{status} {check_name}")

        print("\n" + "=" * 60)

        if summary["overall_success"]:
            print("🎉 ALL TESTS PASSED! ARCHITECTURE IS SOLID!")
            print("🚀 The refactored Chess Analyzer is ready for deployment!")
        else:
            print("⚠️  SOME TESTS FAILED - REVIEW REQUIRED")
            print("🔧 Check failed tests and fix issues before deployment")

        print("=" * 60)

        return summary["overall_success"]


async def main():
    """Función principal del test runner."""
    print("🏁 Chess Player Analyzer - Exhaustive Test Suite")
    print("🏗️  Testing Clean Architecture Implementation")
    print("📦 Validating Domain-Driven Design with CQRS")

    runner = ExhaustiveTestRunner()

    try:
        report = runner.run_all_tests()
        success = runner.print_final_report(report)

        if success:
            print("\n🎊 CONGRATULATIONS!")
            print("Your Chess Player Analyzer refactor is complete and validated!")
            print("All layers of the clean architecture are working correctly.")
            sys.exit(0)
        else:
            print("\n⚠️  TESTING INCOMPLETE")
            print("Some tests failed. Please review and fix issues.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\n⏹️  Test suite interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Test runner failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())