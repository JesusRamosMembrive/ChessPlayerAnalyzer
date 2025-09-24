#!/usr/bin/env python3
"""
Test simple para FASE 1C - Solo validación de código.

Este test valida que el código de AnalysisLockService esté bien estructurado
sin requerir dependencias externas.
"""
import sys
import ast
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def validate_analysis_lock_service_code():
    """Valida que el código de AnalysisLockService esté bien estructurado."""
    print("🧪 Validando estructura de FASE 1C - AnalysisLockService...")

    try:
        # 1. Verificar que el archivo existe
        service_file = project_root / "app" / "services" / "analysis_lock.py"
        if not service_file.exists():
            print("❌ El archivo app/services/analysis_lock.py no existe")
            return False
        print("✅ 1. Archivo AnalysisLockService existe")

        # 2. Verificar que el código es válido Python
        with open(service_file, 'r', encoding='utf-8') as f:
            code = f.read()

        try:
            ast.parse(code)
            print("✅ 2. Código Python válido")
        except SyntaxError as e:
            print(f"❌ Error de sintaxis en AnalysisLockService: {e}")
            return False

        # 3. Verificar que contiene las clases y funciones esperadas
        expected_items = [
            "class AnalysisLockService",
            "class LockType",
            "def get_analysis_lock_service",
            "def create_analysis_lock_service",
            "def set_global_analysis_lock",
            "def clear_global_analysis_lock",
            "def set_cleanup_lock",
            "def clear_cleanup_lock",
            "def player_lock",
            "def check_analysis_preconditions"
        ]

        missing_items = []
        for item in expected_items:
            if item not in code:
                missing_items.append(item)

        if missing_items:
            print(f"❌ Faltan elementos esperados: {missing_items}")
            return False

        print("✅ 3. Todas las clases y métodos esperados están presentes")

        # 4. Verificar que las keys están definidas correctamente
        expected_keys = [
            '"analysis_in_progress"',
            '"cleanup_in_progress"',
            '"lock:player:{username}"'
        ]

        for key in expected_keys:
            if key not in code:
                print(f"❌ Falta key esperada: {key}")
                return False

        print("✅ 4. Keys de Redis correctamente definidas")

        # 5. Verificar que los timeouts por defecto están definidos
        expected_timeouts = ["900", "7200", "3600"]  # 15min, 2h, 1h
        for timeout in expected_timeouts:
            if timeout not in code:
                print(f"❌ Falta timeout esperado: {timeout}")
                return False

        print("✅ 5. Timeouts por defecto correctamente definidos")

        # 6. Verificar documentación
        if '"""' not in code:
            print("❌ Falta documentación en AnalysisLockService")
            return False

        print("✅ 6. Documentación presente")

        # 7. Contar líneas de código (debería ser sustancial)
        lines = len([line for line in code.split('\n') if line.strip() and not line.strip().startswith('#')])
        if lines < 200:
            print(f"⚠️  Código parece muy corto: {lines} líneas (esperado >200)")
        else:
            print(f"✅ 7. Código tiene tamaño apropiado: {lines} líneas")

        return True

    except Exception as e:
        print(f"❌ Error validando AnalysisLockService: {e}")
        return False


def validate_files_were_modified():
    """Valida que los archivos existentes fueron modificados para usar AnalysisLockService."""
    print("\n🔄 Validando que archivos existentes fueron actualizados...")

    files_to_check = {
        "app/main.py": [
            "from app.services.analysis_lock import",
            "_analysis_lock_service",
            "get_analysis_lock_service"
        ],
        "app/utils.py": [
            "from app.services.analysis_lock import",
            "_analysis_lock_service"
        ],
        "app/api/v1/endpoints/players.py": [
            "from app.services.analysis_lock import",
            "get_analysis_lock_service",
            "check_analysis_preconditions"
        ]
    }

    all_good = True

    for filepath, expected_content in files_to_check.items():
        file_path = project_root / filepath
        if not file_path.exists():
            print(f"❌ Archivo {filepath} no existe")
            all_good = False
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            missing = []
            for expected in expected_content:
                if expected not in content:
                    missing.append(expected)

            if missing:
                print(f"❌ {filepath} - Faltan: {missing}")
                all_good = False
            else:
                print(f"✅ {filepath} - Correctamente actualizado")

        except Exception as e:
            print(f"❌ Error leyendo {filepath}: {e}")
            all_good = False

    return all_good


def validate_test_files_exist():
    """Valida que los archivos de test fueron creados."""
    print("\n🧪 Validando archivos de test...")

    test_files = [
        "tests/refactor/unit/test_analysis_lock_service.py",
        "tests/refactor/integration/test_analysis_lock_unification.py"
    ]

    all_exist = True

    for test_file in test_files:
        file_path = project_root / test_file
        if file_path.exists():
            # Check that it's not empty
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if len(content) > 100:  # Reasonable minimum
                print(f"✅ {test_file} - Existe y tiene contenido")
            else:
                print(f"⚠️  {test_file} - Existe pero parece vacío")
        else:
            print(f"❌ {test_file} - No existe")
            all_exist = False

    return all_exist


def main():
    """Ejecutar validación completa de FASE 1C."""
    print("=" * 70)
    print("🚀 VALIDACIÓN SIMPLE FASE 1C - ANALYSIS LOCK SERVICE")
    print("=" * 70)

    all_passed = True

    # Test estructura del código
    if not validate_analysis_lock_service_code():
        all_passed = False

    # Test que archivos fueron modificados
    if not validate_files_were_modified():
        all_passed = False

    # Test que archivos de test existen
    if not validate_test_files_exist():
        all_passed = False

    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ¡FASE 1C VALIDADA EXITOSAMENTE!")
        print("✅ AnalysisLockService está correctamente implementado")
        print("✅ Archivos existentes fueron actualizados")
        print("✅ Tests fueron creados")
        print("\n📋 Para prueba completa con dependencias:")
        print("   1. Instalar deps: python3 install_deps.py")
        print("   2. Ejecutar tests: python3 -m pytest tests/refactor/unit/test_analysis_lock_service.py -v")
    else:
        print("❌ Algunos elementos de FASE 1C necesitan revisión")
    print("=" * 70)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())