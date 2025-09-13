#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

def fix_specific_math_issues():
    """Corrige problemas específicos detectados"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    metrics_file = base_dir / "docs/algorithms/metrics.md"

    if not metrics_file.exists():
        print("Archivo metrics.md no encontrado")
        return

    print("Corrigiendo problemas específicos en metrics.md...")

    try:
        # Leer el archivo
        with open(metrics_file, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()

        original_content = content

        # Correcciones específicas para las líneas problemáticas detectadas
        corrections = [
            ('Z-score = (Residual - �) / �', 'Z-score = (Residual - μ) / σ'),
            ('ò(t) = � + � � ��(t-1) + � � ò(t-1)', 'σ²(t) = μ + α × ε(t-1) + β × σ²(t-1)'),
            ('UCL = � + 3�/n', 'UCL = μ + 3σ/√n'),
            ('LCL = � - 3�/n', 'LCL = μ - 3σ/√n'),
            ('Cp = (USL - LSL) / (6�)', 'Cp = (USL - LSL) / (6σ)'),
            ('Cpk = min((USL - �)/3�, (� - LSL)/3�)', 'Cpk = min((USL - μ)/(3σ), (μ - LSL)/(3σ))'),

            # Caracteres problemáticos individuales
            ('�', 'μ'),  # Reemplazar � restantes por mu
            ('ò', 'σ'),  # Reemplazar ò por sigma cuando sea apropiado
        ]

        fixes_applied = 0

        for bad_text, good_text in corrections:
            if bad_text in content:
                content = content.replace(bad_text, good_text)
                fixes_applied += 1
                print(f"  Corregido: {bad_text[:30]}... -> {good_text[:30]}...")

        # Guardar si hubo cambios
        if content != original_content:
            with open(metrics_file, 'w', encoding='utf-8', newline='\n') as f:
                f.write(content)
            print(f"\nArchivo corregido con {fixes_applied} cambios")
        else:
            print("\nNo se encontraron problemas para corregir")

        # Verificar resultado
        verify_result(metrics_file)

    except Exception as e:
        print(f"Error: {e}")

def verify_result(file_path):
    """Verifica el resultado de las correcciones"""

    print("\nVerificando resultado...")

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        lines = content.split('\n')

        # Buscar líneas que aún tengan problemas
        problem_lines = []
        for i, line in enumerate(lines, 1):
            # Buscar caracteres problemáticos específicos
            if any(char in line for char in ['�', 'Â', 'Ã']):
                problem_lines.append((i, line.strip()))

        if problem_lines:
            print(f"Aún quedan {len(problem_lines)} líneas con problemas:")
            for line_num, line_text in problem_lines[:3]:
                print(f"  Línea {line_num}: {line_text}")
            if len(problem_lines) > 3:
                print(f"  ... y {len(problem_lines) - 3} más")
        else:
            print("¡Éxito! No se detectaron más problemas de encoding")

        # Verificar que tenemos símbolos matemáticos
        math_symbols_count = content.count('μ') + content.count('σ')
        if math_symbols_count > 0:
            print(f"Símbolos matemáticos correctos encontrados: {math_symbols_count}")

    except Exception as e:
        print(f"Error verificando: {e}")

if __name__ == "__main__":
    fix_specific_math_issues()