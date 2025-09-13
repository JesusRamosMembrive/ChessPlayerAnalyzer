#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script específico para corregir símbolos matemáticos mal codificados
"""

from pathlib import Path

def fix_math_symbols():
    """Corrige símbolos matemáticos mal codificados"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")

    # Mapeo de símbolos matemáticos mal codificados
    math_fixes = {
        'Âµ': 'μ',  # mu
        'Ã¼': 'μ',  # mu alternativo
        'Ïƒ': 'σ',  # sigma
        'Ã�': 'σ',  # sigma alternativo
        'Â²': '²',  # superíndice 2
        'Â³': '³',  # superíndice 3
        'Â±': '±',  # más/menos
        'â‰¤': '≤', # menor igual
        'â‰¥': '≥', # mayor igual
        'âˆš': '√', # raíz cuadrada
        'âˆ'': '∑', # sumatoria
        'âˆ«': '∫', # integral
        'Î±': 'α',  # alfa
        'Î²': 'β',  # beta
        'Î³': 'γ',  # gamma
        'Î´': 'δ',  # delta
        'Îµ': 'ε',  # epsilon
        'Î¸': 'θ',  # theta
        'Î»': 'λ',  # lambda
        'Ï€': 'π',  # pi
        'Ï': 'ρ',   # rho
        'Ï„': 'τ',  # tau
        'Ï•': 'φ',  # phi
        'Ï‡': 'χ',  # chi
        'Ï‰': 'ω',  # omega

        # Casos específicos encontrados
        'Â�': 'μ',  # mu mal codificado
        'Â�': 'σ',  # sigma mal codificado

        # Patrones detectados en el archivo
        'Â� + Â� Â� Â��(t-1) + Â� Â� Â�(t-1)': 'μ + α × ε(t-1) + β × σ(t-1)',
        'Z-score = (Residual - Â�) / Â�': 'Z-score = (Residual - μ) / σ',
        'UCL = Â� + 3Â�/n': 'UCL = μ + 3σ/√n',
        'LCL = Â� - 3Â�/n': 'LCL = μ - 3σ/√n',
        'Cp = (USL - LSL) / (6Â�)': 'Cp = (USL - LSL) / (6σ)',
        'Cpk = min((USL - Â�)/3Â�, (Â� - LSL)/3Â�)': 'Cpk = min((USL - μ)/3σ, (μ - LSL)/3σ)',
        'Ã²(t) = Â� + Â� Â� Â��(t-1) + Â� Â� Ã²(t-1)': 'σ²(t) = μ + α × ε(t-1) + β × σ²(t-1)'
    }

    # Archivos a corregir
    files_to_fix = [
        "docs/algorithms/metrics.md",
        "docs/algorithms/statistical-models.md",
        "docs/algorithms/performance.md"
    ]

    files_fixed = 0
    total_fixes = 0

    print("Corrigiendo símbolos matemáticos mal codificados...")
    print("=" * 50)

    for file_name in files_to_fix:
        file_path = base_dir / file_name

        if not file_path.exists():
            print(f"SALTANDO: {file_name} (no existe)")
            continue

        try:
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            original_content = content
            file_fixes = 0

            # Aplicar correcciones
            for bad, good in math_fixes.items():
                if bad in content:
                    count = content.count(bad)
                    content = content.replace(bad, good)
                    file_fixes += count
                    total_fixes += count
                    print(f"  {file_name}: {bad} -> {good} ({count} veces)")

            # Guardar si cambió
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)

                print(f"CORREGIDO: {file_name} ({file_fixes} correcciones)")
                files_fixed += 1
            else:
                print(f"OK: {file_name} (sin cambios)")

        except Exception as e:
            print(f"ERROR: {file_name} - {e}")

    print("\n" + "=" * 50)
    print(f"Archivos corregidos: {files_fixed}")
    print(f"Total correcciones: {total_fixes}")

    # Verificar resultado
    verify_fixes()

def verify_fixes():
    """Verifica que las correcciones funcionaron"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    metrics_file = base_dir / "docs/algorithms/metrics.md"

    if not metrics_file.exists():
        return

    print("\nVerificando correcciones...")
    print("-" * 30)

    try:
        with open(metrics_file, 'r', encoding='utf-8') as f:
            content = f.read()

        # Buscar líneas específicas que sabemos que tenían problemas
        lines = content.split('\n')

        problem_lines = []
        for i, line in enumerate(lines, 1):
            # Buscar caracteres problemáticos restantes
            if any(char in line for char in ['Â', 'Ã', 'â€']):
                problem_lines.append(f"Línea {i}: {line.strip()}")

        if problem_lines:
            print("QUEDAN PROBLEMAS:")
            for problem in problem_lines[:5]:  # Mostrar solo los primeros 5
                print(f"  {problem}")
            if len(problem_lines) > 5:
                print(f"  ... y {len(problem_lines) - 5} más")
        else:
            print("ÉXITO: No se detectaron problemas restantes")

        # Verificar que tenemos símbolos matemáticos correctos
        math_symbols = ['μ', 'σ', 'α', 'β', 'π', '≤', '≥']
        found_symbols = [sym for sym in math_symbols if sym in content]

        if found_symbols:
            print(f"Símbolos matemáticos detectados: {', '.join(found_symbols)}")

    except Exception as e:
        print(f"ERROR verificando: {e}")

if __name__ == "__main__":
    fix_math_symbols()