#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para detectar y corregir caracteres mal codificados en documentacion
"""

import os
import re
from pathlib import Path

def fix_encoding_issues():
    """Detecta y corrige caracteres mal codificados"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    docs_dir = base_dir / "docs"

    # Mapeo de caracteres mal codificados comunes
    encoding_fixes = {
        # Problemas comunes latin-1 -> UTF-8
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã¼': 'ü',
        'Â¿': '¿',
        'Â¡': '¡',
        'Â°': '°',
        'Â±': '±',
        'Â·': '·',
        'Âª': 'ª',
        'Âº': 'º',

        # Palabras completas problemáticas comunes
        'Ã¡lisis': 'álisis',
        'anÃ¡lisis': 'análisis',
        'configuraciÃ³n': 'configuración',
        'documentaciÃ³n': 'documentación',
        'implementaciÃ³n': 'implementación',
        'optimizaciÃ³n': 'optimización',
        'mÃ©todos': 'métodos',
        'fÃ³rmulas': 'fórmulas',
        'estadÃ­sticas': 'estadísticas',
        'automÃ¡tico': 'automático',
        'bÃ¡sico': 'básico',
        'tÃ©cnico': 'técnico',
        'prÃ¡ctico': 'práctico',
        'dinÃ¡mico': 'dinámico',
        'grÃ¡fico': 'gráfico',
        'histÃ³rico': 'histórico',
        'lÃ³gico': 'lógico',
        'mÃ¡quina': 'máquina',
        'nÃºmero': 'número',
        'pÃºblico': 'público',
        'rÃ¡pido': 'rápido',
        'sÃ­mbolo': 'símbolo',
        'tÃ­pico': 'típico',
        'Ãºnico': 'único',
        'Ãºtil': 'útil',
        'mÃ¡s': 'más',
        'tambiÃ©n': 'también',
        'despuÃ©s': 'después',
        'ademÃ¡s': 'además',
        'seÃ±al': 'señal',
        'diseÃ±o': 'diseño',
        'tamaÃ±o': 'tamaño',
        'aÃ±o': 'año',
        'espaÃ±ol': 'español',

        # Patrones de emojis mal codificados
        'âœ…': '✅',
        'â❌': '❌',
        'âš ': '⚠',
        'ðŸŽ¯': '🎯',
        'ðŸ"Š': '📊',
        'ðŸš€': '🚀',
        'ðŸ"': '🔍',
        'ðŸ›¡ï¸': '🛡️',
        'ðŸ"§': '🔧',
        'ðŸ"š': '📚',
        'âª': '⭐',
        'âšª': '⚡',

        # Flechas mal codificadas
        'â†'': '→',
        'â†': '←',
        'â†'': '↑',
        'â†"': '↓',

        # Patrones específicos detectados
        'Ã§': 'ç',
        'Ã¨': 'è',
        'Ã ': 'à',
        'Ã¢': 'â',
        'Ã´': 'ô',
        'Ã®': 'î',
        'Ã»': 'û',
        'Ã«': 'ë',
        'Ã¯': 'ï',
        'Ã¹': 'ù',
        'Ã½': 'ý',
        'Ã¸': 'ø',
        'Ã¦': 'æ',
        'Ã¤': 'ä',
        'Ã¶': 'ö',
        'Ã…': 'Å',
        'Ã†': 'Æ',
        'Ã˜': 'Ø'
    }

    # Buscar archivos de documentación
    file_patterns = ["*.md", "*.yml", "*.yaml"]
    all_files = []

    for pattern in file_patterns:
        all_files.extend(docs_dir.rglob(pattern))

    # Agregar archivos principales
    main_files = [
        base_dir / "README.md",
        base_dir / "mkdocs.yml"
    ]
    all_files.extend([f for f in main_files if f.exists()])

    files_fixed = 0
    total_replacements = 0

    print("Detectando y corrigiendo caracteres mal codificados...")
    print("=" * 60)

    for file_path in all_files:
        try:
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            original_content = content
            file_replacements = 0

            # Aplicar correcciones
            for bad_char, good_char in encoding_fixes.items():
                if bad_char in content:
                    count = content.count(bad_char)
                    content = content.replace(bad_char, good_char)
                    file_replacements += count
                    total_replacements += count

            # Guardar si hubo cambios
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)

                relative_path = file_path.relative_to(base_dir)
                print(f"CORREGIDO: {relative_path} ({file_replacements} reemplazos)")
                files_fixed += 1

        except Exception as e:
            print(f"ERROR procesando {file_path}: {e}")

    print("\n" + "=" * 60)
    print(f"RESUMEN:")
    print(f"  Archivos corregidos: {files_fixed}")
    print(f"  Total de reemplazos: {total_replacements}")

    if files_fixed > 0:
        print(f"\nSe corrigieron {files_fixed} archivos con problemas de codificacion")
    else:
        print("\nNo se encontraron problemas de codificacion")

    return files_fixed > 0

def show_sample_fixes():
    """Muestra ejemplos de los archivos más problemáticos"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")

    # Archivos que suelen tener más problemas
    problem_files = [
        "docs/algorithms/metrics.md",
        "docs/api/endpoints.md",
        "docs/architecture/overview.md"
    ]

    print("\nRevisando archivos principales...")
    print("-" * 40)

    for file_path in problem_files:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    # Leer las primeras líneas
                    lines = f.readlines()[:10]
                    sample = ''.join(lines)

                # Buscar patrones problemáticos
                has_issues = False
                if 'Ã' in sample or 'â' in sample:
                    has_issues = True

                status = "PROBLEMAS DETECTADOS" if has_issues else "OK"
                print(f"  {file_path}: {status}")

            except Exception as e:
                print(f"  {file_path}: ERROR - {e}")

if __name__ == "__main__":
    fixed = fix_encoding_issues()
    show_sample_fixes()

    if fixed:
        print("\n" + "=" * 60)
        print("RECOMENDACION: Revisa manualmente los archivos corregidos")
        print("y ejecuta 'mkdocs serve' para verificar que todo funciona bien")