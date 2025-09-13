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
        'Ã¡': 'á',  # á mal codificado
        'Ã©': 'é',  # é mal codificado
        'Ã­': 'í',  # í mal codificado
        'Ã³': 'ó',  # ó mal codificado
        'Ãº': 'ú',  # ú mal codificado
        'Ã±': 'ñ',  # ñ mal codificado
        'Ã¼': 'ü',  # ü mal codificado
        'Â¿': '¿',  # ¿ mal codificado
        'Â¡': '¡',  # ¡ mal codificado
        'â€™': ''',  # apostrofe curvo
        'â€œ': '"',  # comilla doble izquierda
        'â€': '"',   # comilla doble derecha
        'â€"': '–',  # guión en (en dash)
        'â€"': '—',  # guión em (em dash)
        'â€¦': '…',  # puntos suspensivos
        'Â°': '°',   # símbolo de grado
        'Â±': '±',   # más/menos
        'Â·': '·',   # punto medio
        'Âª': 'ª',   # ordinal femenino
        'Âº': 'º',   # ordinal masculino
        'â‚¬': '€',  # símbolo del euro
        'â†'': '→',  # flecha derecha
        'â†': '←',   # flecha izquierda
        'â†'': '↑',  # flecha arriba
        'â†"': '↓',  # flecha abajo
        'âœ…': '✅', # check mark
        'â�': '❌',  # cross mark
        'â�': '⚠️', # warning
        'ğŸ': '🎯',  # target emoji patterns
        'ğŸ"Š': '📊', # chart emoji
        'ğŸš€': '🚀', # rocket emoji
        'ğŸ"': '🔍',  # magnifying glass
        'ğŸ›¡ï¸': '🛡️', # shield emoji
        'ğŸ"§': '🔧', # wrench emoji
        'ğŸ"': '📚',  # books emoji
        'â�': '⭐',  # star
        'â�': '⚡',  # lightning

        # Caracteres específicos problemáticos
        'Ã¡lisis': 'análisis',
        'configuraciÃ³n': 'configuración',
        'documentaciÃ³n': 'documentación',
        'implementaciÃ³n': 'implementación',
        'optimizaciÃ³n': 'optimización',
        'anÃ¡lisis': 'análisis',
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
        'niÃ±o': 'niño',
        'espaÃ±ol': 'español'
    }

    # Buscar todos los archivos .md y otros archivos de texto
    file_patterns = ["*.md", "*.yml", "*.yaml", "*.txt", "*.py"]
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
            # Leer archivo como UTF-8
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            original_content = content
            file_replacements = 0

            # Aplicar todas las correcciones
            for bad_char, good_char in encoding_fixes.items():
                if bad_char in content:
                    count = content.count(bad_char)
                    content = content.replace(bad_char, good_char)
                    file_replacements += count
                    total_replacements += count

            # Si hubo cambios, guardar el archivo
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

        # Verificar algunos archivos problemáticos conocidos
        verify_common_issues()
    else:
        print("\nNo se encontraron problemas de codificacion")

def verify_common_issues():
    """Verifica archivos que suelen tener problemas"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")

    # Archivos que suelen tener caracteres especiales
    check_files = [
        "docs/algorithms/metrics.md",
        "docs/api/endpoints.md",
        "docs/architecture/overview.md",
        "docs/guides/development.md"
    ]

    print("\nVerificando archivos que suelen tener problemas...")
    print("-" * 40)

    for file_path in check_files:
        full_path = base_dir / file_path
        if full_path.exists():
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Buscar patrones problemáticos
                issues = []

                # Buscar caracteres raros
                if re.search(r'Ã[¡-ÿ]', content):
                    issues.append("caracteres con Ã mal codificados")

                if re.search(r'â€[™œ"]', content):
                    issues.append("comillas/apostrofes mal codificados")

                if re.search(r'â†[→←↑↓]', content):
                    issues.append("flechas mal codificadas")

                if issues:
                    print(f"  {file_path}: REVISAR - {', '.join(issues)}")
                else:
                    print(f"  {file_path}: OK")

            except Exception as e:
                print(f"  {file_path}: ERROR - {e}")

if __name__ == "__main__":
    fix_encoding_issues()