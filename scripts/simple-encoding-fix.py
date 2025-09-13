#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script simple para corregir caracteres mal codificados
"""

import os
from pathlib import Path

def fix_common_encoding_issues():
    """Corrige los problemas de codificacion mas comunes"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    docs_dir = base_dir / "docs"

    # Solo los reemplazos más comunes y seguros
    fixes = {
        'Ã¡': 'á',
        'Ã©': 'é',
        'Ã­': 'í',
        'Ã³': 'ó',
        'Ãº': 'ú',
        'Ã±': 'ñ',
        'Ã¼': 'ü',
        'Â¿': '¿',
        'Â¡': '¡',
        'anÃ¡lisis': 'análisis',
        'configuraciÃ³n': 'configuración',
        'documentaciÃ³n': 'documentación',
        'implementaciÃ³n': 'implementación',
        'optimizaciÃ³n': 'optimización',
        'mÃ©todos': 'métodos',
        'automÃ¡tico': 'automático',
        'bÃ¡sico': 'básico',
        'tÃ©cnico': 'técnico',
        'mÃ¡s': 'más',
        'tambiÃ©n': 'también',
        'diseÃ±o': 'diseño'
    }

    # Buscar archivos .md
    md_files = list(docs_dir.rglob("*.md"))
    md_files.append(base_dir / "README.md")
    md_files.append(base_dir / "mkdocs.yml")

    # Filtrar archivos existentes
    md_files = [f for f in md_files if f.exists()]

    files_fixed = 0
    total_fixes = 0

    print("Corrigiendo problemas de codificacion comunes...")
    print("=" * 50)

    for file_path in md_files:
        try:
            # Leer archivo
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                content = f.read()

            original_content = content
            file_fixes = 0

            # Aplicar correcciones
            for bad, good in fixes.items():
                if bad in content:
                    count = content.count(bad)
                    content = content.replace(bad, good)
                    file_fixes += count
                    total_fixes += count

            # Guardar si cambió
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8', newline='\n') as f:
                    f.write(content)

                rel_path = file_path.relative_to(base_dir)
                print(f"CORREGIDO: {rel_path} ({file_fixes} correcciones)")
                files_fixed += 1

        except Exception as e:
            print(f"ERROR: {file_path} - {e}")

    print("\n" + "=" * 50)
    print(f"Archivos corregidos: {files_fixed}")
    print(f"Total correcciones: {total_fixes}")

    return files_fixed

def check_remaining_issues():
    """Busca problemas restantes"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    docs_dir = base_dir / "docs"

    print("\nBuscando problemas restantes...")
    print("-" * 30)

    # Archivos importantes a revisar
    check_files = [
        "docs/algorithms/metrics.md",
        "docs/api/endpoints.md",
        "docs/architecture/overview.md",
        "mkdocs.yml"
    ]

    issues_found = 0

    for file_name in check_files:
        file_path = base_dir / file_name
        if file_path.exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Buscar patrones problemáticos
                problems = []

                if 'Ã' in content:
                    problems.append("caracteres con Ã")
                if 'â€' in content:
                    problems.append("comillas mal codificadas")
                if 'â†' in content:
                    problems.append("flechas mal codificadas")

                if problems:
                    print(f"  {file_name}: {', '.join(problems)}")
                    issues_found += 1
                else:
                    print(f"  {file_name}: OK")

            except Exception as e:
                print(f"  {file_name}: ERROR - {e}")

    if issues_found == 0:
        print("  No se encontraron problemas adicionales!")

    return issues_found

if __name__ == "__main__":
    fixed = fix_common_encoding_issues()
    remaining = check_remaining_issues()

    print("\n" + "=" * 50)
    if fixed > 0:
        print("COMPLETADO: Se corrigieron problemas de codificacion")

    if remaining == 0:
        print("EXITO: No quedan problemas de codificacion detectados")
        print("Ya puedes ejecutar: mkdocs serve")
    else:
        print(f"ATENCION: Quedan {remaining} archivos con posibles problemas")
        print("Revisa manualmente los archivos marcados arriba")