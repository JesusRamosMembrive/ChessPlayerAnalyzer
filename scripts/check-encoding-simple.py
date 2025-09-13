#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para verificar y corregir el encoding de archivos de documentacion
"""

import os
import sys
from pathlib import Path

def detect_encoding(file_path):
    """Detecta el encoding de un archivo usando diferentes metodos"""

    # Metodo 1: Intentar UTF-8 directamente
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return 'utf-8'
    except UnicodeDecodeError:
        pass

    # Metodo 2: Intentar encodings comunes
    encodings = ['utf-8-sig', 'latin-1', 'cp1252', 'ascii']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return encoding
        except UnicodeDecodeError:
            continue

    return 'unknown'

def check_and_fix_encoding():
    """Verifica y corrige encoding de archivos .md"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    docs_dir = base_dir / "docs"

    if not docs_dir.exists():
        print(f"ERROR: Directorio docs no encontrado: {docs_dir}")
        return

    files_processed = 0
    files_converted = 0
    files_with_issues = []

    print("Verificando encoding de archivos de documentacion...")
    print("=" * 60)

    # Buscar todos los archivos .md
    md_files = list(docs_dir.rglob("*.md"))
    md_files.extend([
        base_dir / "README.md",
        base_dir / "mkdocs.yml"
    ])

    # Filtrar solo archivos existentes
    md_files = [f for f in md_files if f.exists()]

    if not md_files:
        print("ERROR: No se encontraron archivos .md")
        return

    for md_file in md_files:
        try:
            current_encoding = detect_encoding(md_file)
            relative_path = md_file.relative_to(base_dir)

            print(f"Archivo: {relative_path} -> Encoding: {current_encoding}")

            # Si no es UTF-8, convertir
            if current_encoding.lower() not in ['utf-8', 'utf-8-sig']:
                if current_encoding != 'unknown':
                    try:
                        # Leer con encoding detectado
                        with open(md_file, 'r', encoding=current_encoding) as f:
                            content = f.read()

                        # Escribir en UTF-8 sin BOM
                        with open(md_file, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(content)

                        print(f"  -> CONVERTIDO: {current_encoding} -> utf-8")
                        files_converted += 1

                    except Exception as e:
                        print(f"  -> ERROR convirtiendo {relative_path}: {e}")
                        files_with_issues.append(str(relative_path))
                else:
                    print(f"  -> ATENCION: No se pudo determinar encoding")
                    files_with_issues.append(str(relative_path))
            else:
                print(f"  -> OK: Ya esta en UTF-8")

            files_processed += 1

        except Exception as e:
            print(f"ERROR procesando {md_file}: {e}")
            files_with_issues.append(str(md_file))

    print("\n" + "=" * 60)
    print(f"RESUMEN:")
    print(f"  Archivos procesados: {files_processed}")
    print(f"  Archivos convertidos: {files_converted}")
    print(f"  Archivos con problemas: {len(files_with_issues)}")

    if files_with_issues:
        print(f"\nArchivos que requieren atencion manual:")
        for file_path in files_with_issues:
            print(f"  - {file_path}")

    if files_converted > 0:
        print(f"\nSe convirtieron {files_converted} archivos a UTF-8")
    else:
        print(f"\nTodos los archivos ya estan en UTF-8")

if __name__ == "__main__":
    check_and_fix_encoding()