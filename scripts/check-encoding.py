#!/usr/bin/env python3
"""
Script para verificar y corregir el encoding de archivos de documentación
"""

import os
import sys
from pathlib import Path

def detect_encoding(file_path):
    """Detecta el encoding de un archivo usando diferentes métodos"""

    # Método 1: Intentar UTF-8 directamente
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read()
        return 'utf-8'
    except UnicodeDecodeError:
        pass

    # Método 2: Intentar encodings comunes
    encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252', 'ascii']

    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return encoding
        except UnicodeDecodeError:
            continue

    # Método 3: Usar chardet si está disponible
    try:
        import chardet
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        detected = chardet.detect(raw_data)
        return detected.get('encoding', 'unknown')
    except ImportError:
        pass

    return 'unknown'

def check_and_fix_encoding():
    """Verifica y corrige encoding de archivos .md"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    docs_dir = base_dir / "docs"

    if not docs_dir.exists():
        print(f"❌ Directorio docs no encontrado: {docs_dir}")
        return

    files_processed = 0
    files_converted = 0
    files_with_issues = []

    print("🔍 Verificando encoding de archivos de documentación...")
    print("=" * 60)

    # Buscar todos los archivos .md
    md_files = list(docs_dir.rglob("*.md"))

    if not md_files:
        print("❌ No se encontraron archivos .md")
        return

    for md_file in md_files:
        try:
            current_encoding = detect_encoding(md_file)
            relative_path = md_file.relative_to(base_dir)

            print(f"📄 {relative_path}: {current_encoding}")

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

                        print(f"✅ Convertido: {current_encoding} → utf-8")
                        files_converted += 1

                    except Exception as e:
                        print(f"❌ Error convirtiendo {relative_path}: {e}")
                        files_with_issues.append(str(relative_path))
                else:
                    print(f"⚠️  No se pudo determinar encoding de {relative_path}")
                    files_with_issues.append(str(relative_path))
            else:
                print(f"✅ Ya está en UTF-8")

            files_processed += 1

        except Exception as e:
            print(f"❌ Error procesando {md_file}: {e}")
            files_with_issues.append(str(md_file))

    # También verificar archivos principales
    main_files = [
        base_dir / "README.md",
        base_dir / "mkdocs.yml",
    ]

    print("\n🔍 Verificando archivos principales...")
    print("-" * 40)

    for main_file in main_files:
        if main_file.exists():
            try:
                current_encoding = detect_encoding(main_file)
                relative_path = main_file.relative_to(base_dir)

                print(f"📄 {relative_path}: {current_encoding}")

                if current_encoding.lower() not in ['utf-8', 'utf-8-sig'] and current_encoding != 'unknown':
                    try:
                        with open(main_file, 'r', encoding=current_encoding) as f:
                            content = f.read()

                        with open(main_file, 'w', encoding='utf-8', newline='\n') as f:
                            f.write(content)

                        print(f"✅ Convertido: {current_encoding} → utf-8")
                        files_converted += 1
                    except Exception as e:
                        print(f"❌ Error convirtiendo {relative_path}: {e}")

                files_processed += 1

            except Exception as e:
                print(f"❌ Error procesando {main_file}: {e}")

    print("\n" + "=" * 60)
    print(f"📊 RESUMEN:")
    print(f"   📁 Archivos procesados: {files_processed}")
    print(f"   🔄 Archivos convertidos: {files_converted}")
    print(f"   ⚠️  Archivos con problemas: {len(files_with_issues)}")

    if files_with_issues:
        print(f"\n⚠️  Archivos que requieren atención manual:")
        for file_path in files_with_issues:
            print(f"   - {file_path}")

    if files_converted > 0:
        print(f"\n✅ Se convirtieron {files_converted} archivos a UTF-8")
    else:
        print(f"\n✅ Todos los archivos ya están en UTF-8")

def install_chardet():
    """Intenta instalar chardet para mejor detección"""
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "chardet"])
        print("✅ chardet instalado para mejor detección de encoding")
        return True
    except:
        print("⚠️  No se pudo instalar chardet, usando detección básica")
        return False

if __name__ == "__main__":
    # Intentar instalar chardet para mejor detección
    try:
        import chardet
    except ImportError:
        print("📦 Instalando chardet para mejor detección de encoding...")
        install_chardet()

    check_and_fix_encoding()