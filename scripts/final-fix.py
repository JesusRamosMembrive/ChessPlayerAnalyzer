#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path

def final_fix():
    """Corrección final de caracteres problemáticos"""

    base_dir = Path("C:/Users/jesus/Documents/ChessPlayerAnalyzer")
    metrics_file = base_dir / "docs/algorithms/metrics.md"

    if not metrics_file.exists():
        print("Archivo no encontrado")
        return

    try:
        # Leer archivo
        with open(metrics_file, 'rb') as f:
            content_bytes = f.read()

        # Convertir a string, reemplazando caracteres problemáticos
        content = content_bytes.decode('utf-8', errors='replace')

        # Lista de reemplazos específicos basados en lo que vimos
        replacements = [
            # Reemplazos byte por byte de los caracteres problemáticos detectados
            ('¼', 'μ'),  # mu
            ('Ã', 'σ'),  # sigma
            ('±', 'α'),  # alfa
            ('²', 'β'),  # beta
            ('µ', 'ε'),  # epsilon
            ('É', 'μ'),  # mu alternativo

            # Limpiar caracteres de control
            ('\x1a', ''),

            # Corregir fórmulas específicas conocidas
            ('UCL = μ + 3σ/n', 'UCL = μ + 3σ/√n'),
            ('LCL = μ - 3σ/n', 'LCL = μ - 3σ/√n'),
        ]

        # Aplicar reemplazos
        for old, new in replacements:
            content = content.replace(old, new)

        # Escribir de vuelta como UTF-8
        with open(metrics_file, 'w', encoding='utf-8', newline='\n') as f:
            f.write(content)

        print("Archivo corregido con símbolos matemáticos apropiados")

        # Verificación simple
        if 'μ' in content and 'σ' in content:
            print("Verificación: Símbolos matemáticos μ y σ encontrados")
        else:
            print("Advertencia: No se detectaron todos los símbolos esperados")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    final_fix()