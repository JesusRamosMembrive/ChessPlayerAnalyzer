#!/usr/bin/env python3
import requests
import json
import time
import subprocess
import sys

def test_real_cancel():
    base_url = "http://localhost:8000"
    username = "Hikaru"  # Jugador real de chess.com
    
    print("=== TEST REAL CANCEL ===")
    
    # 1. Crear un jugador y iniciar análisis
    print(f"\n1. Creando jugador {username} e iniciando análisis...")
    try:
        resp = requests.post(f"{base_url}/players/{username}")
        print(f"POST /players/{username} - Status: {resp.status_code}")
        if resp.status_code == 202:
            data = resp.json()
            print(f"Player created: {json.dumps(data, indent=2, ensure_ascii=False)}")
            task_id = data.get('task_id')
        else:
            print(f"Error creating player: {resp.text}")
            return
    except Exception as e:
        print(f"Error creating player: {e}")
        return
    
    # 2. Esperar un poco para que el análisis comience
    print(f"\n2. Esperando 10 segundos para que el análisis comience...")
    time.sleep(10)
    
    # 3. Verificar estado antes de cancelar
    print(f"\n3. Verificando estado antes de cancelar...")
    try:
        resp = requests.get(f"{base_url}/players/{username}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Player status before cancel: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error getting status: {resp.text}")
            return
    except Exception as e:
        print(f"Error getting status: {e}")
        return
    
    # 4. Intentar cancelar con el CLI
    print(f"\n4. Intentando cancelar con CLI...")
    try:
        result = subprocess.run([
            sys.executable, "player_analyze_cli.py", "cancel", username
        ], capture_output=True, text=True, cwd=".")
        
        print(f"CLI exit code: {result.returncode}")
        print(f"CLI stdout: {result.stdout}")
        print(f"CLI stderr: {result.stderr}")
        
    except Exception as e:
        print(f"Error running CLI: {e}")
    
    # 5. Esperar y verificar estado después de cancelar
    print(f"\n5. Esperando 15 segundos y verificando estado...")
    time.sleep(15)
    
    try:
        resp = requests.get(f"{base_url}/players/{username}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Player status after cancel: {json.dumps(data, indent=2, ensure_ascii=False)}")
            
            # Verificar si el análisis realmente se detuvo
            if data.get('status') == 'ready' and data.get('error') == 'Analysis stopped by user':
                print("✅ Análisis cancelado exitosamente")
            elif data.get('status') == 'error':
                print("⚠️ El análisis falló por error")
            else:
                print("⚠️ El análisis puede no haberse detenido completamente")
        else:
            print(f"Error getting status: {resp.text}")
    except Exception as e:
        print(f"Error getting status: {e}")

if __name__ == "__main__":
    test_real_cancel() 