#!/usr/bin/env python3
import requests
import json
import time
import subprocess
import sys

def test_cli_cancel():
    base_url = "http://localhost:8000"
    username = "test_cli_player"
    
    print("=== TEST CLI CANCEL ===")
    
    # 1. Crear un jugador y iniciar análisis
    print(f"\n1. Creando jugador {username} e iniciando análisis...")
    try:
        resp = requests.post(f"{base_url}/players/{username}")
        print(f"POST /players/{username} - Status: {resp.status_code}")
        if resp.status_code == 202:
            data = resp.json()
            print(f"Player created: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error creating player: {resp.text}")
            return
    except Exception as e:
        print(f"Error creating player: {e}")
        return
    
    # 2. Esperar un poco para que el análisis comience
    print(f"\n2. Esperando 3 segundos para que el análisis comience...")
    time.sleep(3)
    
    # 3. Probar el CLI de cancelación
    print(f"\n3. Probando CLI de cancelación...")
    try:
        result = subprocess.run([
            sys.executable, "player_analyze_cli.py", "cancel", username
        ], capture_output=True, text=True, cwd=".")
        
        print(f"CLI exit code: {result.returncode}")
        print(f"CLI stdout: {result.stdout}")
        print(f"CLI stderr: {result.stderr}")
        
    except Exception as e:
        print(f"Error running CLI: {e}")
    
    # 4. Verificar estado después de cancelar
    print(f"\n4. Verificando estado después de cancelar...")
    time.sleep(2)
    try:
        resp = requests.get(f"{base_url}/players/{username}")
        print(f"GET /players/{username} - Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Player status after cancel: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error getting status: {resp.text}")
    except Exception as e:
        print(f"Error getting status: {e}")

if __name__ == "__main__":
    test_cli_cancel() 