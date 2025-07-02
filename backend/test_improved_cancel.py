#!/usr/bin/env python3
import requests
import json
import time

def test_improved_cancel():
    base_url = "http://localhost:8000"
    username = "test_cancel_player"
    
    print("=== TEST IMPROVED CANCEL ===")
    
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
    print(f"\n2. Esperando 5 segundos para que el análisis comience...")
    time.sleep(5)
    
    # 3. Verificar estado
    print(f"\n3. Verificando estado del jugador...")
    try:
        resp = requests.get(f"{base_url}/players/{username}")
        print(f"GET /players/{username} - Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Player status: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error getting status: {resp.text}")
            return
    except Exception as e:
        print(f"Error getting status: {e}")
        return
    
    # 4. Intentar cancelar
    print(f"\n4. Intentando cancelar análisis...")
    try:
        resp = requests.post(f"{base_url}/players/{username}/stop")
        print(f"POST /players/{username}/stop - Status: {resp.status_code}")
        print(f"Response: {resp.text}")
        
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"Cancel response: {json.dumps(data, indent=2, ensure_ascii=False)}")
            except:
                print("Response is not JSON")
    except Exception as e:
        print(f"Error canceling: {e}")
    
    # 5. Verificar estado después de cancelar
    print(f"\n5. Verificando estado después de cancelar...")
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
    test_improved_cancel() 