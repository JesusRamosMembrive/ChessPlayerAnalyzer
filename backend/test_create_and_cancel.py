#!/usr/bin/env python3
import requests
import json

def test_create_and_cancel():
    base_url = "http://localhost:8000"
    username = "test_player_123"
    
    print("=== TEST CREATE AND CANCEL ===")
    
    # 1. Crear un jugador
    print(f"\n1. Creando jugador {username}...")
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
    
    # 2. Verificar estado
    print(f"\n2. Verificando estado del jugador...")
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
    
    # 3. Intentar cancelar
    print(f"\n3. Intentando cancelar análisis...")
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

if __name__ == "__main__":
    test_create_and_cancel() 