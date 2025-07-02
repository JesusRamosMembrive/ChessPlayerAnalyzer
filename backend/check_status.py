#!/usr/bin/env python3
import requests
import json

def check_status():
    base_url = "http://localhost:8000"
    username = "test_player_123"
    
    print("=== CHECKING PLAYER STATUS ===")
    
    try:
        resp = requests.get(f"{base_url}/players/{username}")
        print(f"GET /players/{username} - Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"Player status: {json.dumps(data, indent=2, ensure_ascii=False)}")
        else:
            print(f"Error: {resp.text}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_status() 