import httpx
import json

# Test the cancel functionality
def test_cancel():
    base_url = "http://localhost:8000"
    username = "Affan_khan123"
    
    with httpx.Client(base_url=base_url, timeout=30) as client:
        # First check if player exists
        player_url = f"/players/{username}"
        print(f"Checking player at: {base_url}{player_url}")
        
        resp = client.get(player_url)
        print(f"Player check response status: {resp.status_code}")
        print(f"Player check response text: {resp.text}")
        
        if resp.status_code == 200:
            player_data = resp.json()
            print(f"Player data: {json.dumps(player_data, indent=2)}")
            
            # Try to cancel
            stop_url = f"{player_url}/stop"
            print(f"\nTrying to cancel at: {base_url}{stop_url}")
            
            cancel_resp = client.post(stop_url)
            print(f"Cancel response status: {cancel_resp.status_code}")
            print(f"Cancel response text: {cancel_resp.text}")
        else:
            print("Player not found, cannot test cancel")

if __name__ == "__main__":
    test_cancel()