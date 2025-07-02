import httpx
import json
import time

def test_improved_cancel():
    """Test the improved cancellation functionality"""
    base_url = "http://localhost:8000"
    username = "Affan_khan123"
    
    with httpx.Client(base_url=base_url, timeout=30) as client:
        print("=== Testing Improved Cancellation Mechanism ===")
        
        # 1. Check if player exists and current status
        player_url = f"/players/{username}"
        print(f"\n1. Checking player status at: {base_url}{player_url}")
        
        resp = client.get(player_url)
        print(f"   Status: {resp.status_code}")
        
        if resp.status_code == 200:
            player_data = resp.json()
            print(f"   Current status: {player_data.get('status')}")
            print(f"   Progress: {player_data.get('progress', 0)}%")
            print(f"   Task ID: {player_data.get('last_task_id', 'None')}")
            
            # 2. If there's an analysis in progress, try to cancel it
            if player_data.get('status') == 'pending' or player_data.get('progress', 0) > 0:
                print(f"\n2. Analysis in progress detected, attempting to cancel...")
                
                stop_url = f"{player_url}/stop"
                print(f"   Sending cancel request to: {base_url}{stop_url}")
                
                cancel_resp = client.post(stop_url)
                print(f"   Cancel response status: {cancel_resp.status_code}")
                
                if cancel_resp.status_code == 200:
                    cancel_data = cancel_resp.json()
                    print(f"   Cancel response: {json.dumps(cancel_data, indent=4)}")
                    
                    # 3. Wait a bit and check if the status actually changed
                    print(f"\n3. Waiting 5 seconds to verify cancellation...")
                    time.sleep(5)
                    
                    # Check status again
                    verify_resp = client.get(player_url)
                    if verify_resp.status_code == 200:
                        verify_data = verify_resp.json()
                        print(f"   Post-cancel status: {verify_data.get('status')}")
                        print(f"   Post-cancel progress: {verify_data.get('progress', 0)}%")
                        print(f"   Error message: {verify_data.get('error', 'None')}")
                        
                        if verify_data.get('status') == 'ready' and 'stopped by user' in str(verify_data.get('error', '')):
                            print("   ✅ Cancellation appears successful!")
                        else:
                            print("   ⚠️  Cancellation may not have worked properly")
                    else:
                        print(f"   ❌ Could not verify cancellation: {verify_resp.status_code}")
                        
                else:
                    print(f"   ❌ Cancel request failed: {cancel_resp.text}")
                    
            else:
                print(f"\n2. No analysis in progress (status: {player_data.get('status')})")
                print("   Starting a new analysis to test cancellation...")
                
                # Start a new analysis
                analyze_resp = client.post(player_url)
                if analyze_resp.status_code == 200:
                    print("   ✅ Analysis started, waiting 3 seconds...")
                    time.sleep(3)
                    
                    # Now try to cancel
                    print("   Attempting to cancel the new analysis...")
                    stop_url = f"{player_url}/stop"
                    cancel_resp = client.post(stop_url)
                    
                    if cancel_resp.status_code == 200:
                        cancel_data = cancel_resp.json()
                        print(f"   Cancel response: {json.dumps(cancel_data, indent=4)}")
                        
                        # Verify cancellation
                        time.sleep(3)
                        verify_resp = client.get(player_url)
                        if verify_resp.status_code == 200:
                            verify_data = verify_resp.json()
                            print(f"   Final status: {verify_data.get('status')}")
                            if verify_data.get('status') == 'ready':
                                print("   ✅ Test cancellation successful!")
                            else:
                                print("   ⚠️  Test cancellation may not have worked")
                    else:
                        print(f"   ❌ Test cancel failed: {cancel_resp.text}")
                else:
                    print(f"   ❌ Could not start test analysis: {analyze_resp.text}")
                    
        else:
            print(f"   ❌ Player not found or error: {resp.text}")

if __name__ == "__main__":
    test_improved_cancel()