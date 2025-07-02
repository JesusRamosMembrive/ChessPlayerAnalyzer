import httpx
import json
import time

def test_active_cancellation():
    """Test cancellation of an actively running analysis"""
    base_url = "http://localhost:8000"
    username = "Affan_khan123"
    
    with httpx.Client(base_url=base_url, timeout=60) as client:
        print("=== Testing Active Analysis Cancellation ===")
        
        # 1. Reset the player to ensure clean state
        player_url = f"/players/{username}"
        reset_url = f"{player_url}/reset"
        
        print(f"\n1. Resetting player state...")
        reset_resp = client.post(reset_url)
        if reset_resp.status_code == 200:
            print("   ✅ Player reset successful")
        else:
            print(f"   ⚠️  Reset failed: {reset_resp.text}")
        
        # 2. Start a new analysis
        print(f"\n2. Starting new analysis for {username}...")
        analyze_resp = client.post(player_url)
        
        if analyze_resp.status_code == 200:
            analyze_data = analyze_resp.json()
            print(f"   ✅ Analysis started")
            print(f"   Task ID: {analyze_data.get('task_id', 'Unknown')}")
            
            # 3. Wait a bit for the analysis to actually start
            print(f"\n3. Waiting 10 seconds for analysis to begin...")
            time.sleep(10)
            
            # 4. Check current status
            status_resp = client.get(player_url)
            if status_resp.status_code == 200:
                status_data = status_resp.json()
                print(f"   Current status: {status_data.get('status')}")
                print(f"   Current progress: {status_data.get('progress', 0)}%")
                print(f"   Games done: {status_data.get('done_games', 0)}/{status_data.get('total_games', '?')}")
                
                # 5. If analysis is running, try to cancel it
                if status_data.get('status') == 'pending':
                    print(f"\n4. Analysis is running, attempting cancellation...")
                    
                    stop_url = f"{player_url}/stop"
                    cancel_resp = client.post(stop_url)
                    
                    if cancel_resp.status_code == 200:
                        cancel_data = cancel_resp.json()
                        print(f"   ✅ Cancel request successful")
                        print(f"   Revoked tasks: {cancel_data.get('revoked_tasks', 0)}")
                        
                        # 6. Monitor for a while to see if tasks actually stop
                        print(f"\n5. Monitoring for 30 seconds to verify tasks stop...")
                        
                        for i in range(6):  # Check every 5 seconds for 30 seconds
                            time.sleep(5)
                            monitor_resp = client.get(player_url)
                            if monitor_resp.status_code == 200:
                                monitor_data = monitor_resp.json()
                                current_progress = monitor_data.get('progress', 0)
                                current_status = monitor_data.get('status')
                                
                                print(f"   Check {i+1}: Status={current_status}, Progress={current_progress}%")
                                
                                if current_status == 'ready':
                                    print("   ✅ Analysis successfully stopped!")
                                    break
                                elif i == 5:  # Last check
                                    print("   ⚠️  Analysis may still be running after 30 seconds")
                                    
                    else:
                        print(f"   ❌ Cancel request failed: {cancel_resp.text}")
                        
                else:
                    print(f"   ⚠️  Analysis not in pending state: {status_data.get('status')}")
                    
            else:
                print(f"   ❌ Could not check status: {status_resp.text}")
                
        else:
            print(f"   ❌ Failed to start analysis: {analyze_resp.text}")

if __name__ == "__main__":
    test_active_cancellation()