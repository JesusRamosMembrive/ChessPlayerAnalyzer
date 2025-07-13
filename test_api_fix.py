#!/usr/bin/env python3
"""
Test script to verify API endpoint works without JSON serialization errors.
"""

import sys
import os
import requests
import json

def test_api_endpoint():
    """Test that the API endpoint returns valid JSON without NaN errors."""
    print("🧪 Testing API endpoint for JSON serialization...")
    
    try:
        url = "http://localhost:8000/metrics/player/Affan_khan123"
        
        print(f"Making request to: {url}")
        response = requests.get(url, timeout=30)
        
        if response.status_code == 200:
            data = response.json()
            
            json_str = json.dumps(data)
            print(f"✅ API response successful, JSON length: {len(json_str)}")
            
            performance = data.get('performance', {})
            print(f"✅ Performance metrics:")
            print(f"  - trend_acpl: {performance.get('trend_acpl')}")
            print(f"  - trend_match_rate: {performance.get('trend_match_rate')}")
            print(f"  - roi_curve: {performance.get('roi_curve')}")
            
            if performance.get('trend_acpl') is None:
                print("⚠️  Warning: trend_acpl is null")
            if performance.get('trend_match_rate') is None:
                print("⚠️  Warning: trend_match_rate is null")
                
            return True
            
        else:
            print(f"❌ API response failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Make sure it's running on localhost:8000")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ JSON decode error: {e}")
        return False
    except Exception as e:
        print(f"❌ API test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting API endpoint test...\n")
    
    if test_api_endpoint():
        print(f"\n🎉 API endpoint test successful!")
        sys.exit(0)
    else:
        print(f"\n💥 API endpoint test failed!")
        sys.exit(1)
