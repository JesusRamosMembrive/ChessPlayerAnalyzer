#!/usr/bin/env python3
"""
Test script for JSON export functionality in Docker environment.
"""

import sys
from pathlib import Path

from app.models import Player, PlayerStatus
from app.celery_app import export_analysis_to_json

def test_json_export():
    """Test the JSON export functionality."""
    
    print("Testing JSON export functionality...")
    
    try:
        test_player = Player(username="test_user", status=PlayerStatus.pending)
        result = export_analysis_to_json(test_player, "test_user", "test")
        print(f"✅ Test export successful: {result}")
        
        if result and Path(result).exists():
            print(f"✅ JSON file created successfully at: {result}")
            with open(result, 'r') as f:
                content = f.read()
                print(f"✅ File content preview (first 300 chars): {content[:300]}...")
            return True
        else:
            print("❌ JSON file was not created")
            return False
            
    except Exception as e:
        print(f"❌ Test export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_json_export()
    if success:
        print("\n🎉 JSON export functionality test PASSED!")
    else:
        print("\n❌ JSON export functionality test FAILED!")
        sys.exit(1)
