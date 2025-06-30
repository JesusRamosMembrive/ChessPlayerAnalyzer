#!/usr/bin/env python3
"""
Test script for JSON export functionality.
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, '/app')

from app.database import engine
from app.models import Player, GameAnalysisDetailed, PlayerAnalysisDetailed
from app.celery_app import export_analysis_to_json
from sqlmodel import Session

def test_json_export():
    """Test the JSON export functionality with existing data."""
    
    print("Testing JSON export functionality...")
    
    try:
        from app.models import PlayerStatus
        test_player = Player(username="test_user", status=PlayerStatus.pending)
        result = export_analysis_to_json(test_player, "test_user", "test")
        print(f"✅ Test export successful: {result}")
        
        if result and Path(result).exists():
            print(f"✅ JSON file created successfully at: {result}")
            with open(result, 'r') as f:
                content = f.read()
                print(f"✅ File content preview (first 200 chars): {content[:200]}...")
        else:
            print("❌ JSON file was not created")
            
    except Exception as e:
        print(f"❌ Test export failed: {e}")
    
    try:
        with Session(engine) as session:
            game_analysis = session.query(GameAnalysisDetailed).first()
            if game_analysis:
                print("\nTesting GameAnalysisDetailed export...")
                result = export_analysis_to_json(game_analysis, "test_user", "game")
                print(f"Game analysis exported to: {result}")
            else:
                print("No GameAnalysisDetailed records found in database")
            
            player_analysis = session.query(PlayerAnalysisDetailed).first()
            if player_analysis:
                print("\nTesting PlayerAnalysisDetailed export...")
                result = export_analysis_to_json(player_analysis, "test_user", "player")
                print(f"Player analysis exported to: {result}")
            else:
                print("No PlayerAnalysisDetailed records found in database")
                
    except Exception as e:
        print(f"❌ Database test failed: {e}")

if __name__ == "__main__":
    test_json_export()
