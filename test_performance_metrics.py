#!/usr/bin/env python3
"""
Test script to verify performance metrics are working correctly.
This script is designed to run inside the Docker container.
"""

import sys
import os
sys.path.append('/app')

from app.database import engine
from sqlmodel import Session, select
from app.models import PlayerAnalysisDetailed
import json

def test_performance_metrics(username="Affan_khan123"):
    """Test that performance metrics are now calculated correctly."""
    
    with Session(engine) as session:
        stmt = select(PlayerAnalysisDetailed).where(PlayerAnalysisDetailed.username == username)
        player_analysis = session.exec(stmt).first()
        
        if not player_analysis:
            print(f"❌ No analysis found for player {username}")
            return False
            
        print(f"=== PERFORMANCE METRICS TEST for {username} ===")
        
        print(f"performance object: {player_analysis.performance}")
        
        if not player_analysis.performance:
            print("❌ Performance object is None")
            return False
            
        trend_acpl = player_analysis.performance.get('trend_acpl')
        trend_match_rate = player_analysis.performance.get('trend_match_rate')
        roi_curve = player_analysis.performance.get('roi_curve')
        
        print(f"trend_acpl: {trend_acpl}")
        print(f"trend_match_rate: {trend_match_rate}")
        print(f"roi_curve: {roi_curve}")
        
        success = True
        
        if trend_acpl is None:
            print("❌ trend_acpl is still null")
            success = False
        else:
            print("✅ trend_acpl has a calculated value")
            
        if trend_match_rate is None:
            print("❌ trend_match_rate is still null")
            success = False
        else:
            print("✅ trend_match_rate has a calculated value")
            
        if roi_curve is None:
            print("❌ roi_curve is still null")
            success = False
        else:
            print("✅ roi_curve has a calculated value")
            
        return success

if __name__ == "__main__":
    success = test_performance_metrics()
    if success:
        print("\n🎉 Performance metrics test PASSED!")
        exit(0)
    else:
        print("\n⚠️ Performance metrics test FAILED!")
        exit(1)
