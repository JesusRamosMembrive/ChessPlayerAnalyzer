#!/usr/bin/env python3
"""
Verification script to check if the calculation fixes are working correctly.
Tests the example player "Affan_khan123" mentioned by the user.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.database import engine
from sqlmodel import Session, select
from app.models import PlayerAnalysisDetailed
import json

def verify_player_fixes(username="Affan_khan123"):
    """Verify that all the calculation fixes are working for the specified player."""
    
    with Session(engine) as session:
        stmt = select(PlayerAnalysisDetailed).where(PlayerAnalysisDetailed.username == username)
        player_analysis = session.exec(stmt).first()
        
        if not player_analysis:
            print(f"❌ No analysis found for player {username}")
            return False
            
        print(f"=== VERIFICATION: Player Analysis Results for {username} ===")
        
        print(f"avg_ipr: {player_analysis.avg_ipr} (should be calculated value, not 0)")
        print(f"roi_std: {player_analysis.roi_std} (should be calculated value, not null)")
        print(f"step_function_detected: {player_analysis.step_function_detected} (should be true/false, not null)")
        print(f"step_function_magnitude: {player_analysis.step_function_magnitude} (should be calculated value, not null)")
        
        print(f"peer_delta_acpl: {player_analysis.peer_delta_acpl} (should be calculated value, not 0)")
        print(f"peer_delta_match: {player_analysis.peer_delta_match} (should be calculated value, not 0)")
        
        print(f"longest_streak: {player_analysis.longest_streak} (should be calculated value, not 0)")
        print(f"first_game_date: {player_analysis.first_game_date} (should be calculated date, not null)")
        print(f"last_game_date: {player_analysis.last_game_date} (should be calculated date, not null)")
        
        print(f"opening_patterns: {player_analysis.opening_patterns} (should have mean_entropy > 0, not -0)")
        if player_analysis.opening_patterns:
            print(f"  mean_entropy: {player_analysis.opening_patterns.get('mean_entropy', 'NOT_FOUND')}")
        print(f"risk_score: {player_analysis.risk_score} (should be calculated value)")
        print(f"risk_factors: {player_analysis.risk_factors} (should be calculated dict)")
        print(f"favorite_openings: {player_analysis.favorite_openings} (should be list of openings, not null)")
        
        print("=== END VERIFICATION ===")
        
        fixes_working = True
        
        if player_analysis.avg_ipr == 0:
            print("❌ avg_ipr still returning 0")
            fixes_working = False
        else:
            print("✅ avg_ipr fix working")
            
        if player_analysis.roi_std is None:
            print("❌ roi_std still returning null")
            fixes_working = False
        else:
            print("✅ roi_std fix working")
            
        if player_analysis.step_function_detected is None:
            print("❌ step_function_detected still returning null")
            fixes_working = False
        else:
            print("✅ step_function_detected fix working")
            
        if player_analysis.peer_delta_acpl == 0:
            print("❌ peer_delta_acpl still returning 0")
            fixes_working = False
        else:
            print("✅ peer_delta_acpl fix working")
            
        if player_analysis.peer_delta_match == 0:
            print("❌ peer_delta_match still returning 0")
            fixes_working = False
        else:
            print("✅ peer_delta_match fix working")
            
        if player_analysis.longest_streak == 0:
            print("❌ longest_streak still returning 0")
            fixes_working = False
        else:
            print("✅ longest_streak fix working")
            
        if player_analysis.first_game_date is None:
            print("❌ first_game_date still returning null")
            fixes_working = False
        else:
            print("✅ first_game_date fix working")
            
        if player_analysis.last_game_date is None:
            print("❌ last_game_date still returning null")
            fixes_working = False
        else:
            print("✅ last_game_date fix working")
            
        if player_analysis.opening_patterns and player_analysis.opening_patterns.get('mean_entropy', 0) <= 0:
            print("❌ mean_entropy still returning 0 or negative")
            fixes_working = False
        else:
            print("✅ mean_entropy fix working")
            
        if player_analysis.risk_score == 0 and not player_analysis.risk_factors:
            print("❌ risk metrics still returning 0/empty")
            fixes_working = False
        else:
            print("✅ risk metrics fix working")
            
        if player_analysis.favorite_openings is None:
            print("❌ favorite_openings still returning null")
            fixes_working = False
        else:
            print("✅ favorite_openings fix working")
            
        print(f"performance: {player_analysis.performance} (should have trend_acpl, trend_match_rate, roi_curve)")
        if player_analysis.performance and player_analysis.performance.get('trend_acpl') is None:
            print("❌ trend_acpl still returning null")
            fixes_working = False
        else:
            print("✅ trend_acpl fix working")
            
        if player_analysis.performance and player_analysis.performance.get('trend_match_rate') is None:
            print("❌ trend_match_rate still returning null")
            fixes_working = False
        else:
            print("✅ trend_match_rate fix working")
            
        if player_analysis.performance and player_analysis.performance.get('roi_curve') is None:
            print("❌ roi_curve still returning null")
            fixes_working = False
        else:
            print("✅ roi_curve fix working")
            
        return fixes_working

if __name__ == "__main__":
    success = verify_player_fixes()
    if success:
        print("\n🎉 All fixes verified successfully!")
    else:
        print("\n⚠️  Some fixes may need additional work")
