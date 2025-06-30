#!/usr/bin/env python3
"""
End-to-end test to verify benchmark calculation fix works for the exact scenario
reported by the user (Affan_khan123 with null percentile values).
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_benchmark_end_to_end():
    """Test benchmark calculation with the exact scenario from user's API response."""
    print("🧪 Testing benchmark calculation end-to-end...")
    
    try:
        from app.analysis.benchmark import compute_benchmark
        from app.analysis.openings import aggregate_player_opening_patterns
        from app.analysis.engine import ChessAnalysisEngine
        
        print("📊 Simulating Affan_khan123 data from user's API response...")
        
        games_analyzed = 65
        avg_acpl = 5101.7398788270675
        mean_entropy = 0  # This was the problematic case from user's response
        
        base_date = datetime.now() - timedelta(days=100)
        test_data = []
        
        for i in range(games_analyzed):
            game_date = base_date + timedelta(days=i*1.5)
            test_data.append({
                "game_id": f"game_{i}",
                "created_at": game_date,
                "date": game_date,
                "eco_code": "A00",  # Same opening (explains mean_entropy = 0)
                "acpl": avg_acpl + np.random.normal(0, 100),  # Around the reported avg_acpl
                "match_rate": 0.301 + np.random.normal(0, 0.01),  # Around reported avg_match_rate
                "ipr": -310 + np.random.normal(0, 50),  # Around reported avg_ipr
                "white_elo": 1500,
                "black_elo": 1500,
                "result": np.random.choice(['win', 'loss', 'draw']),
                "player_color": np.random.choice(['white', 'black']),
            })
        
        games_df = pd.DataFrame(test_data)
        print(f"✅ Created test DataFrame with {len(games_df)} games")
        
        opening_feats = aggregate_player_opening_patterns(games_df, [])
        calculated_mean_entropy = opening_feats.get("mean_entropy")
        
        print(f"📈 Opening features calculated:")
        print(f"  - mean_entropy: {calculated_mean_entropy} (expected: ~0)")
        print(f"  - opening_breadth: {opening_feats.get('opening_breadth')}")
        print(f"  - novelty_depth: {opening_feats.get('novelty_depth')}")
        
        test_scenarios = [
            {"name": "No ELO (None)", "player_elo": None, "expected_default": 1600},
            {"name": "Low ELO (800)", "player_elo": 800, "expected_bucket": 800},
            {"name": "Medium ELO (1600)", "player_elo": 1600, "expected_bucket": 1600},
            {"name": "High ELO (2000)", "player_elo": 2000, "expected_bucket": 2000},
            {"name": "Very High ELO (2400)", "player_elo": 2400, "expected_bucket": 2400},
        ]
        
        print(f"\n🔍 Testing benchmark calculation scenarios...")
        
        all_scenarios_passed = True
        
        for scenario in test_scenarios:
            print(f"\n📋 Scenario: {scenario['name']}")
            print(f"  Input: avg_acpl={avg_acpl}, mean_entropy={calculated_mean_entropy}, player_elo={scenario['player_elo']}")
            
            benchmark = compute_benchmark(avg_acpl, calculated_mean_entropy, scenario['player_elo'])
            
            print(f"  Result: {benchmark}")
            
            if not benchmark:
                print(f"  ❌ FAILED: Empty benchmark result")
                all_scenarios_passed = False
                continue
            
            percentile_acpl = benchmark.get('percentile_acpl')
            percentile_entropy = benchmark.get('percentile_entropy')
            
            if percentile_acpl is None:
                print(f"  ❌ FAILED: percentile_acpl is None")
                all_scenarios_passed = False
            else:
                print(f"  ✅ percentile_acpl: {percentile_acpl} (type: {type(percentile_acpl)})")
            
            if percentile_entropy is None:
                print(f"  ❌ FAILED: percentile_entropy is None")
                all_scenarios_passed = False
            else:
                print(f"  ✅ percentile_entropy: {percentile_entropy} (type: {type(percentile_entropy)})")
            
            if percentile_acpl is not None and not (0 <= percentile_acpl <= 100):
                print(f"  ⚠️  WARNING: percentile_acpl out of range: {percentile_acpl}")
            
            if percentile_entropy is not None and not (0 <= percentile_entropy <= 100):
                print(f"  ⚠️  WARNING: percentile_entropy out of range: {percentile_entropy}")
        
        print(f"\n🧪 Testing specific edge cases...")
        
        edge_case_benchmark = compute_benchmark(avg_acpl, 0.0, 1600)
        print(f"Edge case (mean_entropy=0): {edge_case_benchmark}")
        
        if edge_case_benchmark and edge_case_benchmark.get('percentile_entropy') is not None:
            print(f"✅ Edge case handled correctly: percentile_entropy = {edge_case_benchmark.get('percentile_entropy')}")
        else:
            print(f"❌ Edge case failed: percentile_entropy is None")
            all_scenarios_passed = False
        
        nan_benchmark = compute_benchmark(np.nan, np.nan, 1600)
        print(f"NaN case: {nan_benchmark}")
        
        if nan_benchmark and nan_benchmark.get('percentile_acpl') is None and nan_benchmark.get('percentile_entropy') is None:
            print(f"✅ NaN case handled correctly: both percentiles are None")
        else:
            print(f"⚠️  NaN case behavior: {nan_benchmark}")
        
        print(f"\n📊 Test Summary:")
        print(f"  - Games analyzed: {games_analyzed}")
        print(f"  - Average ACPL: {avg_acpl}")
        print(f"  - Mean entropy: {calculated_mean_entropy}")
        print(f"  - All scenarios passed: {all_scenarios_passed}")
        
        if all_scenarios_passed:
            print(f"\n🎉 All benchmark calculation tests PASSED!")
            print(f"🎯 The fix should resolve the null percentile values for Affan_khan123")
            return True
        else:
            print(f"\n💥 Some benchmark calculation tests FAILED!")
            return False
        
    except Exception as e:
        print(f"❌ End-to-end test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting benchmark end-to-end test...\n")
    
    if test_benchmark_end_to_end():
        print(f"\n🎉 Benchmark end-to-end test successful!")
        print(f"✅ The benchmark calculation fix should work for Affan_khan123!")
        sys.exit(0)
    else:
        print(f"\n💥 Benchmark end-to-end test failed!")
        sys.exit(1)
