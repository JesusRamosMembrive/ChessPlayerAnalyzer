#!/usr/bin/env python3
"""
Test script to debug benchmark calculation issues.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_benchmark_calculation():
    """Test benchmark calculation with realistic data."""
    print("🧪 Testing benchmark calculation...")
    
    try:
        from app.analysis.benchmark import compute_benchmark
        from app.analysis.openings import aggregate_player_opening_patterns
        
        avg_acpl = 5101.7398788270675  # From user's response
        
        base_date = datetime.now() - timedelta(days=100)
        test_data = []
        
        for i in range(65):  # 65 games like Affan_khan123
            game_date = base_date + timedelta(days=i*1.5)
            test_data.append({
                "game_id": f"game_{i}",
                "created_at": game_date,
                "date": game_date,
                "eco_code": np.random.choice(['A00', 'B01', 'C20', 'D00', 'E00']),  # Random ECO codes
                "acpl": 5000 + (i * 10) + np.random.normal(0, 100),
                "match_rate": 0.3 + (i * 0.001) + np.random.normal(0, 0.05),
                "ipr": -300 + (i * 2) + np.random.normal(0, 50),
            })
        
        games_df = pd.DataFrame(test_data)
        
        opening_feats = aggregate_player_opening_patterns(games_df, [])
        mean_entropy = opening_feats.get("mean_entropy")
        
        print(f"✅ Test data created:")
        print(f"  - avg_acpl: {avg_acpl}")
        print(f"  - mean_entropy: {mean_entropy}")
        print(f"  - opening_feats: {opening_feats}")
        
        test_elos = [None, 800, 1200, 1600, 2000, 2400]
        
        for player_elo in test_elos:
            print(f"\n🔍 Testing with player_elo: {player_elo}")
            
            benchmark = compute_benchmark(avg_acpl, mean_entropy, player_elo)
            print(f"  Result: {benchmark}")
            
            if benchmark:
                print(f"  ✅ percentile_acpl: {benchmark.get('percentile_acpl')}")
                print(f"  ✅ percentile_entropy: {benchmark.get('percentile_entropy')}")
            else:
                print(f"  ❌ Empty benchmark result")
        
        print(f"\n🧪 Testing edge cases...")
        
        benchmark_none = compute_benchmark(None, None, 1600)
        print(f"  None values result: {benchmark_none}")
        
        benchmark_nan = compute_benchmark(np.nan, np.nan, 1600)
        print(f"  NaN values result: {benchmark_nan}")
        
        benchmark_high = compute_benchmark(5101.74, 0.0, 1600)
        print(f"  High ACPL result: {benchmark_high}")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting benchmark debug test...\n")
    
    if test_benchmark_calculation():
        print(f"\n🎉 Benchmark debug test completed!")
        sys.exit(0)
    else:
        print(f"\n💥 Benchmark debug test failed!")
        sys.exit(1)
