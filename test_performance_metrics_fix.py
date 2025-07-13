#!/usr/bin/env python3
"""
Test script to verify performance metrics fix works correctly.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_performance_metrics_fix():
    """Test that performance metrics are calculated correctly."""
    print("🧪 Testing performance metrics fix...")
    
    try:
        from app.analysis.longitudinal import compute_trends, aggregate_longitudinal_features, roi_per_game
        
        base_date = datetime.now() - timedelta(days=100)
        test_data = []
        
        for i in range(65):
            game_date = base_date + timedelta(days=i*1.5)
            test_data.append({
                "game_id": f"game_{i}",
                "created_at": game_date,
                "date": game_date,  # Required for compute_trends
                "acpl": 5000 + (i * 10) + np.random.normal(0, 100),
                "match_rate": 0.3 + (i * 0.001) + np.random.normal(0, 0.05),
                "ipr": -300 + (i * 2) + np.random.normal(0, 50),
                "white_elo": 1500,
                "black_elo": 1500,
                "result": np.random.choice(['win', 'loss', 'draw']),
                "player_color": np.random.choice(['white', 'black']),
            })
        
        games_df = pd.DataFrame(test_data)
        games_df['roi'] = roi_per_game(games_df)  # Required for compute_trends
        
        print(f"✅ Created test DataFrame with {len(games_df)} games")
        print(f"✅ DataFrame columns: {list(games_df.columns)}")
        
        trend_feats = compute_trends(games_df)
        print(f"✅ compute_trends result: {trend_feats}")
        
        expected_fields = ['trend_acpl', 'trend_match_rate', 'roi_curve']
        all_good = True
        
        for field in expected_fields:
            value = trend_feats.get(field)
            if value is None:
                print(f"❌ {field} is null")
                all_good = False
            else:
                print(f"✅ {field}: {value} (type: {type(value)})")
        
        long_features = aggregate_longitudinal_features(games_df, None)
        print(f"✅ aggregate_longitudinal_features completed")
        
        conflicting_fields = ['trend_acpl', 'trend_match_rate', 'roi_curve']
        for field in conflicting_fields:
            if field in long_features:
                print(f"⚠️  WARNING: {field} found in aggregate_longitudinal_features - this could cause conflicts")
        
        long_features.update({"performance": trend_feats})
        final_performance = long_features.get("performance", {})
        
        print(f"✅ Final performance object: {final_performance}")
        
        for field in expected_fields:
            value = final_performance.get(field)
            if value is None:
                print(f"❌ Final {field} is null")
                all_good = False
            else:
                print(f"✅ Final {field}: {value}")
        
        if all_good:
            print(f"\n🎉 Performance metrics fix test PASSED!")
            return True
        else:
            print(f"\n💥 Performance metrics fix test FAILED!")
            return False
        
    except Exception as e:
        print(f"❌ Performance metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting performance metrics fix test...\n")
    
    if test_performance_metrics_fix():
        print(f"\n🎉 Performance metrics fix verification successful!")
        sys.exit(0)
    else:
        print(f"\n💥 Performance metrics fix verification failed!")
        sys.exit(1)
