#!/usr/bin/env python3
"""
Test script to verify NaN handling fix in longitudinal calculations.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

def test_nan_handling():
    """Test that longitudinal calculations don't produce NaN values."""
    print("🧪 Testing NaN handling in longitudinal calculations...")
    
    try:
        from app.analysis.longitudinal import aggregate_longitudinal_features
        
        base_date = datetime.now() - timedelta(days=100)
        test_data = []
        
        for i in range(10):
            game_date = base_date + timedelta(days=i*5)
            test_data.append({
                "game_id": f"game_{i}",
                "created_at": game_date,
                "date": game_date,
                "acpl": 50 + (i * 2),
                "match_rate": 0.3 + (i * 0.01),
                "ipr": 1200 + (i * 10),
            })
        
        df = pd.DataFrame(test_data)
        
        from app.analysis.longitudinal import roi_per_game
        df['roi'] = roi_per_game(df)
        
        features = aggregate_longitudinal_features(df, None)
        
        def check_for_nan(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    check_for_nan(value, f"{path}.{key}" if path else key)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    check_for_nan(item, f"{path}[{i}]")
            elif isinstance(obj, float):
                if math.isnan(obj) or math.isinf(obj):
                    raise ValueError(f"Found NaN/inf at {path}: {obj}")
        
        check_for_nan(features)
        
        json_str = json.dumps(features, default=str)
        print(f"✅ JSON serialization successful, length: {len(json_str)}")
        
        empty_df = pd.DataFrame()
        empty_features = aggregate_longitudinal_features(empty_df, None)
        check_for_nan(empty_features)
        json.dumps(empty_features, default=str)
        print(f"✅ Empty DataFrame test passed")
        
        single_df = df.head(1)
        single_features = aggregate_longitudinal_features(single_df, None)
        check_for_nan(single_features)
        json.dumps(single_features, default=str)
        print(f"✅ Single row test passed")
        
        print(f"✅ All NaN handling tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ NaN handling test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Starting NaN handling verification test...\n")
    
    if test_nan_handling():
        print(f"\n🎉 NaN handling fix verification successful!")
        sys.exit(0)
    else:
        print(f"\n💥 NaN handling fix verification failed!")
        sys.exit(1)
