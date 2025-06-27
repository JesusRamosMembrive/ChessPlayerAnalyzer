#!/usr/bin/env python3
"""
Script de prueba para verificar la corrección del cálculo de performance metrics.
"""

import sys
import os

# Agregar el directorio backend al path
backend_path = os.path.join(os.path.dirname(__file__), 'backend')
sys.path.insert(0, backend_path)

from app.analysis.longitudinal import compute_trends, roi_per_game
from app.analysis.engine import ChessAnalysisEngine
from app.database import engine
from sqlmodel import Session, select
from app.models import Game, GameAnalysisDetailed
import pandas as pd
import numpy as np
from datetime import datetime, timezone

def test_compute_trends():
    """Prueba la función compute_trends con datos de ejemplo."""
    
    # Crear un DataFrame de ejemplo con las columnas requeridas
    dates = pd.date_range('2025-01-01', periods=20, freq='D')
    data = {
        'date': dates,
        'acpl': np.random.normal(50, 10, 20),
        'match_rate': np.random.normal(0.6, 0.1, 20),
        'roi': np.random.normal(2000, 100, 20)
    }
    
    df = pd.DataFrame(data)
    print(f"Test DataFrame shape: {df.shape}")
    print(f"Test DataFrame columns: {list(df.columns)}")
    print(f"Test DataFrame head:\n{df.head()}")
    
    # Probar compute_trends
    result = compute_trends(df)
    print(f"compute_trends result: {result}")
    
    # Verificar que los campos están presentes
    expected_fields = ['trend_acpl', 'trend_match_rate', 'roi_curve']
    for field in expected_fields:
        if field in result:
            print(f"✓ Field '{field}' found with value: {result[field]}")
        else:
            print(f"✗ Field '{field}' missing!")

def test_engine_performance_calculation(username="Affan_khan123"):
    """Prueba el cálculo de performance en el engine."""
    
    engine_instance = ChessAnalysisEngine()
    
    with Session(engine) as session:
        # Obtener el DataFrame de partidas
        games_df = engine_instance._get_player_games_with_analysis(username, session)
        
        print(f"Games DataFrame shape: {games_df.shape}")
        print(f"Games DataFrame columns: {list(games_df.columns)}")
        
        if games_df.empty:
            print("❌ Games DataFrame is empty")
            return False
            
        # Verificar columnas requeridas
        required_columns = ['created_at', 'date', 'acpl', 'match_rate', 'roi']
        for col in required_columns:
            if col in games_df.columns:
                print(f"✓ Column '{col}' found")
            else:
                print(f"✗ Column '{col}' missing!")
                return False
        
        # Verificar que hay datos válidos
        print(f"Sample data:")
        print(f"  created_at: {games_df['created_at'].iloc[0]}")
        print(f"  date: {games_df['date'].iloc[0]}")
        print(f"  acpl: {games_df['acpl'].iloc[0]}")
        print(f"  match_rate: {games_df['match_rate'].iloc[0]}")
        print(f"  roi: {games_df['roi'].iloc[0]}")
        
        # Probar compute_trends directamente
        try:
            trend_feats = compute_trends(games_df)
            print(f"compute_trends result: {trend_feats}")
            
            if trend_feats:
                print("✅ compute_trends returned data")
                return True
            else:
                print("❌ compute_trends returned empty dict")
                return False
        except Exception as e:
            print(f"❌ compute_trends failed with error: {e}")
            return False

def test_current_performance_data(username="Affan_khan123"):
    """Verifica los datos de performance actuales en la base de datos."""
    
    with Session(engine) as session:
        stmt = select(GameAnalysisDetailed).where(
            GameAnalysisDetailed.game_id.in_(
                select(Game.id).where(
                    (Game.white_username == username) |
                    (Game.black_username == username)
                )
            )
        )
        analyses = session.exec(stmt).all()
        
        print(f"Found {len(analyses)} game analyses for {username}")
        
        if analyses:
            print(f"Sample analysis:")
            sample = analyses[0]
            print(f"  game_id: {sample.game_id}")
            print(f"  acpl: {sample.acpl}")
            print(f"  match_rate: {sample.match_rate}")
            print(f"  analyzed_at: {sample.analyzed_at}")

if __name__ == "__main__":
    print("=== Testing compute_trends function ===")
    test_compute_trends()
    
    print("\n=== Testing engine performance calculation ===")
    test_engine_performance_calculation()
    
    print("\n=== Testing current performance data ===")
    test_current_performance_data()
    
    print("\n=== Test completed ===") 