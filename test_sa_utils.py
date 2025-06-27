#!/usr/bin/env python3
"""
Test script for SQLAlchemy utility functions.
Tests sa_to_dict and pretty_print_sa with existing models.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from app.database import engine
from app.models import Player, Game, MoveAnalysis
from app.utils import sa_to_dict, pretty_print_sa
from sqlmodel import Session

def test_sa_utils():
    """Test the SQLAlchemy utility functions with existing models."""
    
    with Session(engine) as session:
        player = session.query(Player).first()
        if player:
            print("=== Testing with Player object ===")
            print("sa_to_dict result:")
            player_dict = sa_to_dict(player)
            print(f"Keys: {list(player_dict.keys())}")
            print("\npretty_print_sa result:")
            pretty_print_sa(player)
            print("\n" + "="*50 + "\n")
        
        game = session.query(Game).first()
        if game:
            print("=== Testing with Game object ===")
            print("sa_to_dict result:")
            game_dict = sa_to_dict(game)
            print(f"Keys: {list(game_dict.keys())}")
            print("\npretty_print_sa result:")
            pretty_print_sa(game)
            print("\n" + "="*50 + "\n")
        
        if not player and not game:
            print("No data found in database. Creating a simple test...")
            from app.models import PlayerStatus
            test_player = Player(username="test_user", status=PlayerStatus.pending)
            print("=== Testing with new Player object (not saved) ===")
            print("sa_to_dict result:")
            test_dict = sa_to_dict(test_player)
            print(f"Keys: {list(test_dict.keys())}")
            print("\npretty_print_sa result:")
            pretty_print_sa(test_player)

if __name__ == "__main__":
    test_sa_utils()
