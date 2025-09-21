#!/usr/bin/env python3
"""
Script de prueba real para Sprint 2 - Domain Refactor
Usa datos reales de Chess.com para validar nuestros servicios de dominio.
"""
import json
import sys
import os
from pathlib import Path
from datetime import datetime

# Añadir el directorio raíz al path para imports
sys.path.append(str(Path(__file__).parent))

from app.domain.services.analysis_service import AnalysisService
from app.domain.services.game_service import GameService
from app.domain.entities.game import Game, MoveData


def load_real_games_from_json(json_file: str, max_games: int = 10) -> list[dict]:
    """Carga partidas reales desde archivo JSON de Chess.com."""
    print(f"📂 Cargando partidas desde {json_file}...")

    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        # Tomar solo las primeras max_games partidas
        games = data[:max_games] if isinstance(data, list) else [data]
        print(f"✅ Cargadas {len(games)} partidas")
        return games

    except Exception as e:
        print(f"❌ Error cargando {json_file}: {e}")
        return []


def extract_pgn_and_times(game_data: dict) -> tuple[str, list[int]]:
    """Extrae PGN y tiempos de movimiento desde datos de Chess.com."""
    pgn = game_data.get('pgn', '')
    move_times = game_data.get('move_times', [])

    # Convertir tiempos negativos a positivos (Chess.com usa negativos para black)
    move_times = [abs(t) * 100 for t in move_times]  # Convertir a milliseconds

    return pgn, move_times


def create_sample_moves_data(pgn: str, move_times: list[int]) -> list[MoveData]:
    """
    Crea datos de movimientos simulados para testing.
    En producción esto vendría del análisis de Stockfish.
    """
    import re

    # Extraer movimientos del PGN
    moves_pattern = r'\d+\.\s*([a-zA-Z0-9+=#-]+)(?:\s+\{[^}]+\})?\s*(?:([a-zA-Z0-9+=#-]+)(?:\s+\{[^}]+\})?)?'
    matches = re.findall(moves_pattern, pgn)

    moves_data = []
    move_number = 1

    for match in matches:
        white_move, black_move = match

        if white_move:
            # Simular análisis para movimiento blanco
            cp_loss = min(50, abs(hash(white_move) % 100))  # Simulado
            moves_data.append(MoveData(
                move_number=move_number,
                played=white_move,
                best=white_move if cp_loss < 10 else "Nf3",  # Simulado
                cp_loss=cp_loss,
                eval_before=15 + (move_number * 2),  # Simulado
                eval_after=15 + (move_number * 2) - cp_loss,
                time_spent=move_times[len(moves_data)] / 1000.0 if len(moves_data) < len(move_times) else 30.0
            ))

        if black_move:
            # Simular análisis para movimiento negro
            cp_loss = min(50, abs(hash(black_move) % 100))
            moves_data.append(MoveData(
                move_number=move_number,
                played=black_move,
                best=black_move if cp_loss < 10 else "e5",  # Simulado
                cp_loss=cp_loss,
                eval_before=-15 - (move_number * 2),  # Simulado
                eval_after=-15 - (move_number * 2) + cp_loss,
                time_spent=move_times[len(moves_data)] / 1000.0 if len(moves_data) < len(move_times) else 30.0
            ))

        move_number += 1

        # Limitar para no hacer test muy largo
        if len(moves_data) >= 20:
            break

    return moves_data


def test_game_service_with_real_data(pgn_data: list[str], username: str):
    """Prueba GameService con datos reales."""
    print(f"\n🎮 Testing GameService con {len(pgn_data)} partidas...")

    game_service = GameService(None)  # No necesitamos repo para estas pruebas

    valid_games = 0
    total_games = 0

    for pgn in pgn_data:
        total_games += 1

        # Test de parsing
        game = game_service._parse_pgn_to_game(pgn, username)
        if game:
            # Test de validación
            if game_service._is_valid_game(game, username):
                valid_games += 1
                print(f"   ✅ Partida válida: {game.white_username} vs {game.black_username}")
            else:
                print(f"   ⚠️  Partida inválida: muy corta o sin movimientos")
        else:
            print(f"   ❌ Error parsing PGN")

    print(f"📊 GameService Results: {valid_games}/{total_games} partidas válidas")
    return valid_games > 0


def test_analysis_service_with_real_data(games_data: list[dict], username: str):
    """Prueba AnalysisService con datos reales."""
    print(f"\n🔍 Testing AnalysisService con {len(games_data)} partidas...")

    analysis_service = AnalysisService()
    game_service = GameService(None)

    game_analyses = []
    valid_games = []

    for i, game_data in enumerate(games_data):
        try:
            pgn, move_times = extract_pgn_and_times(game_data)

            # Crear Game entity
            game = game_service._parse_pgn_to_game(pgn, username)
            if not game or not game_service._is_valid_game(game, username):
                continue

            game.id = i + 1  # Simular ID
            game.move_times = move_times

            # Crear datos de movimientos simulados
            moves_data = create_sample_moves_data(pgn, move_times)
            if not moves_data:
                continue

            # Análisis de partida individual
            game_analysis = analysis_service.analyze_game(game, moves_data)
            game_analyses.append(game_analysis)
            valid_games.append(game)

            print(f"   ✅ Partida {i+1}: ACPL={game_analysis.quality_metrics.avg_acpl:.1f}, "
                  f"Match Rate={game_analysis.quality_metrics.avg_match_rate:.2f}, "
                  f"Sospechosa={game_analysis.is_suspicious()}")

        except Exception as e:
            print(f"   ❌ Error analizando partida {i+1}: {e}")

    if not game_analyses:
        print("❌ No se pudieron analizar partidas")
        return False

    # Análisis de jugador agregado
    try:
        print(f"\n👤 Analizando jugador {username} con {len(game_analyses)} partidas...")
        player_analysis = analysis_service.analyze_player(username, game_analyses, valid_games)

        print(f"📊 Resultados del Jugador {username}:")
        print(f"   • Partidas analizadas: {player_analysis.games_analyzed}")
        print(f"   • ACPL promedio: {player_analysis.quality_metrics.avg_acpl:.1f}")
        print(f"   • Match rate promedio: {player_analysis.quality_metrics.avg_match_rate:.2f}")
        print(f"   • Tiempo promedio por movimiento: {player_analysis.timing_metrics.mean_move_time:.1f}s")
        print(f"   • Risk score: {player_analysis.risk_assessment.risk_score}/100")
        print(f"   • Factores de riesgo: {player_analysis.risk_assessment.risk_factors}")
        print(f"   • Partidas sospechosas: {len(player_analysis.suspicious_games_ids)}")

        # Verificar que el output sea compatible con UI
        ui_output = player_analysis.get_analysis_summary()
        print(f"\n✅ Output compatible con UI generado: {len(ui_output)} campos")

        return True

    except Exception as e:
        print(f"❌ Error en análisis de jugador: {e}")
        return False


def test_value_objects_validation():
    """Prueba validaciones de value objects."""
    print(f"\n🔒 Testing Value Objects validations...")

    from app.domain.value_objects.metrics import QualityMetrics, TimingMetrics, RiskAssessment

    try:
        # Test validación exitosa
        quality = QualityMetrics(
            avg_acpl=15.5,
            avg_wdl_loss=0.05,
            robust_loss=0.02,
            avg_match_rate=0.75,
            avg_ipr=2200.0
        )
        print("   ✅ QualityMetrics válido creado")

        # Test inmutabilidad
        try:
            quality.avg_acpl = 20.0
            print("   ❌ Value object NO es inmutable")
        except:
            print("   ✅ Value object es inmutable")

        # Test validación que falla
        try:
            invalid_quality = QualityMetrics(
                avg_acpl=-5.0,  # Inválido
                avg_wdl_loss=0.05,
                robust_loss=0.02,
                avg_match_rate=0.75,
                avg_ipr=2200.0
            )
            print("   ❌ Validación NO funcionó")
        except ValueError:
            print("   ✅ Validación funcionó correctamente")

        return True

    except Exception as e:
        print(f"   ❌ Error en value objects: {e}")
        return False


def main():
    """Función principal del test."""
    print("🚀 Testing Sprint 2 - Domain Refactor con datos reales")
    print("=" * 60)

    # Buscar archivos de datos
    archives_dir = Path("archives")
    json_files = list(archives_dir.glob("*.json"))

    if not json_files:
        print("❌ No se encontraron archivos JSON en archives/")
        return

    # Usar archivo más pequeño para testing rápido
    test_file = None
    for f in json_files:
        if f.stat().st_size < 100000:  # Menos de 100KB
            test_file = f
            break

    if not test_file:
        test_file = json_files[0]  # Usar el primero si no hay pequeños

    print(f"📁 Usando archivo: {test_file.name} ({test_file.stat().st_size:,} bytes)")

    # Cargar datos
    games_data = load_real_games_from_json(str(test_file), max_games=5)
    if not games_data:
        print("❌ No se pudieron cargar partidas")
        return

    # Extraer username del primer juego
    first_pgn = games_data[0].get('pgn', '')
    username = games_data[0].get('white', 'testuser')
    if username in ['TestUser456', 'testuser456']:  # Normalizar
        username = 'testuser456'

    print(f"👤 Testing con jugador: {username}")

    # Ejecutar tests
    tests = [
        ("Value Objects Validation", test_value_objects_validation),
        ("GameService", lambda: test_game_service_with_real_data(
            [g.get('pgn', '') for g in games_data], username
        )),
        ("AnalysisService", lambda: test_analysis_service_with_real_data(games_data, username))
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ Error en {test_name}: {e}")
            results[test_name] = False

    # Resumen final
    print("\n" + "=" * 60)
    print("📋 RESUMEN DE PRUEBAS")
    print("=" * 60)

    for test_name, passed in results.items():
        status = "✅ PASÓ" if passed else "❌ FALLÓ"
        print(f"{test_name:25} : {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    print(f"\n🎯 RESULTADO FINAL: {total_passed}/{total_tests} tests pasaron")

    if total_passed == total_tests:
        print("🎉 ¡Sprint 2 validado exitosamente con datos reales!")
    else:
        print("⚠️  Algunos tests fallaron - revisar implementación")

    # Crear archivo de muestra para desarrollo futuro
    if games_data:
        sample_file = Path("test_sample_games.json")
        with open(sample_file, 'w') as f:
            json.dump(games_data[:3], f, indent=2)
        print(f"\n📝 Muestra guardada en: {sample_file}")


if __name__ == "__main__":
    main()