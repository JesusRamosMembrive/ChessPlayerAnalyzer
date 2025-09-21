"""
Servicio de dominio para gestión de partidas.
"""
from typing import List, Optional
import re
from datetime import datetime

from ..entities.game import Game, MoveData
from ..repositories.game_repository import GameRepository


class GameService:
    """
    Servicio de dominio para lógica de negocio de partidas.
    """

    def __init__(self, game_repo: GameRepository):
        self.game_repo = game_repo

    async def process_player_games(self, username: str, pgn_data: List[str]) -> List[Game]:
        """
        Procesa las partidas de un jugador desde datos PGN.

        Args:
            username: Nombre del jugador
            pgn_data: Lista de strings PGN

        Returns:
            Lista de partidas procesadas y guardadas
        """
        games = []

        for pgn in pgn_data:
            try:
                game = self._parse_pgn_to_game(pgn, username)
                if game and self._is_valid_game(game, username):
                    games.append(game)
            except Exception as e:
                # Log error pero continúa con el resto
                continue

        # Guardar en batch para eficiencia
        if games:
            saved_games = await self.game_repo.bulk_save(games)
            return saved_games

        return []

    async def get_player_games_summary(self, username: str) -> dict:
        """
        Obtiene resumen de partidas de un jugador.

        Args:
            username: Nombre del jugador

        Returns:
            Dict con estadísticas de partidas
        """
        games = await self.game_repo.get_by_player(username)

        if not games:
            return {
                "total_games": 0,
                "analyzed_games": 0,
                "white_games": 0,
                "black_games": 0,
                "time_controls": [],
                "date_range": None
            }

        analyzed_count = sum(1 for game in games if game.is_analyzed())
        white_count = sum(1 for game in games if game.white_username == username)
        black_count = len(games) - white_count

        time_controls = list(set(game.time_control for game in games
                               if game.time_control))

        dates = [game.created_at for game in games if game.created_at]
        date_range = {
            "earliest": min(dates) if dates else None,
            "latest": max(dates) if dates else None
        }

        return {
            "total_games": len(games),
            "analyzed_games": analyzed_count,
            "white_games": white_count,
            "black_games": black_count,
            "time_controls": time_controls,
            "date_range": date_range
        }

    async def delete_player_games(self, username: str) -> int:
        """
        Elimina todas las partidas de un jugador.

        Args:
            username: Nombre del jugador

        Returns:
            Número de partidas eliminadas
        """
        return await self.game_repo.delete_by_player(username)

    def _parse_pgn_to_game(self, pgn: str, target_username: str) -> Optional[Game]:
        """
        Parsea un string PGN a objeto Game.

        Args:
            pgn: String PGN
            target_username: Username objetivo para verificar participación

        Returns:
            Game parseado o None si hay error
        """
        try:
            # Extraer headers del PGN usando regex
            headers = self._extract_pgn_headers(pgn)

            white_username = headers.get('White', '').strip('"')
            black_username = headers.get('Black', '').strip('"')

            # Verificar que el jugador objetivo participa
            if target_username not in [white_username, black_username]:
                return None

            # Extraer otros campos
            white_elo = self._safe_int_parse(headers.get('WhiteElo'))
            black_elo = self._safe_int_parse(headers.get('BlackElo'))
            time_control = headers.get('TimeControl')
            termination = headers.get('Termination')
            eco_code = headers.get('ECO')

            # Extraer fecha
            date_str = headers.get('Date', '').strip('"')
            created_at = self._parse_pgn_date(date_str)

            return Game(
                id=None,  # Se asignará al guardar
                pgn=pgn,
                white_username=white_username,
                black_username=black_username,
                white_elo=white_elo,
                black_elo=black_elo,
                time_control=time_control,
                termination=termination,
                eco_code=eco_code,
                created_at=created_at
            )

        except Exception:
            return None

    def _extract_pgn_headers(self, pgn: str) -> dict:
        """Extrae headers del PGN usando regex."""
        headers = {}
        header_pattern = r'\[(\w+)\s+"([^"]+)"\]'

        for match in re.finditer(header_pattern, pgn):
            key, value = match.groups()
            headers[key] = value

        return headers

    def _safe_int_parse(self, value: Optional[str]) -> Optional[int]:
        """Parsea string a int de forma segura."""
        if not value:
            return None

        try:
            # Limpiar string (remover comillas, espacios)
            clean_value = str(value).strip('"').strip()
            return int(clean_value) if clean_value.isdigit() else None
        except (ValueError, AttributeError):
            return None

    def _parse_pgn_date(self, date_str: str) -> Optional[datetime]:
        """Parsea fecha del PGN."""
        if not date_str or date_str == '??':
            return None

        try:
            # Formato típico: "2023.12.25"
            if '.' in date_str:
                parts = date_str.split('.')
                if len(parts) == 3:
                    year, month, day = parts
                    if all(part.isdigit() for part in parts):
                        return datetime(int(year), int(month), int(day))

            # Otros formatos posibles...
            return None
        except (ValueError, IndexError):
            return None

    def _is_valid_game(self, game: Game, username: str) -> bool:
        """
        Valida que una partida sea válida para análisis.

        Args:
            game: Partida a validar
            username: Username del jugador

        Returns:
            True si la partida es válida
        """
        # Verificar que el jugador participa
        if username not in [game.white_username, game.black_username]:
            return False

        # Verificar que el PGN no esté vacío
        if not game.pgn or len(game.pgn.strip()) < 10:
            return False

        # Verificar que tenga movimientos básicos
        if '1.' not in game.pgn:
            return False

        # Filtrar partidas muy cortas (< 5 movimientos)
        move_count = len(re.findall(r'\d+\.', game.pgn))
        if move_count < 5:
            return False

        return True

    def validate_pgn(self, pgn: str) -> dict:
        """
        Valida formato PGN.

        Args:
            pgn: String PGN a validar

        Returns:
            Dict con resultado de validación
        """
        if not pgn or not pgn.strip():
            return {"valid": False, "reason": "PGN is empty"}

        # Verificar headers básicos
        required_headers = ['White', 'Black', 'Result']
        headers = self._extract_pgn_headers(pgn)

        missing_headers = [h for h in required_headers if h not in headers]
        if missing_headers:
            return {
                "valid": False,
                "reason": f"Missing required headers: {', '.join(missing_headers)}"
            }

        # Verificar que tenga movimientos
        if not re.search(r'\d+\.', pgn):
            return {"valid": False, "reason": "No moves found in PGN"}

        return {"valid": True, "reason": "Valid PGN format"}