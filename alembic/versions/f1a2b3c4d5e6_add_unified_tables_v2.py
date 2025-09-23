"""add unified tables v2

Revision ID: f1a2b3c4d5e6
Revises: 9a2b9628cf02
Create Date: 2025-09-23 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'f1a2b3c4d5e6'
down_revision = '9a2b9628cf02'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create new unified tables for refactor v2"""

    # Create player_status_v2 enum
    op.execute("CREATE TYPE player_status_v2 AS ENUM ('not_analyzed', 'pending', 'ready', 'error')")

    # Create game_v2 table (simplified)
    op.create_table('game_v2',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('pgn', sa.Text(), nullable=False),
        sa.Column('white_username', sa.String(), nullable=True),
        sa.Column('black_username', sa.String(), nullable=True),
        sa.Column('white_elo', sa.Integer(), nullable=True),
        sa.Column('black_elo', sa.Integer(), nullable=True),
        sa.Column('time_control', sa.String(), nullable=True),
        sa.Column('termination', sa.String(), nullable=True),
        sa.Column('eco_code', sa.String(), nullable=True),
        sa.Column('opening_key', sa.String(), nullable=True),
        sa.Column('move_times', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for game_v2
    op.create_index(op.f('ix_game_v2_white_username'), 'game_v2', ['white_username'], unique=False)
    op.create_index(op.f('ix_game_v2_black_username'), 'game_v2', ['black_username'], unique=False)

    # Create analysis_result_v2 table (unified)
    op.create_table('analysis_result_v2',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('game_id', sa.Integer(), nullable=False),
        sa.Column('player_username', sa.String(), nullable=False),
        sa.Column('player_color', sa.String(), nullable=False),
        sa.Column('analyzed_at', sa.DateTime(timezone=True), nullable=False),
        sa.Column('engine_depth', sa.Integer(), nullable=False, default=12),
        sa.Column('moves_analyzed', sa.Integer(), nullable=False, default=0),
        sa.Column('metrics', sa.JSON(), nullable=False),
        sa.ForeignKeyConstraint(['game_id'], ['game_v2.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create indexes for analysis_result_v2
    op.create_index(op.f('ix_analysis_result_v2_game_id'), 'analysis_result_v2', ['game_id'], unique=False)
    op.create_index(op.f('ix_analysis_result_v2_player_username'), 'analysis_result_v2', ['player_username'], unique=False)

    # Create player_v2 table (compatible with existing endpoints)
    op.create_table('player_v2',
        sa.Column('username', sa.String(), nullable=False),
        sa.Column('status', sa.Enum('not_analyzed', 'pending', 'ready', 'error', name='player_status_v2'), nullable=False),
        sa.Column('requested_at', sa.DateTime(), nullable=True),
        sa.Column('finished_at', sa.DateTime(), nullable=True),
        sa.Column('progress', sa.Integer(), nullable=False, default=0),
        sa.Column('total_games', sa.Integer(), nullable=False, default=0),
        sa.Column('done_games', sa.Integer(), nullable=False, default=0),
        sa.Column('error', sa.Text(), nullable=True),
        sa.Column('last_task_id', sa.String(), nullable=True),
        sa.Column('aggregated_metrics', sa.JSON(), nullable=True),
        sa.Column('first_game_date', sa.DateTime(), nullable=True),
        sa.Column('last_game_date', sa.DateTime(), nullable=True),
        sa.Column('analyzed_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('username')
    )

    # Create reference_stats_v2 table (unchanged functionality)
    op.create_table('reference_stats_v2',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('elo_range_min', sa.Integer(), nullable=False),
        sa.Column('elo_range_max', sa.Integer(), nullable=False),
        sa.Column('expected_acpl', sa.Float(), nullable=False),
        sa.Column('expected_match_rate', sa.Float(), nullable=False),
        sa.Column('expected_time_variance', sa.Float(), nullable=False),
        sa.Column('expected_opening_entropy', sa.Float(), nullable=False),
        sa.Column('std_acpl', sa.Float(), nullable=False),
        sa.Column('std_match_rate', sa.Float(), nullable=False),
        sa.Column('std_time_variance', sa.Float(), nullable=False),
        sa.Column('std_opening_entropy', sa.Float(), nullable=False),
        sa.Column('sample_size', sa.Integer(), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade() -> None:
    """Drop unified tables v2"""

    # Drop tables in reverse order (respecting foreign keys)
    op.drop_table('reference_stats_v2')
    op.drop_table('player_v2')

    # Drop indexes first
    op.drop_index(op.f('ix_analysis_result_v2_player_username'), table_name='analysis_result_v2')
    op.drop_index(op.f('ix_analysis_result_v2_game_id'), table_name='analysis_result_v2')
    op.drop_table('analysis_result_v2')

    # Drop game_v2 indexes and table
    op.drop_index(op.f('ix_game_v2_black_username'), table_name='game_v2')
    op.drop_index(op.f('ix_game_v2_white_username'), table_name='game_v2')
    op.drop_table('game_v2')

    # Drop enum type
    op.execute("DROP TYPE player_status_v2")