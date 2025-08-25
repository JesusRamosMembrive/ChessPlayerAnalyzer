"""add_done_tasks_to_player

Revision ID: e5b8c71b3f2c
Revises: a1b2c3d4e5f6
Create Date: 2025-08-23 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = 'e5b8c71b3f2c'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column('player', sa.Column('done_tasks', sa.Integer(), nullable=True))
    op.execute("UPDATE player SET done_tasks = COALESCE(done_games, 0) * 2")


def downgrade() -> None:
    op.drop_column('player', 'done_tasks')
