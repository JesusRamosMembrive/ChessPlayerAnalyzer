"""add_wdl_loss_columns

Revision ID: a1b2c3d4e5f6
Revises: 9a2b9628cf02
Create Date: 2025-08-23 00:29:41.765802

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'a1b2c3d4e5f6'
down_revision = '9a2b9628cf02'
branch_labels = None
depends_on = None


def upgrade():
    op.add_column('game_analysis_detailed', sa.Column('wdl_loss', sa.Float(), nullable=True))
    op.add_column('player_analysis_detailed', sa.Column('avg_wdl_loss', sa.Float(), nullable=True))


def downgrade():
    op.drop_column('game_analysis_detailed', 'wdl_loss')
    op.drop_column('player_analysis_detailed', 'avg_wdl_loss')
