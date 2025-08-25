"""fix pgn index size limit

Revision ID: e1f2g3h4i5j6
Revises: a1b2c3d4e5f6
Create Date: 2025-08-25 19:18:10.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'e1f2g3h4i5j6'
down_revision = 'a1b2c3d4e5f6'
branch_labels = None
depends_on = None


def upgrade():
    op.drop_index('ix_game_pgn_white_black', table_name='game')
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_game_pgn_md5_white_black 
        ON game (MD5(pgn), white_username, black_username)
    """)


def downgrade():
    op.drop_index('ix_game_pgn_md5_white_black', table_name='game')
    
    op.execute("""
        CREATE INDEX IF NOT EXISTS ix_game_pgn_white_black 
        ON game (pgn, white_username, black_username)
    """)
