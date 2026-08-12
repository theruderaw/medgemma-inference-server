"""add bm25 now

Revision ID: 2a5ca54bbfb6
Revises: 01bf6ee279f2
Create Date: 2026-08-12 21:41:04.654432

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '2a5ca54bbfb6'
down_revision: Union[str, Sequence[str], None] = '01bf6ee279f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None



def upgrade() -> None:
    op.execute("""
        ALTER TABLE chunks
        ADD COLUMN search_vector tsvector
        GENERATED ALWAYS AS (
            to_tsvector(
                'english',
                coalesce(chunk_content, '')
            )
        ) STORED
    """)

    op.execute("""
        CREATE INDEX chunks_search_vector_gin_idx
        ON chunks
        USING GIN (search_vector)
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS chunks_search_vector_gin_idx
    """)

    op.execute("""
        ALTER TABLE chunks
        DROP COLUMN IF EXISTS search_vector
    """)
