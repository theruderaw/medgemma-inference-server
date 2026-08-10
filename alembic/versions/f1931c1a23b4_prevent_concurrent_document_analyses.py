"""prevent concurrent document analyses

Revision ID: f1931c1a23b4
Revises: 87c6eaf8fc69
Create Date: 2026-08-10 23:34:05.972718

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1931c1a23b4'
down_revision: Union[str, Sequence[str], None] = '87c6eaf8fc69'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("""
        CREATE UNIQUE INDEX uq_active_analysis_per_document
        ON analyses (document_id)
        WHERE status IN ('ANALYZING', 'CHUNKING', 'EMBEDDING')
    """)


def downgrade() -> None:
    op.execute("""
        DROP INDEX IF EXISTS uq_active_analysis_per_document
    """)