"""add analysis document created index

Revision ID: a7193e1eabd0
Revises: f1931c1a23b4
Create Date: 2026-08-10 23:56:08.754047

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'a7193e1eabd0'
down_revision: Union[str, Sequence[str], None] = 'f1931c1a23b4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_index(
        "analyses_document_created_idx",
        "analyses",
        ["document_id", "created_at"],
        unique=False,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "analyses_document_created_idx",
        table_name="analyses",
    )
