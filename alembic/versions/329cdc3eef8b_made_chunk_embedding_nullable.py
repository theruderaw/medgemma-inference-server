"""made chunk.embedding nullable

Revision ID: 329cdc3eef8b
Revises: 7287b812b713
Create Date: 2026-08-08 01:02:06.325259

"""

from typing import Sequence, Union

from alembic import op
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.

revision: str = "329cdc3eef8b"
down_revision: Union[str, Sequence[str], None] = "7287b812b713"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(768),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "chunks",
        "embedding",
        existing_type=Vector(768),
        nullable=False,
    )