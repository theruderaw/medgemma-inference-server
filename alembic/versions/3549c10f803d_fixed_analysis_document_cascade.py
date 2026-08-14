"""fixed analysis.document cascade

Revision ID: 3549c10f803d
Revises: 447f6ca51029
Create Date: 2026-08-14 10:21:49.023258
"""

from typing import Sequence, Union

from alembic import op


revision: str = "3549c10f803d"
down_revision: Union[str, Sequence[str], None] = "447f6ca51029"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Remove ON DELETE CASCADE
    op.drop_constraint(
        "analyses_document_id_fkey",
        "analyses",
        type_="foreignkey",
    )

    op.create_foreign_key(
        "analyses_document_id_fkey",
        "analyses",
        "documents",
        ["document_id"],
        ["document_id"],
    )


def downgrade() -> None:
    # Restore ON DELETE CASCADE
    op.drop_constraint(
        "analyses_document_id_fkey",
        "analyses",
        type_="foreignkey",
    )

    op.create_foreign_key(
        "analyses_document_id_fkey",
        "analyses",
        "documents",
        ["document_id"],
        ["document_id"],
        ondelete="CASCADE",
    )