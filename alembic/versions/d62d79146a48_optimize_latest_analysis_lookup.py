"""optimize latest analysis lookup

Revision ID: d62d79146a48
Revises: 74f2e283c6ef
Create Date: 2026-08-12 12:06:07.323967

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd62d79146a48'
down_revision: Union[str, Sequence[str], None] = '74f2e283c6ef'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "ix_analyses_document_created_at",
        "analyses",
        ["document_id", "created_at"],
        postgresql_ops={"created_at": "DESC"},
    )


def downgrade() -> None:
    op.drop_index(
        "ix_analyses_document_created_at",
        table_name="analyses",
    )