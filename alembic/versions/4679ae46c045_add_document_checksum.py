"""add document checksum

Revision ID: 4679ae46c045
Revises: 7cbabf818731
Create Date: 2026-08-11 18:57:41.334122

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '4679ae46c045'
down_revision: Union[str, Sequence[str], None] = '7cbabf818731'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


"""add document checksum

Revision ID: <new_revision>
Revises: <previous_revision>
Create Date: 2026-08-11
"""



def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column(
            "checksum",
            sa.String(length=64),
            nullable=True,
        ),
    )

    op.create_index(
        "ix_documents_checksum",
        "documents",
        ["checksum"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_documents_checksum",
        table_name="documents",
    )

    op.drop_column(
        "documents",
        "checksum",
    )