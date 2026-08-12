"""add_extract_prompt_template_to_analyses

Revision ID: 74f2e283c6ef
Revises: ce5f679b5564
Create Date: 2026-08-12 11:28:13.803252

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '74f2e283c6ef'
down_revision: Union[str, Sequence[str], None] = 'ce5f679b5564'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add as nullable – no default needed
    op.add_column(
        'analyses',
        sa.Column('extract_prompt_template', sa.Text(), nullable=True)
    )


def downgrade() -> None:
    op.drop_column('analyses', 'extract_prompt_template')