"""add_prompt_template_to_analyses

Revision ID: ce5f679b5564
Revises: 4679ae46c045
Create Date: 2026-08-12 11:16:03.858774

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'ce5f679b5564'
down_revision: Union[str, Sequence[str], None] = '4679ae46c045'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Step 1: Add nullable
    op.add_column('analyses', sa.Column('prompt_template', sa.Text(), nullable=True))
    
    # Step 2: Backfill existing rows (optional - adjust placeholder as needed)
    op.execute("UPDATE analyses SET prompt_template = 'legacy' WHERE prompt_template IS NULL")
    
    # Step 3: Make it NOT NULL (only if you really need it)
    op.alter_column('analyses', 'prompt_template', nullable=False)


def downgrade() -> None:
    op.drop_column('analyses', 'prompt_template')