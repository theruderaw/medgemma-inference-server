"""sync schema with current models

Revision ID: 46342948017a
Revises: d3a008a9c5e3
Create Date: 2026-08-06 17:23:29.684506

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '46342948017a'
down_revision: Union[str, Sequence[str], None] = 'd3a008a9c5e3'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    analysisstatus = postgresql.ENUM(
        'READY', 'ANALYZING', 'CHUNKING', 'EMBEDDING', 'COMPLETE', 'FAILED', 'DELETED',
        name='analysisstatus'
    )
    analysisstatus.create(op.get_bind(), checkfirst=True)

    op.add_column(
        'analyses',
        sa.Column('status', analysisstatus, nullable=False, server_default='READY')
    )
    op.alter_column('analyses', 'status', server_default=None)  # drop default after backfill, if you don't want one long-term


def downgrade():
    op.drop_column('analyses', 'status')
    postgresql.ENUM(name='analysisstatus').drop(op.get_bind(), checkfirst=True)
