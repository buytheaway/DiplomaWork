"""add embedding pipeline

Revision ID: 0003_embeddings_pipeline
Revises: 0002_audit_logs
Create Date: 2026-04-26 00:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0003_embeddings_pipeline"
down_revision = "0002_audit_logs"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("embeddings") as batch_op:
        batch_op.add_column(sa.Column("pipeline", sa.String(length=20), nullable=True))

    op.execute("UPDATE embeddings SET pipeline = 'pretrained' WHERE pipeline IS NULL")

    with op.batch_alter_table("embeddings") as batch_op:
        batch_op.alter_column(
            "pipeline",
            existing_type=sa.String(length=20),
            nullable=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("embeddings") as batch_op:
        batch_op.drop_column("pipeline")
