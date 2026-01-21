"""init

Revision ID: 0001_init
Revises: 
Create Date: 2026-01-20 00:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_init"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("label", sa.Text(), nullable=True),
        sa.Column("status", sa.String(length=20), nullable=False, server_default="active"),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
    )
    op.create_table(
        "embeddings",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("person_id", sa.UUID(), sa.ForeignKey("persons.id")),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
    )
    op.create_table(
        "index_snapshots",
        sa.Column("id", sa.UUID(), primary_key=True, nullable=False),
        sa.Column("index_type", sa.String(length=20), nullable=False),
        sa.Column("params", sa.JSON(), nullable=False),
        sa.Column("path", sa.Text(), nullable=False),
        sa.Column("embeddings_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("index_snapshots")
    op.drop_table("embeddings")
    op.drop_table("persons")
