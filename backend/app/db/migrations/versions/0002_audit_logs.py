"""add audit logs

Revision ID: 0002_audit_logs
Revises: 0001_init
Create Date: 2026-04-08 00:00:00.000000

"""
from __future__ import annotations

import sqlalchemy as sa
from alembic import op

from app.db.types import GUID, JSONType

revision = "0002_audit_logs"
down_revision = "0001_init"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "audit_logs",
        sa.Column("id", GUID(), primary_key=True, nullable=False),
        sa.Column("event_type", sa.String(length=50), nullable=False),
        sa.Column("actor_role", sa.String(length=20), nullable=False),
        sa.Column("route", sa.Text(), nullable=False),
        sa.Column("status_code", sa.Integer(), nullable=False),
        sa.Column("details", JSONType(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
