#!/usr/bin/env sh
set -e

max_tries=10
try=1

until alembic upgrade head; do
  if [ "$try" -ge "$max_tries" ]; then
    echo "Alembic migration failed after ${max_tries} attempts." >&2
    exit 1
  fi
  echo "Database not ready, retrying..." >&2
  try=$((try + 1))
  sleep 2
done

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
