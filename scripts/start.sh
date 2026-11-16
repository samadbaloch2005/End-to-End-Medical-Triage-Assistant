#!/bin/sh
# Container entry point: bring the database up to date, add demo data if asked, start the server.
set -e

python -m triagecare.cli migrate

if [ "${SEED_DEMO:-false}" = "true" ]; then
  python -m triagecare.cli seed
fi

exec uvicorn triagecare.main:app \
  --host 0.0.0.0 --port "${PORT:-8000}" \
  --proxy-headers --forwarded-allow-ips="*"
