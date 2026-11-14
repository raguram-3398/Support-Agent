#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -v checkpoint_pw="$SUPPORT_CHECKPOINT_PASSWORD" <<'SQL'
CREATE ROLE support_checkpoint LOGIN PASSWORD :'checkpoint_pw';
CREATE DATABASE checkpoints OWNER support_checkpoint;
REVOKE ALL ON DATABASE checkpoints FROM PUBLIC;
GRANT CONNECT ON DATABASE checkpoints TO support_checkpoint;
SQL
