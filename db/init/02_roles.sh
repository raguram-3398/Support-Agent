#!/bin/bash
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
    --username "$POSTGRES_USER" \
    --dbname "$POSTGRES_DB" \
    -v db="$POSTGRES_DB" \
    -v read_pw="$SUPPORT_READ_PASSWORD" \
    -v write_pw="$SUPPORT_WRITE_PASSWORD" <<'SQL'
CREATE ROLE support_read LOGIN PASSWORD :'read_pw';
CREATE ROLE support_write LOGIN PASSWORD :'write_pw';

REVOKE ALL ON DATABASE :"db" FROM PUBLIC;
GRANT CONNECT ON DATABASE :"db" TO support_read, support_write;
GRANT USAGE ON SCHEMA public TO support_read, support_write;

GRANT SELECT ON customers, orders, refunds TO support_read;

GRANT SELECT ON orders TO support_write;
GRANT INSERT ON refunds TO support_write;
GRANT SELECT (id, order_id) ON refunds TO support_write;
SQL
