CREATE TABLE customers (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    name text NOT NULL,
    email text NOT NULL UNIQUE
);

CREATE TABLE orders (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_id integer NOT NULL REFERENCES customers (id),
    status text NOT NULL CHECK (status IN ('placed', 'shipped', 'delivered', 'cancelled')),
    total_cents integer NOT NULL CHECK (total_cents > 0),
    placed_at timestamptz NOT NULL,
    note text
);

CREATE INDEX orders_customer_id_idx ON orders (customer_id);

CREATE TABLE refunds (
    id integer GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id integer NOT NULL UNIQUE REFERENCES orders (id),
    amount_cents integer NOT NULL CHECK (amount_cents > 0),
    reason text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
