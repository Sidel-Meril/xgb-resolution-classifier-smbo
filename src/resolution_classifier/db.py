"""PostgreSQL connection and dataframe reads for the training extract.

The same Neon database as ``data-cleaning-smbo``. Configure it with
``DATABASE_URL`` or the libpq ``PG*`` variables (copy ``.env.example`` to ``.env``).
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd
import psycopg
from psycopg.rows import dict_row


def _resolve_conninfo(conninfo: str | None) -> str:
    if conninfo:
        return conninfo
    return os.getenv("DATABASE_URL", "")


def has_database_config(conninfo: str | None = None) -> bool:
    """True when ``conninfo``, ``DATABASE_URL``, or libpq ``PG*`` vars are set."""
    if conninfo or os.getenv("DATABASE_URL"):
        return True
    return bool(os.getenv("PGHOST") and os.getenv("PGDATABASE") and os.getenv("PGUSER"))


def connect(conninfo: str | None = None, *, connect_timeout: int = 30) -> psycopg.Connection:
    """Open a configured psycopg connection with dict rows."""
    kwargs = {"row_factory": dict_row, "connect_timeout": connect_timeout}
    resolved = _resolve_conninfo(conninfo)
    if resolved:
        return psycopg.connect(resolved, **kwargs)
    return psycopg.connect(
        host=os.getenv("PGHOST") or None,
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD"),
        sslmode=os.getenv("PGSSLMODE", "require"),
        **kwargs,
    )


@contextmanager
def get_connection(conninfo: str | None = None) -> Iterator[psycopg.Connection]:
    """Yield a configured psycopg connection, closing it on exit."""
    connection = connect(conninfo)
    try:
        yield connection
    finally:
        connection.close()


def fetch_dataframe(connection: psycopg.Connection, statement: str) -> pd.DataFrame:
    """Execute ``statement`` and return the rows as a DataFrame (empty-safe)."""
    with connection.cursor() as cursor:
        cursor.execute(statement)
        rows = cursor.fetchall()
        names = [col.name for col in cursor.description] if cursor.description else []
    if not rows:
        return pd.DataFrame(columns=names)
    return pd.DataFrame(rows)
