"""Temporary test databases; never run application SQL against the admin database."""
from contextlib import contextmanager
import ipaddress
import json
import os
import re
import socket
from datetime import datetime, timezone
from uuid import uuid4

import pytest
from sqlalchemy import create_engine as sa_create_engine, event, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import NullPool


def checked_engine(url, **kwargs):
    """Verify the actual database on every PostgreSQL pool checkout, before SQL."""
    parsed = make_url(url)
    expected = parsed.database
    if parsed.get_backend_name() == "postgresql":
        assert re.fullmatch(r"fitzen_test_[0-9a-f]{32}", expected or ""), "non-test database rejected"
        kwargs.setdefault("connect_args", {"connect_timeout": 3})
        if not parsed.query.get("hostaddr") and "hostaddr" not in kwargs["connect_args"]:
            hostaddr = _verified_loopback_hostaddr(parsed)
            if hostaddr:
                kwargs["connect_args"] = {**kwargs["connect_args"], "hostaddr": hostaddr}
    # Exercise production connection defaults; keep the exact PG whitelist
    # above and the physical checkout identity verification below.
    from app.db.session import create_database_engine
    engine = create_database_engine(url, **kwargs)
    if engine.dialect.name == "postgresql":
        @event.listens_for(engine, "checkout")
        def verify_database(connection, record, proxy):
            # Do not leave a verification transaction open in the pooled connection.
            previous = connection.autocommit
            connection.autocommit = True
            try:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT current_database()")
                    actual = cursor.fetchone()[0]
                assert actual == expected, "test database identity mismatch"
            finally:
                connection.autocommit = previous
    return engine


def _audit_pg_database(name, event):
    """Record exact temporary database lifecycle events when an audit path is configured.

    Only the generated database name and the event are written; never a connection URL.
    """
    path = os.environ.get("FITZEN_PG_DATABASE_AUDIT")
    if not path:
        return
    assert re.fullmatch(r"fitzen_test_[0-9a-f]{32}", name), "unexpected temporary database identity"
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"database": name, "event": event,
                                 "utc": datetime.now(timezone.utc).isoformat()}) + "\n")


_VERIFIED_HOSTADDR = {}


def _verified_loopback_hostaddr(parsed):
    """Verify once per process, outside request deny-network scope, which loopback
    address actually connects for the authorized test PostgreSQL host, so libpq
    can skip per-connection resolution via hostaddr while host keeps auth/TLS
    identity. Numeric hosts and explicit hostaddr values are kept as configured;
    ambiguous or unverifiable names are refused, never guessed.
    """
    host = parsed.host
    if host is None:
        raise RuntimeError("test PostgreSQL configuration requires an explicit host")
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        return None
    if parsed.query.get("hostaddr"):
        return None
    if host != "localhost":
        raise RuntimeError("test PostgreSQL host must be numeric or localhost")
    if host in _VERIFIED_HOSTADDR:
        return _VERIFIED_HOSTADDR[host]
    candidates = []
    for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM):
        candidate = info[4][0]
        if ipaddress.ip_address(candidate).is_loopback and candidate not in candidates:
            candidates.append(candidate)
    verified = None
    for candidate in candidates:
        probe = sa_create_engine(parsed.set(host=candidate, database="postgres"),
                                 connect_args={"connect_timeout": 2}, poolclass=NullPool)
        try:
            with probe.connect():
                verified = candidate
        except Exception:
            continue
        finally:
            probe.dispose()
        if verified:
            break
    if verified is None:
        raise RuntimeError("no verified loopback address for the test PostgreSQL host; refusing to guess")
    _VERIFIED_HOSTADDR[host] = verified
    return verified


def _pinned_test_url(parsed):
    hostaddr = _verified_loopback_hostaddr(parsed)
    if not hostaddr:
        return parsed
    return parsed.update_query_dict({"hostaddr": hostaddr})


class TempPGDatabase:
    """Only CREATE/DROP the generated database; never inspect or clean business rows."""
    def __init__(self):
        from app.core.config import settings
        parsed = _pinned_test_url(make_url(settings.database_url))
        self.name = "fitzen_test_" + uuid4().hex
        self.url = parsed.set(database=self.name).render_as_string(hide_password=False)
        self.raw = sa_create_engine(parsed.set(database="postgres"),
                                    connect_args={"connect_timeout": 3})
        self.created = False

    def create(self):
        try:
            with self.raw.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text('CREATE DATABASE "' + self.name + '"'))
            self.created = True
            _audit_pg_database(self.name, "create")
        except Exception as exc:
            self.raw.dispose()
            if getattr(getattr(exc, "orig", None), "sqlstate", None) == "42501":
                pytest.skip("test role cannot create a temporary database; no fallback")
            raise

    def drop(self):
        try:
            if self.created:
                assert re.fullmatch(r"fitzen_test_[0-9a-f]{32}", self.name)
                with self.raw.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                    # Every test engine/connection must already be closed. Do not kill sessions.
                    conn.execute(text('DROP DATABASE "' + self.name + '"'))
                self.created = False
                _audit_pg_database(self.name, "drop")
        finally:
            self.raw.dispose()


@contextmanager
def isolated_database(monkeypatch, tmp_path, backend):
    from app.core.config import settings
    from app.db import session as db
    from app.db.session import init_db
    temp_db = None
    engine = None
    try:
        if backend == "postgres":
            if os.getenv("FITZEN_DB_TESTS") != "1" or not settings.database_url.startswith("postgresql"):
                pytest.skip("PostgreSQL integration requires explicit opt-in")
            temp_db = TempPGDatabase()
            temp_db.create()
            url = temp_db.url
        else:
            url = "sqlite:///" + str(tmp_path / "quality.db")
        engine = checked_engine(url)
        with monkeypatch.context() as scoped:
            scoped.setattr(db, "engine", engine)
            scoped.setattr(db, "SessionLocal", sessionmaker(bind=engine, expire_on_commit=False))
            scoped.setattr(db, "DB_READY", False)
            scoped.setattr(settings, "database_url", url)
            scoped.setattr(settings, "memory_read_backend", "compact_v1")
            assert init_db(), "temporary database initialization failed"
            if engine.dialect.name == "sqlite":
                from conftest import _USER_FOOD_LOG_DDL
                with engine.begin() as conn:
                    conn.execute(text(_USER_FOOD_LOG_DDL))
            yield engine
    finally:
        if engine is not None:
            engine.dispose()
        if temp_db is not None:
            temp_db.drop()
