import hashlib
import logging

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.models import Base
from app.db import wiki_models
from app.db import memory_models  # register compact memory tables
from app.db import model_budget_models  # register durable model budget tables
from app.db import quality_models  # register feedback, diagnostics and review tables
from app.db import workout_session_models  # register normalized session tables before create_all
from app.db import execution_ledger_models  # register v2 execution ledger tables before create_all
from app.db.execution_ledger_schema import ensure_execution_ledger_schema  # verify frozen ledger shape


logger = logging.getLogger(__name__)
_SQLITE_FK_MARKER = "_fitzen_sqlite_foreign_keys_configured"


def _sqlite_set_foreign_keys(dbapi_connection, _connection_record) -> None:
    """Enable SQLite foreign keys on every new physical connection (batch A).

    Foreign-key enforcement is per connection and a no-op inside a transaction,
    so this runs on the raw DBAPI connection before any business statement.
    """
    cursor = dbapi_connection.cursor()
    try:
        cursor.execute("PRAGMA foreign_keys=ON")
    finally:
        cursor.close()


def _sqlite_is_in_memory(engine: Engine) -> bool:
    database = str(engine.url.database or "")
    if not database or ":memory:" in database or "mode=memory" in database:
        return True
    return "mode=memory" in str(engine.url)


def configure_sqlite_engine(engine: Engine) -> Engine:
    """Attach the production SQLite connection configuration (idempotent).

    Call this before the engine's first connection so every new physical
    connection, pooled reuse and post-dispose connection has
    ``PRAGMA foreign_keys=1``. If the engine already has pooled connections
    before configuration, idle connections are recycled only when nothing is
    checked out (never during an active business transaction, and never for
    in-memory databases, whose data lives in the pooled connection); otherwise
    the caller recycles the pool at a safe point. PostgreSQL engines are
    returned unchanged.
    """
    if engine is None or engine.dialect.name != "sqlite":
        return engine
    if getattr(engine, _SQLITE_FK_MARKER, False):
        return engine
    event.listen(engine, "connect", _sqlite_set_foreign_keys)
    setattr(engine, _SQLITE_FK_MARKER, True)
    try:
        if not _sqlite_is_in_memory(engine):
            checkedout = getattr(engine.pool, "checkedout", None)
            if checkedout is not None and checkedout() == 0:
                engine.dispose()
    except Exception:
        logger.debug("sqlite_engine_recycle_skipped", exc_info=True)
    return engine


def create_database_engine(database_url: str, **kwargs) -> Engine:
    """Production engine factory; SQLite gets foreign keys on every connection.

    Tests can create isolated engines through this same entry point instead of
    installing their own PRAGMA listeners.
    """
    if database_url.startswith("postgresql"):
        connect_args = dict(kwargs.pop("connect_args", None) or {})
        connect_args.setdefault("connect_timeout", 3)
        return create_engine(database_url, pool_pre_ping=True, connect_args=connect_args, **kwargs)
    return configure_sqlite_engine(create_engine(database_url, pool_pre_ping=True, **kwargs))


engine = create_database_engine(settings.database_url) if settings.database_url else None
SessionLocal = sessionmaker(bind=engine, class_=Session, expire_on_commit=False) if engine else None
DB_READY = False


def _legacy_hash(title: str, content: str) -> str:
    normalized = f"{title or ''}\n{content or ''}".strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _migrate_knowledge_schema(connection) -> None:
    """为已有 knowledge_chunks 表补齐父子文档和去重字段。"""
    statements = (
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS document_id VARCHAR(128)",
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS parent_id VARCHAR(128)",
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS chunk_index INTEGER DEFAULT 0",
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS chunk_count INTEGER DEFAULT 1",
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
        "ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS metadata_json TEXT",
    )
    for statement in statements:
        connection.execute(text(statement))

    legacy_rows = connection.execute(
        text("SELECT id, title, content, source FROM knowledge_chunks WHERE document_id IS NULL OR content_hash IS NULL")
    ).mappings().all()
    for row in legacy_rows:
        content_hash = _legacy_hash(row["title"], row["content"])
        document_id = f"doc-{content_hash[:24]}"
        connection.execute(
            text(
                """
                INSERT INTO knowledge_documents
                    (document_id, title, content, source, content_hash)
                VALUES (:document_id, :title, :content, :source, :content_hash)
                ON CONFLICT (document_id) DO NOTHING
                """
            ),
            {"document_id": document_id, "title": row["title"], "content": row["content"], "source": row["source"], "content_hash": content_hash},
        )
        connection.execute(
            text(
                """
                UPDATE knowledge_chunks
                SET document_id=:document_id, parent_id=:document_id,
                    chunk_index=0, chunk_count=1, content_hash=:content_hash
                WHERE id=:id
                """
            ),
            {"document_id": document_id, "content_hash": content_hash, "id": row["id"]},
        )

    connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_documents_content_hash ON knowledge_documents(content_hash)"))
    connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_knowledge_chunks_content_hash ON knowledge_chunks(content_hash) WHERE content_hash IS NOT NULL"))
    # pgvector 默认只建了主键/普通索引时，数据量上来会退化为全表排序。
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw ON knowledge_chunks USING hnsw (embedding vector_cosine_ops)"))
    connection.execute(text("CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_source_document ON knowledge_chunks(source, document_id)"))


def _migrate_workout_schema(connection) -> None:
    """为已有训练表补齐难度和完成状态字段。"""
    statements = (
        "ALTER TABLE workout_sessions ADD COLUMN IF NOT EXISTS source_record_id INTEGER REFERENCES workout_records(id) ON DELETE SET NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_workout_sessions_source_record ON workout_sessions(source_record_id)",
        "ALTER TABLE workout_records ADD COLUMN IF NOT EXISTS difficulty VARCHAR(20) DEFAULT '未记录'",
        "ALTER TABLE workout_records ADD COLUMN IF NOT EXISTS completed BOOLEAN DEFAULT TRUE",
        "ALTER TABLE workout_records ADD COLUMN IF NOT EXISTS set_details TEXT",
        "ALTER TABLE workout_records ADD COLUMN IF NOT EXISTS idempotency_key VARCHAR(150)",
        "ALTER TABLE workout_session_exercises ADD COLUMN IF NOT EXISTS difficulty VARCHAR(20) DEFAULT '未记录'",
        "ALTER TABLE workout_session_sets ADD COLUMN IF NOT EXISTS difficulty VARCHAR(20) DEFAULT '未记录'",
        # R012 upgrade source: persistent source-record binding on one action.
        # Submitted as upgrade source only; no real migration is executed here.
        "ALTER TABLE workout_session_exercises ADD COLUMN IF NOT EXISTS source_record_id INTEGER REFERENCES workout_records(id) ON DELETE SET NULL",
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_workout_session_exercises_source_record ON workout_session_exercises(source_record_id)",
    )
    for statement in statements:
        connection.execute(text(statement))
    connection.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_workout_records_user_idempotency ON workout_records(user_id, idempotency_key) WHERE idempotency_key IS NOT NULL"))


def _migrate_memory_schema(connection) -> None:
    """Add retry-operation state and user soft-delete columns on existing databases."""
    if connection.dialect.name == "sqlite":
        columns = {row[1] for row in connection.execute(text("PRAGMA table_info(memory_turns)"))}
        if columns and "operations" not in columns:
            connection.execute(text("ALTER TABLE memory_turns ADD COLUMN operations JSON"))
        if columns and "hidden_at" not in columns:
            connection.execute(text("ALTER TABLE memory_turns ADD COLUMN hidden_at DATETIME"))
        return
    connection.execute(text("ALTER TABLE memory_turns ADD COLUMN IF NOT EXISTS operations JSON"))
    connection.execute(text("ALTER TABLE memory_turns ADD COLUMN IF NOT EXISTS hidden_at TIMESTAMPTZ"))


def _sqlite_dbapi_connection(connection):
    raw = getattr(connection, "connection", None)
    if raw is None:
        return None
    return (getattr(raw, "dbapi_connection", None)
            or getattr(raw, "driver_connection", None)
            or getattr(raw, "connection", None))


def _ensure_sqlite_transaction(connection) -> None:
    """Start a real SQLite transaction before the first DDL (batch B).

    The pysqlite driver can defer ``BEGIN`` until DML, so the Python-level
    ``engine.begin()`` context alone is not atomicity evidence. Emit ``BEGIN``
    only when the DBAPI connection is not already in a transaction; never
    commit or roll back the caller's transaction.
    """
    dbapi = _sqlite_dbapi_connection(connection)
    if dbapi is None:
        return
    if getattr(dbapi, "in_transaction", False):
        return
    connection.exec_driver_sql("BEGIN")


_WORKOUT_BINDING_TABLES = (
    ("workout_sessions", "uq_workout_sessions_source_record"),
    ("workout_session_exercises", "uq_workout_session_exercises_source_record"),
)


def _sqlite_index_columns(connection, index_name: str):
    """Column names of one index, read as metadata (None for expressions)."""
    rows = connection.execute(
        text("SELECT name FROM pragma_index_info(:index_name)"), {"index_name": index_name}
    ).mappings().all()
    return [None if row["name"] is None else str(row["name"]) for row in rows]


def _sqlite_source_unique_guarantee(connection, indexes) -> bool:
    """True only when metadata proves the whole source column is unique.

    A real guarantee is a unique, non-partial index whose only column is
    source_record_id (the model's automatic unique index or a correctly named
    index). Names, composite/expression/partial indexes and uniqueness of other
    columns do not count.
    """
    for row in indexes:
        if not int(row["unique"] or 0) or int(row.get("partial") or 0):
            continue
        if _sqlite_index_columns(connection, str(row["name"])) == ["source_record_id"]:
            return True
    return False


def _sqlite_workout_binding_upgrade(connection) -> None:
    """Add the nullable source_record_id binding to old SQLite workout tables.

    Supported shape: the other required workout fields and relation constraints
    are intact and only these nullable bindings are missing (or their effective
    unique guarantee is missing). The column is added as ``INTEGER NULL
    REFERENCES workout_records(id) ON DELETE SET NULL`` (no default, so
    existing rows stay NULL) and uniqueness is added as a separate index,
    avoiding a table rebuild. Existing rows, IDs, values and other columns are
    untouched. A pre-existing column with an unsupported type, nullability or
    foreign key fails initialization instead of being silently repaired;
    duplicate non-null bindings fail the unique index creation and roll the
    whole upgrade back. Uniqueness is verified from real index metadata: a
    same-named index that does not forbid duplicate non-null source values is
    rejected (never dropped, rewritten or bypassed with another name) unless
    another valid single-column source unique constraint already exists.
    """
    for table, index_name in _WORKOUT_BINDING_TABLES:
        info = connection.execute(text(f"PRAGMA table_info({table})")).mappings().all()
        if not info:
            raise RuntimeError(f"workout schema upgrade: table {table} is missing")
        column_types = {str(row["name"]): str(row["type"] or "") for row in info}
        column_notnull = {str(row["name"]): int(row["notnull"] or 0) for row in info}
        if "source_record_id" not in column_types:
            connection.execute(text(
                f"ALTER TABLE {table} ADD COLUMN source_record_id INTEGER "
                "REFERENCES workout_records(id) ON DELETE SET NULL"
            ))
        elif ("INT" not in column_types["source_record_id"].upper()
                or column_notnull.get("source_record_id")):
            raise RuntimeError(
                f"workout schema upgrade: {table}.source_record_id has an unsupported shape")
        foreign_keys = connection.execute(text(f"PRAGMA foreign_key_list({table})")).mappings().all()
        binding = [row for row in foreign_keys if str(row.get("from")) == "source_record_id"]
        if (len(binding) != 1
                or str(binding[0].get("table")) != "workout_records"
                or str(binding[0].get("to")) != "id"
                or str(binding[0].get("on_delete") or "").upper() != "SET NULL"):
            raise RuntimeError(
                f"workout schema upgrade: {table}.source_record_id foreign key is unsupported")
        indexes = connection.execute(text(f"PRAGMA index_list({table})")).mappings().all()
        if not _sqlite_source_unique_guarantee(connection, indexes):
            occupied = {str(row["name"]).casefold() for row in indexes}
            if index_name.casefold() in occupied:
                raise RuntimeError(
                    f"workout schema upgrade: {table}.source_record_id has no effective unique "
                    f"constraint and index {index_name} does not provide it")
            connection.execute(text(
                f"CREATE UNIQUE INDEX IF NOT EXISTS {index_name} ON {table}(source_record_id)"))
            created = connection.execute(text(f"PRAGMA index_list({table})")).mappings().all()
            if not _sqlite_source_unique_guarantee(connection, created):
                raise RuntimeError(
                    f"workout schema upgrade: {table}.source_record_id unique constraint "
                    "could not be verified after creation")


def init_db() -> bool:
    global DB_READY
    if engine is None:
        return False
    try:
        with engine.begin() as connection:
            if connection.dialect.name == "sqlite":
                # Real DDL transaction before the first schema statement.
                _ensure_sqlite_transaction(connection)
            if connection.dialect.name == "postgresql":
                connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            Base.metadata.create_all(bind=connection)
            from app.db.cardio_metrics_schema import ensure_cardio_metrics_schema
            ensure_cardio_metrics_schema(connection)
            from app.db.profile_schema import ensure_profile_schema
            ensure_profile_schema(connection)
            ensure_execution_ledger_schema(connection)
            if connection.dialect.name == "postgresql":
                connection.execute(text("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS semantic_embedding vector(1024)"))
                connection.execute(text("ALTER TABLE knowledge_chunks ADD COLUMN IF NOT EXISTS embedding_version VARCHAR(100)"))
                connection.execute(text("ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS memory_epoch INTEGER NOT NULL DEFAULT 0"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS ix_knowledge_semantic_hnsw ON knowledge_chunks USING hnsw (semantic_embedding vector_cosine_ops)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS ix_memory_semantic_hnsw ON memory_turns USING hnsw (summary_embedding vector_cosine_ops)"))
                connection.execute(text("CREATE INDEX IF NOT EXISTS ix_wiki_summary_hnsw ON wiki_pages USING hnsw (summary_embedding vector_cosine_ops)"))
                _migrate_knowledge_schema(connection)
                _migrate_workout_schema(connection)
                connection.execute(text("ALTER TABLE user_profiles ADD COLUMN IF NOT EXISTS target_weight_kg DOUBLE PRECISION"))
                from app.db.food_repository import migrate_food_schema

                migrate_food_schema(connection)
            elif connection.dialect.name == "sqlite":
                _sqlite_workout_binding_upgrade(connection)
            from app.db.knowledge_version import install_knowledge_versioning

            install_knowledge_versioning(connection)
            _migrate_memory_schema(connection)
        DB_READY = True
        return True
    except Exception:
        logger.exception("database_initialization_failed")
        DB_READY = False
        return False
