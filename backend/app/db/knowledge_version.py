"""Transactional knowledge revisions, including writes made by import scripts.

Database triggers cover ORM, bulk SQL and other processes. A rolled-back write
rolls back its revision too. Install inside init_db's schema transaction.
"""
from sqlalchemy import text


def install_knowledge_versioning(connection) -> None:
    connection.execute(text("""
        INSERT INTO knowledge_index_versions (corpus, revision)
        VALUES ('knowledge', 0) ON CONFLICT (corpus) DO NOTHING
    """))
    dialect = connection.dialect.name
    tables = ("knowledge_chunks", "knowledge_documents")
    if dialect == "postgresql":
        connection.execute(text("""
            CREATE OR REPLACE FUNCTION bump_knowledge_index_revision()
            RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN
                EXECUTE format(
                    'UPDATE %I.knowledge_index_versions SET revision = revision + 1 WHERE corpus = %L',
                    TG_TABLE_SCHEMA, 'knowledge');
                RETURN NULL;
            END;
            $$
        """))
        for table in tables:
            # Identifiers are fixed application constants, never user input.
            connection.execute(text(f"DROP TRIGGER IF EXISTS knowledge_revision_change ON {table}"))
            connection.execute(text(f"""
                CREATE TRIGGER knowledge_revision_change
                AFTER INSERT OR UPDATE OR DELETE OR TRUNCATE ON {table}
                FOR EACH STATEMENT EXECUTE FUNCTION bump_knowledge_index_revision()
            """))
    elif dialect == "sqlite":
        # Also allow isolated, file-backed unit tests without PostgreSQL.
        for table in tables:
            for operation in ("INSERT", "UPDATE", "DELETE"):
                connection.execute(text(f"""
                    CREATE TRIGGER IF NOT EXISTS {table}_revision_{operation.lower()}
                    AFTER {operation} ON {table}
                    BEGIN
                        UPDATE knowledge_index_versions SET revision = revision + 1
                        WHERE corpus = 'knowledge';
                    END
                """))
    else:
        raise RuntimeError("Knowledge versioning requires PostgreSQL or SQLite")
