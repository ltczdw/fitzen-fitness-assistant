"""Import official summaries using the parent-document/child-chunk pipeline."""

from app.db import session as db_session
from app.services.rag.ingestion import KnowledgeIngestionService
from app.services.rag.official_documents import OFFICIAL_DOCUMENTS


def main() -> int:
    if not db_session.init_db() or db_session.SessionLocal is None:
        print("数据库不可用：请先确认 PostgreSQL 和 SSH 隧道已启动。")
        return 2

    service = KnowledgeIngestionService()
    with db_session.SessionLocal() as session:
        result = service.ingest_documents(session, OFFICIAL_DOCUMENTS)
        session.commit()

    print(
        "官方知识父子切片入库完成："
        f"documents_added={result.documents_added}, "
        f"chunks_added={result.chunks_added}, "
        f"skipped_exact={result.skipped_exact}, "
        f"skipped_similar={result.skipped_similar}"
    )
    for document in OFFICIAL_DOCUMENTS:
        print(f"- {document.title} [{document.source}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
