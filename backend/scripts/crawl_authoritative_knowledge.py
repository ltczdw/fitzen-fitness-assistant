from __future__ import annotations

import argparse

from app.services.rag.authority_sources import AUTHORITY_SEEDS
from app.services.rag.crawler import AuthorityCrawler
from app.services.rag.pipeline import HybridRAGPipeline
from app.db.session import init_db


def main() -> None:
    parser = argparse.ArgumentParser(description="抓取预置权威健身/康复/营养/运动科学页面并按父子切片入库")
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    crawler = AuthorityCrawler(max_pages=args.max_pages, max_depth=args.max_depth)
    documents = crawler.crawl(AUTHORITY_SEEDS)
    print(f"crawled_documents={len(documents)}")
    for document in documents:
        print(f"- {document.title} | {document.metadata.get('url')}")
    if not args.dry_run and documents:
        if not init_db():
            raise SystemExit("database initialization failed; check DATABASE_URL and PostgreSQL/pgvector availability")
        result = HybridRAGPipeline().ingest_documents(documents)
        print(f"ingested_documents={result.documents_added} ingested_chunks={result.chunks_added} exact_skipped={result.skipped_exact} similar_skipped={result.skipped_similar}")


if __name__ == "__main__":
    main()
