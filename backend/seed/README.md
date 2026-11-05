# Public reference seed

`official_documents.py` contains original Chinese summaries with publisher and source URLs. It contains no user conversations, workouts, meals or personal records. These summaries also support the in-memory lexical fallback; optional import into PostgreSQL: `python -m scripts.ingest_official_knowledge` from backend after schema initialization. The structured nutrition reference table remains application lookup data, not demonstration user records.
