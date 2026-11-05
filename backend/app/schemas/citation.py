"""Public provenance DTO; do not expose arbitrary retrieval metadata."""
from urllib.parse import urlsplit
from pydantic import BaseModel, field_validator


class Citation(BaseModel):
    title: str
    source: str
    chunk_id: str | None = None
    page: str | int | None = None
    url: str | None = None

    @field_validator("url", mode="before")
    @classmethod
    def safe_url(cls, value):
        if not isinstance(value, str):
            return None
        parsed = urlsplit(value)
        return value if parsed.scheme in {"http", "https"} and parsed.hostname and not parsed.username and not parsed.password else None


def normalize_citations(items) -> list[dict]:
    """Accept historical title-only cache entries while always emitting objects."""
    output = []
    for item in items or []:
        if isinstance(item, str):
            item = {"title": item, "source": item}
        elif isinstance(item, Citation):
            item = item.model_dump()
        value = Citation.model_validate(item).model_dump()
        if value not in output:
            output.append(value)
    return output


def citations_from_hits(hits) -> list[dict]:
    return normalize_citations([{
        "title": hit.title, "source": hit.source, "chunk_id": hit.chunk_id,
        "page": hit.metadata.get("page", hit.metadata.get("page_number")),
        "url": hit.metadata.get("url") or (hit.source if hit.source.startswith(("https://", "http://")) else None),
    } for hit in hits])
