"""Bounded official-source snapshots and bilingual discovery metadata."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.core import clock
from app.services.rag.authority_sources import is_authority_url
from app.services.rag.crawler import _PageParser
from app.services.rag.ingestion import SourceDocument, content_hash

AGENT = "FitnessAssistantKnowledgeBot/1.0 (user-authorized public reference import)"
OGL = "https://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/"


def safe_url(url):
    parsed = urlsplit(url)
    return bool(is_authority_url(url) and parsed.scheme == "https" and not parsed.username and not parsed.password
                and parsed.port in {None, 443} and not parsed.fragment)


def load_catalog(path):
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = set()
    for entry in entries:
        if (not isinstance(entry, dict) or not safe_url(entry.get("url", ""))
                or not entry.get("title_zh") or not entry.get("topic")
                or not entry.get("id") or entry["id"] in ids
                or not isinstance(entry.get("questions"), list) or not entry["questions"]):
            raise ValueError("invalid_official_source_catalog")
        ids.add(entry["id"])
    task_types = {'training_qa', 'nutrition_advice', 'safety_triage'}
    for entry in entries:
        if entry.get('task_type') and entry['task_type'] not in task_types:
            raise ValueError('invalid_coverage_task_type')
        for index, sources in entry.get('expected_sources', {}).items():
            if (not str(index).isdigit() or int(index) >= len(entry['questions'])
                    or not isinstance(sources, list) or not sources or not set(sources) <= ids):
                raise ValueError('invalid_coverage_expected_sources')
        for index, task_type in entry.get('question_task_types', {}).items():
            if not str(index).isdigit() or int(index) >= len(entry['questions']) or task_type not in task_types:
                raise ValueError('invalid_coverage_task_type')
    return entries


class SourceFetchError(ValueError):
    pass


def _get(client, url, host, *, rule=None, redirects=4):
    for _ in range(redirects + 1):
        if not safe_url(url) or urlsplit(url).hostname != host:
            raise SourceFetchError("redirect_outside_source_host")
        if rule is not None and not rule.can_fetch(AGENT, url):
            raise SourceFetchError("robots_disallowed")
        response = client.get(url)
        if response.status_code in {301, 302, 303, 307, 308}:
            location = response.headers.get("location")
            if not location:
                raise SourceFetchError("redirect_without_location")
            url = urljoin(url, location)
            continue
        response.raise_for_status()
        return response
    raise SourceFetchError("too_many_redirects")


def crawl_catalog(entries, *, client=None, delay=.4, progress=None):
    """Only explicit seed pages; no link traversal or unbounded host crawling."""
    owned = client is None
    client = client or httpx.Client(timeout=18, follow_redirects=False, trust_env=False,
                                   headers={"User-Agent": AGENT, "Accept": "text/html"})
    documents, failures, robots = [], [], {}
    try:
        for entry in entries:
            url = entry["url"]
            host = urlsplit(url).hostname
            try:
                if not safe_url(url):
                    raise SourceFetchError("unapproved_source_url")
                if host not in robots:
                    robot_url = "https://" + host + "/robots.txt"
                    response = client.get(robot_url)
                    if response.status_code == 404:
                        robots[host] = None
                    else:
                        response.raise_for_status()
                        rule = RobotFileParser(robot_url)
                        rule.parse(response.text.splitlines())
                        robots[host] = rule
                if robots[host] is not None and not robots[host].can_fetch(AGENT, url):
                    raise SourceFetchError("robots_disallowed")
                response = _get(client, url, host, rule=robots[host])
                if "html" not in response.headers.get("content-type", "").lower():
                    raise SourceFetchError("not_html")
                if len(response.content) > 2_000_000:
                    raise SourceFetchError("page_too_large")
                parser = _PageParser()
                parser.feed(response.text)
                body = parser.content()
                if not 200 <= len(body) <= 150_000:
                    raise SourceFetchError("invalid_article_length")
                original_title = " ".join(parser.title_parts).strip()
                title = entry["title_zh"] + " | " + original_title
                nhs = host == "www.nhs.uk"
                timestamp = clock.utc_now().isoformat()
                metadata = {
                    "url": str(response.url), "source_url": str(response.url), "requested_url": url,
                    "publisher": host, "authority_level": "official_public_health", "language": "en",
                    "title_language": "zh-en", "coverage_id": entry["id"], "topics": [entry["topic"]],
                    "retrieved_at": timestamp, "original_title": original_title,
                    "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                    "license": "Open Government Licence v3.0" if nhs else "US government health information; third-party exclusions apply",
                    "license_url": OGL if nhs else "https://www.nih.gov/about-nih/what-we-do/nih-almanac/disclaimer",
                    "attribution": "Information from the NHS website is licensed under the Open Government Licence v3.0" if nhs else "Source: " + host,
                    "jurisdiction": "UK" if nhs else "US", "snapshot_version": 1,
                }
                # The Chinese title is a discovery label, not invented medical
                # evidence. The full clinical body remains the fetched original.
                document = SourceDocument(title=title[:240], content=body,
                    source="official-coverage:" + entry["id"], metadata=metadata)
                documents.append({"id": entry["id"], **asdict(document)})
                if progress: progress({"id": entry["id"], "status": "fetched", "chars": len(body)})
            except (httpx.HTTPError, SourceFetchError, ValueError) as error:
                code = str(error) if isinstance(error, SourceFetchError) else type(error).__name__
                failures.append({"id": entry["id"], "url": url, "error": code,
                                 "http_status": getattr(getattr(error, "response", None), "status_code", None)})
                if progress: progress({"id": entry["id"], "status": "failed", "error": code})
            if delay:
                time.sleep(delay)
    finally:
        if owned: client.close()
    return {"version": 1, "retrieved_at": clock.utc_now().isoformat(), "documents": documents, "failures": failures}


def load_snapshot(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("version") != 1:
        raise ValueError("unknown_snapshot_version")
    documents = []
    seen = set()
    for item in raw["documents"]:
        metadata = item["metadata"]
        if (not item.get("id") or item["id"] in seen
                or metadata.get("coverage_id") != item["id"]
                or item.get("source") != "official-coverage:" + item["id"]
                or not item.get("title") or not metadata.get("topics")
                or not safe_url(metadata.get("url", ""))
                or not safe_url(metadata.get("requested_url", ""))
                or urlsplit(metadata["url"]).hostname != urlsplit(metadata["requested_url"]).hostname
                or hashlib.sha256(item["content"].encode()).hexdigest() != metadata.get("content_sha256")):
            raise ValueError("snapshot_source_or_hash_invalid")
        seen.add(item["id"])
        documents.append(SourceDocument(**{key: item[key] for key in ("title", "content", "source", "metadata")}))
        translation = item.get("translation")
        if translation:
            text_zh = translation.get("text_zh", "")
            if (translation.get("source_sha256") != metadata["content_sha256"]
                    or not isinstance(text_zh, str) or not 30 <= len(text_zh) <= 2000):
                raise ValueError("translation_source_invalid")
            # A separate, explicitly labelled derived document preserves the
            # original article's identity, body and existing vectors on retry.
            body = "# 项目中文摘译（非原机构审定译文）\n" + text_zh
            translated_metadata = {**metadata, "language": "zh", "content_origin": "project_translation_of_official_source",
                "source_content_sha256": metadata["content_sha256"],
                "content_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "translation_editor": translation.get("editor"), "translation_created_at": translation.get("created_at")}
            documents.append(SourceDocument(title=item["title"] + " | 中文摘译", content=body,
                source=item["source"] + ":zh", metadata=translated_metadata))
    return documents


def document_ids(documents):
    return ["doc-" + content_hash(document.title, document.content)[:24] for document in documents]
