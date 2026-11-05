from __future__ import annotations
from app.core import clock

import hashlib
import re
from collections import deque
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urldefrag, urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import httpx

from app.core.config import settings
from app.services.rag.authority_sources import AUTHORITY_SEEDS, AuthoritySeed, is_authority_url
from app.services.rag.ingestion import SourceDocument


class _PageParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg", "nav", "footer", "header", "aside", "form"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []
        self._ignored_depth = 0
        self._in_title = False
        self._in_main = False
        self._main_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "main":
            self._in_main = True
        if self._in_main and self._ignored_depth == 0:
            if tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                self._main_parts.append("\n" + "#" * int(tag[1]) + " ")
            elif tag in {"p", "li", "div", "section", "br", "tr"}:
                self._main_parts.append("\n")
        if tag in self._ignored_tags:
            self._ignored_depth += 1
        if tag == "title":
            self._in_title = True
        if tag == "a":
            href = dict(attrs).get("href")
            if href:
                self.links.append(href)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._in_main and tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "div", "section", "tr"}:
            self._main_parts.append("\n")
        if tag == "main":
            self._in_main = False
        if tag == "title":
            self._in_title = False
        if tag in self._ignored_tags and self._ignored_depth:
            self._ignored_depth -= 1

    def handle_data(self, data: str) -> None:
        cleaned = re.sub(r"\s+", " ", data).strip()
        if not cleaned:
            return
        if self._in_title:
            self.title_parts.append(cleaned)
        if self._ignored_depth == 0:
            self.text_parts.append(cleaned)
            if self._in_main:
                self._main_parts.append(cleaned + " ")

    def content(self) -> str:
        if self._main_parts:
            return "\n".join(line.strip() for line in "".join(self._main_parts).splitlines() if line.strip())
        return " ".join(self.text_parts).strip()


class AuthorityCrawler:
    """只抓取预置权威域名，同域、有限深度、有限页数并尊重 robots.txt。"""

    def __init__(
        self,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
        timeout_seconds: float | None = None,
        user_agent: str = "FitnessAssistantKnowledgeBot/1.0 (+authorized-knowledge-import)",
    ) -> None:
        self.max_pages = max(1, max_pages or settings.rag_crawl_max_pages)
        self.max_depth = max(0, max_depth if max_depth is not None else settings.rag_crawl_max_depth)
        self.timeout = timeout_seconds or settings.rag_crawl_timeout_seconds
        self.user_agent = user_agent
        self._robots: dict[str, RobotFileParser | None] = {}

    @staticmethod
    def _normalize_url(url: str) -> str:
        clean, _ = urldefrag(url.strip())
        parsed = urlsplit(clean)
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path or "/", parsed.query, ""))

    @staticmethod
    def _same_authority_host(left: str, right: str) -> bool:
        left_host = (urlsplit(left).hostname or "").lower().rstrip(".")
        right_host = (urlsplit(right).hostname or "").lower().rstrip(".")
        return left_host == right_host

    def _robots_allowed(self, client: httpx.Client, url: str) -> bool:
        parsed = urlsplit(url)
        host = (parsed.hostname or "").lower()
        if host not in self._robots:
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            try:
                response = client.get(robots_url, timeout=self.timeout)
                if response.status_code >= 400:
                    self._robots[host] = None
                else:
                    parser = RobotFileParser()
                    parser.set_url(robots_url)
                    parser.parse(response.text.splitlines())
                    self._robots[host] = parser
            except httpx.HTTPError:
                # A missing/unreachable robots endpoint is not a reason to
                # invent a deny rule; the crawler remains domain and page bound.
                self._robots[host] = None
        parser = self._robots[host]
        return parser is None or parser.can_fetch(self.user_agent, url)

    def crawl(self, seeds: tuple[AuthoritySeed, ...] = AUTHORITY_SEEDS) -> list[SourceDocument]:
        queue: deque[tuple[AuthoritySeed, str, int]] = deque()
        for seed in seeds:
            if is_authority_url(seed.url):
                queue.append((seed, self._normalize_url(seed.url), 0))
        visited: set[str] = set()
        documents: list[SourceDocument] = []
        headers = {"User-Agent": self.user_agent, "Accept": "text/html,application/xhtml+xml"}
        with httpx.Client(follow_redirects=True, headers=headers, timeout=self.timeout) as client:
            while queue and len(visited) < self.max_pages:
                seed, url, depth = queue.popleft()
                if url in visited or depth > self.max_depth or not is_authority_url(url):
                    continue
                visited.add(url)
                if not self._robots_allowed(client, url):
                    continue
                try:
                    response = client.get(url)
                    response.raise_for_status()
                except httpx.HTTPError:
                    continue
                final_url = self._normalize_url(str(response.url))
                if not is_authority_url(final_url) or not self._same_authority_host(seed.url, final_url):
                    continue
                content_type = response.headers.get("content-type", "").lower()
                if "html" not in content_type:
                    continue
                parser = _PageParser()
                try:
                    parser.feed(response.text)
                except (ValueError, TypeError):
                    continue
                content = parser.content()
                if len(content) >= 200:
                    title = " ".join(parser.title_parts).strip() or seed.name
                    metadata = {
                        "url": final_url,
                        "source_url": final_url,
                        "publisher": urlsplit(final_url).hostname,
                        "authority_level": "official_or_national_medical_library",
                        "topics": list(seed.topics),
                        "crawl_depth": depth,
                        "retrieved_at": clock.utc_now().isoformat(),
                        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                    }
                    documents.append(SourceDocument(title=title[:240], content=content[:100_000], source="authority-crawl", metadata=metadata))
                if depth >= self.max_depth:
                    continue
                for href in parser.links:
                    next_url = self._normalize_url(urljoin(final_url, href))
                    if self._same_authority_host(seed.url, next_url) and is_authority_url(next_url) and next_url not in visited:
                        queue.append((seed, next_url, depth + 1))
        return documents
