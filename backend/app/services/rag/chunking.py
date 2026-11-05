"""Markdown sections, intact table rows and whole-sentence overlap."""
from dataclasses import dataclass
import re

CHUNKER_VERSION = "sections-v2"


@dataclass(frozen=True)
class TextChunk:
    content: str
    section_path: str
    content_type: str


def _prose(text, size, overlap):
    sentences = [s.strip() for s in re.split(r"(?<=[。！？!?；;])\s*|(?<=\.)\s+|\n+", text) if s.strip()]
    parts, current = [], []
    for sentence in sentences:
        # Only an individual oversized sentence is hard-split; do not fabricate
        # a partial-sentence overlap for these exceptional fragments.
        if len(sentence) > size:
            if current:
                parts.append("\n".join(current))
                current = []
            parts.extend(sentence[i:i + size] for i in range(0, len(sentence), size))
            continue
        if current and len("\n".join(current + [sentence])) > size:
            parts.append("\n".join(current))
            tail = []
            for previous in reversed(current):
                if len("\n".join([previous] + tail)) > overlap:
                    break
                tail.insert(0, previous)
            while tail and len("\n".join(tail + [sentence])) > size:
                tail.pop(0)
            current = tail
        current.append(sentence)
    if current:
        parts.append("\n".join(current))
    return parts


def split_document(text, *, title="", chunk_size=500, overlap=80, table_max_rows=10):
    if chunk_size < 1 or overlap < 0 or table_max_rows < 1:
        raise ValueError("Invalid chunk limits")
    overlap = min(overlap, chunk_size - 1)
    lines = text.splitlines()
    headings, prose, output = [], [], []
    fence = None

    def path():
        return " / ".join(([title] if title else []) + [h[1] for h in headings])

    def flush():
        if prose:
            output.extend(TextChunk(piece, path(), "text")
                          for piece in _prose("\n".join(prose), chunk_size, overlap))
            prose.clear()

    i = 0
    separator = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$")
    while i < len(lines):
        line = lines[i]
        marker_match = re.match(r"^\s*(`{3,}|~{3,})(.*)$", line)
        if fence is not None:
            if marker_match:
                marker, suffix = marker_match.groups()
                if marker[0] == fence[0] and len(marker) >= fence[1] and not suffix.strip():
                    fence = None
            prose.append(line)
            i += 1
            continue
        if marker_match:
            marker, _suffix = marker_match.groups()
            fence = (marker[0], len(marker))
            prose.append(line)
            i += 1
            continue
        header = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line) if fence is None else None
        if header:
            flush()
            level = len(header[1])
            headings = [h for h in headings if h[0] < level]
            headings.append((level, header[2]))
            i += 1
        elif fence is None and i + 1 < len(lines) and "|" in line and separator.match(lines[i + 1]):
            flush()
            table_header = [line, lines[i + 1]]
            i += 2
            rows = []
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                candidate = "\n".join(table_header + rows + [lines[i]])
                if rows and (len(rows) >= table_max_rows or len(candidate) > chunk_size):
                    output.append(TextChunk("\n".join(table_header + rows), path(), "table"))
                    rows = []
                rows.append(lines[i])
                i += 1
            output.append(TextChunk("\n".join(table_header + rows), path(), "table"))
        else:
            prose.append(line)
            i += 1
    flush()
    return output
