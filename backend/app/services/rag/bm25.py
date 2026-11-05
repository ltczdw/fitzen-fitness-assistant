from __future__ import annotations

import re

from collections import Counter
from math import log


def tokenize(text: str) -> list[str]:
    """保留词频的中英文分词；中文同时加入相邻二元词。"""
    normalized = text.lower()
    words = re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", normalized)
    bigrams = [
        normalized[index : index + 2]
        for index in range(len(normalized) - 1)
        if "\u4e00" <= normalized[index] <= "\u9fff"
        and "\u4e00" <= normalized[index + 1] <= "\u9fff"
    ]
    return words + bigrams


def query_terms(text: str) -> list[str]:
    """查询词项的单一权威规则（G003）：重叠中文二元词 + 完整拉丁/数字串。

    与 ``tokenize`` 的二元词口径一致（逐字符滑窗，不按奇偶偏移丢词），供
    候选过滤（rag/retrieval.py）与覆盖率评估（rag/quality.py）共用；单字
    仅在查询本身为单字时输出，避免放开单字匹配制造虚假高召回。
    """
    normalized = text.lower()
    terms: list[str] = []
    for run in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", normalized):
        if "\u4e00" <= run[0] <= "\u9fff":
            if len(run) == 1:
                terms.append(run)
            else:
                terms.extend(run[index : index + 2] for index in range(len(run) - 1))
        else:
            terms.append(run)
    return terms


class BM25Index:
    """轻量 BM25Okapi 实现，避免为 MVP 引入额外运行时依赖。"""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.document_tokens = [tokenize(document) for document in documents]
        self.document_lengths = [len(tokens) for tokens in self.document_tokens]
        self.average_document_length = sum(self.document_lengths) / max(len(self.document_lengths), 1)
        self.document_frequency: Counter[str] = Counter()
        for tokens in self.document_tokens:
            self.document_frequency.update(set(tokens))
        document_count = len(self.document_tokens)
        self.inverse_document_frequency = {
            token: log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in self.document_frequency.items()
        }

    def score(self, query: str, document_index: int) -> float:
        query_tokens = set(tokenize(query))
        frequencies = Counter(self.document_tokens[document_index])
        document_length = self.document_lengths[document_index]
        normalization = self.k1 * (1 - self.b + self.b * document_length / max(self.average_document_length, 1))
        score = 0.0
        for token in query_tokens:
            term_frequency = frequencies.get(token, 0)
            if not term_frequency:
                continue
            idf = self.inverse_document_frequency.get(token, 0.0)
            score += idf * (term_frequency * (self.k1 + 1)) / (term_frequency + normalization)
        return score

    def rank(self, query: str, top_k: int) -> list[tuple[int, float]]:
        ranked = [(index, self.score(query, index)) for index in range(len(self.document_tokens))]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
