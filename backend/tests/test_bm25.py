import unittest

from app.services.rag.bm25 import BM25Index, tokenize


class BM25Tests(unittest.TestCase):
    def test_tokenizer_preserves_repeated_terms(self):
        tokens = tokenize("蛋白质 蛋白质 训练")
        self.assertGreaterEqual(tokens.count("蛋"), 2)

    def test_bm25_ranks_matching_document_first(self):
        index = BM25Index(["训练后轻微酸痛恢复", "蛋白质和热量规划", "训练计划和渐进超负荷"])
        ranked = index.rank("酸痛恢复", top_k=3)
        self.assertEqual(ranked[0][0], 0)
        self.assertGreater(ranked[0][1], ranked[1][1])
