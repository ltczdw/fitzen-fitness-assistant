import unittest

from app.services.rag.ingestion import chunk_text, clean_text, content_hash


class IngestionTests(unittest.TestCase):
    def test_clean_text_normalizes_markup_and_whitespace(self):
        self.assertEqual(clean_text(" <p>训练&nbsp;&nbsp;记录</p>\r\n\u200b "), "训练 记录")

    def test_chunk_text_respects_size_and_creates_multiple_children(self):
        chunks = chunk_text("训练恢复建议。" * 30, chunk_size=40, overlap=8)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk) <= 40 for chunk in chunks))

    def test_hash_is_stable_after_cleaning(self):
        self.assertEqual(content_hash("标题", "内容  "), content_hash(" 标题 ", "内容"))


if __name__ == "__main__":
    unittest.main()
