"""Explicit wire protocols. A rerank endpoint is not an OpenAI chat API."""
from urllib.parse import urlsplit


class DashScopeProtocol:
    def endpoint(self, base):
        parsed = urlsplit(base)
        return f"{parsed.scheme}://{parsed.netloc}/api/v1/services/rerank/text-rerank/text-rerank"

    def payload(self, model, query, documents):
        return {"model": model, "input": {"query": query, "documents": documents},
                "parameters": {"top_n": len(documents)}}

    def results(self, data):
        return data["output"]["results"]


class JinaCohereProtocol:
    def endpoint(self, base):
        # The configured base includes /v1 or /v2, or the full /rerank URL.
        base = base.rstrip("/")
        return base if base.endswith("/rerank") else base + "/rerank"

    def payload(self, model, query, documents):
        return {"model": model, "query": query, "documents": documents, "top_n": len(documents)}

    def results(self, data):
        return data["results"]


PROTOCOLS = {"dashscope": DashScopeProtocol(), "jina_cohere": JinaCohereProtocol()}
