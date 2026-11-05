import json
from types import SimpleNamespace
from dataclasses import dataclass,field
import httpx
import pytest
from app.services.rag.chunking import split_document
from app.services.rag.model_reranker import ModelReranker,RerankUnavailable
from app.core.config import settings

def test_heading_paths_and_whole_sentence_overlap():
 chunks=split_document('# 第一章\n第一句话。第二句话。第三句话。\n## 小节\n另一句话。\n# 第二章\n新内容。',title='指南',chunk_size=12,overlap=6)
 assert chunks[0].section_path=='指南 / 第一章'
 assert chunks[0].content.endswith('第二句话。') and chunks[1].content.startswith('第二句话。')
 assert chunks[-2].section_path=='指南 / 第一章 / 小节'
 assert chunks[-1].section_path=='指南 / 第二章'
 assert all(len(c.content)<=12 for c in chunks)

def test_table_splits_repeat_headers_keep_rows_and_section():
 header='|动作|次数|\n|---|---|'
 rows=[f'|动作{i}|{i}次|' for i in range(23)]
 chunks=split_document('# 计划\n'+header+'\n'+'\n'.join(rows),title='指南',chunk_size=10000)
 assert len(chunks)==3
 assert all(c.content.startswith(header) and c.content_type=='table' and c.section_path=='指南 / 计划' for c in chunks)
 assert [r for c in chunks for r in c.content.splitlines()[2:]]==rows

def test_english_words_and_fenced_headings_are_preserved():
 chunks=split_document('First sentence. Second sentence.\n```text\n# not a heading\n```',chunk_size=200)
 assert 'sentence.Second' not in chunks[0].content
 assert '# not a heading' in chunks[0].content and chunks[0].section_path==''

@dataclass
class Hit:
 title:str='title'
 content:str='content'
 score:float=0
 metadata:dict=field(default_factory=dict)

@pytest.mark.parametrize('provider',['dashscope','jina_cohere'])
def test_independent_rerank_credentials_and_wire_protocol(monkeypatch,provider):
 monkeypatch.setattr(settings,'rerank_enabled',True)
 monkeypatch.setattr(settings,'rerank_provider',provider)
 monkeypatch.setattr(settings,'rerank_base_url','https://rerank.invalid/v2')
 monkeypatch.setattr(settings,'rerank_api_key','rerank-only')
 monkeypatch.setattr(settings,'retrieval_embedding_api_key','embedding-only')
 original=httpx.Client;seen=[]
 def handle(request):
  seen.append(request);payload=json.loads(request.content)
  assert request.headers['authorization']=='Bearer rerank-only'
  values=[{'index':0,'relevance_score':.9}]
  if provider=='dashscope':
   assert payload['input']['documents']==['title\ncontent'];assert '/api/v1/services/rerank/' in str(request.url)
   return httpx.Response(200,json={'output':{'results':values}})
  assert payload['documents']==['title\ncontent'] and str(request.url).endswith('/v2/rerank')
  return httpx.Response(200,json={'results':values})
 monkeypatch.setattr(httpx,'Client',lambda **kw:original(transport=httpx.MockTransport(handle),**kw))
 assert ModelReranker().rerank('query',[Hit()])[0].score==.9 and len(seen)==1

def test_partial_rerank_config_does_not_borrow_embedding_key(monkeypatch):
 monkeypatch.setattr(settings,'rerank_enabled',True)
 monkeypatch.setattr(settings,'rerank_base_url','https://rerank.invalid/v1')
 monkeypatch.setattr(settings,'rerank_api_key','')
 monkeypatch.setattr(settings,'retrieval_embedding_api_key','secret-other-provider')
 with pytest.raises(RerankUnavailable,match='not_configured'):ModelReranker().rerank('q',[Hit()])
