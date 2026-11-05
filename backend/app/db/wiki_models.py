from datetime import datetime
from sqlalchemy import String,Text,Integer,DateTime,JSON,ForeignKey,UniqueConstraint,func
from sqlalchemy.orm import Mapped,mapped_column
from app.db.models import Base,Vector

class WikiPage(Base):
    __tablename__='wiki_pages'
    page_id:Mapped[str]=mapped_column(String(36),primary_key=True)
    path:Mapped[str]=mapped_column(String(240),unique=True)
    title:Mapped[str]=mapped_column(String(200))
    parent_id:Mapped[str|None]=mapped_column(ForeignKey('wiki_pages.page_id'),nullable=True,index=True)
    kind:Mapped[str]=mapped_column(String(30),default='memory')
    source_turn_id:Mapped[str|None]=mapped_column(String(36),nullable=True,unique=True)
    content:Mapped[str]=mapped_column(Text,default='')
    summary:Mapped[str]=mapped_column(Text,default='')
    summary_embedding:Mapped[list|None]=mapped_column(Vector(1024),nullable=True)
    embedding_version:Mapped[str|None]=mapped_column(String(100),nullable=True)
    revision:Mapped[int]=mapped_column(Integer,default=1)
    memory_epoch:Mapped[int]=mapped_column(Integer,default=0)
    updated_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now(),onupdate=func.now())
    deleted_at:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)

class WikiSection(Base):
    __tablename__='wiki_sections'
    section_id:Mapped[str]=mapped_column(String(36),primary_key=True)
    page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    title:Mapped[str]=mapped_column(String(200))
    content:Mapped[str]=mapped_column(Text)
    position:Mapped[int]=mapped_column(Integer)
    source_message_id:Mapped[str|None]=mapped_column(String(100),nullable=True)

class WikiEvidence(Base):
    __tablename__='wiki_evidence'
    evidence_id:Mapped[str]=mapped_column(String(36),primary_key=True)
    page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    section_id:Mapped[str]=mapped_column(ForeignKey('wiki_sections.section_id'))
    turn_id:Mapped[str]=mapped_column(String(36),index=True)
    message_id:Mapped[str]=mapped_column(String(100))
    source_hash:Mapped[str]=mapped_column(String(64))

class WikiFact(Base):
    __tablename__='wiki_facts'
    fact_id:Mapped[str]=mapped_column(String(36),primary_key=True)
    page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    field:Mapped[str]=mapped_column(String(160))
    value:Mapped[str]=mapped_column(Text)
    source_message_id:Mapped[str]=mapped_column(String(100))
    status:Mapped[str]=mapped_column(String(20),default='active')
    valid_from:Mapped[datetime]=mapped_column(DateTime(timezone=True))
    valid_to:Mapped[datetime|None]=mapped_column(DateTime(timezone=True),nullable=True)
    supersedes:Mapped[str|None]=mapped_column(String(36),nullable=True)

class WikiRevision(Base):
    __tablename__='wiki_revisions'
    __table_args__=(UniqueConstraint('page_id','revision'),)
    id:Mapped[str]=mapped_column(String(36),primary_key=True)
    page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    revision:Mapped[int]=mapped_column(Integer)
    snapshot:Mapped[dict]=mapped_column(JSON)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),server_default=func.now())

class WikiLink(Base):
    __tablename__='wiki_links'
    __table_args__=(UniqueConstraint('from_page_id','to_page_id','relation'),)
    id:Mapped[str]=mapped_column(String(36),primary_key=True)
    from_page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    to_page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
    relation:Mapped[str]=mapped_column(String(40))

class WikiAlias(Base):
    __tablename__='wiki_aliases'
    alias:Mapped[str]=mapped_column(String(240),primary_key=True)
    page_id:Mapped[str]=mapped_column(ForeignKey('wiki_pages.page_id'),index=True)
