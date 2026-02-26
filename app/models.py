from pydantic import BaseModel
from typing import Optional


class ResearchRequest(BaseModel):
    topic: str
    session_id: Optional[str] = None


class ArxivSearchRequest(BaseModel):
    query: str
    max_results: int = 10
    sort_by: str = "relevance"        # relevance | date
    date_from: Optional[str] = None   # YYYY-MM-DD
    date_to: Optional[str] = None     # YYYY-MM-DD


class AddPaperRequest(BaseModel):
    session_id: str
    paper: dict   # {title, summary, url, published, authors}


class SemanticSearchRequest(BaseModel):
    session_id: str
    query: str
    n_results: int = 5
