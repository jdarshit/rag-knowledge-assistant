from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class AskRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []
    top_k: int | None = None
    max_distance: float | None = None


class Source(BaseModel):
    source: str
    page: int
    distance: float
    rerank_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[Source] = []
    search_query: str | None = None
    closest_distance: float | None = None


class DocumentInfo(BaseModel):
    name: str
    chunks: int


class DocumentsResponse(BaseModel):
    documents: list[DocumentInfo]