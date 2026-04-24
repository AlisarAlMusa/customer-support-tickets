from pydantic import BaseModel


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


class RetrievedTicket(BaseModel):
    text: str
    score: float
    distance: float | None = None
    response_text: str | None = None
    company: str | None = None
    source: str | None = None
    created_at: str | None = None
    customer_tweet_id: int | None = None
    company_response_id: int | None = None


class RetrieveResponse(BaseModel):
    query: str
    results: list[RetrievedTicket]
