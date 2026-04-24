from fastapi import APIRouter

from app.config import settings
from rag.chunker import build_issue_response_chunks
from rag.embedder import build_chunk_embedding_pairs
from rag.loader import load_dataset
from app.schemas.schema_retrieve import RetrieveRequest
from app.services.rag_service import retrieve_similar_tickets


router = APIRouter(prefix="/validate", tags=["validate"])
VALIDATION_EMBEDDING_CHUNK_COUNT = 5 
#The validator only embeds the first 5 chunks from the chunker output, so the API response doesn’t become massive.


@router.get("/chunker")
def validate_chunker():
    df = load_dataset(settings.dataset_path, sample_size=settings.validation_sample_size, random_state=42)
    chunks = build_issue_response_chunks(df)

    return {
        "passed": True,
        "input_rows": len(df),
        "chunks_created": len(chunks),
        "expected_chunks": None,
        "sample_chunks": chunks,
    }


@router.get("/embedder")
def validate_embedder():
    df = load_dataset(settings.dataset_path, sample_size=settings.validation_sample_size, random_state=42)
    chunks = build_issue_response_chunks(df)
    chunks_to_embed = chunks[:VALIDATION_EMBEDDING_CHUNK_COUNT]
    chunk_embedding_pairs = build_chunk_embedding_pairs(chunks_to_embed)

    return {
        "passed": len(chunk_embedding_pairs) == len(chunks_to_embed),
        "input_rows": len(df),
        "chunks_used": len(chunks_to_embed),
        "chunk_embedding_pairs": chunk_embedding_pairs,
    }

# this endpoint validate the full retrieval flow:
# from embedding the query to returning similar tickets, without involving the actual LLM generation step. 
@router.post("/retrieve")
def validate_retrieve(request: RetrieveRequest):
    results = retrieve_similar_tickets(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "top_k": request.top_k,
        "results_count": len(results),
        "results": [ticket.model_dump() for ticket in results],
    }
