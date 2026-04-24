import logging
from collections.abc import Iterable

from fastapi import APIRouter

from app.config import settings
from app.schemas.schema_ingest import IngestResponse
from rag.chunker import build_issue_response_chunks
from rag.embedder import embed_chunk_customer_texts
from rag.loader import load_dataset
from rag.store import get_existing_ids, store_chunks


router = APIRouter(prefix="/ingest", tags=["ingest"])
logger = logging.getLogger(__name__)


@router.post("/", response_model=IngestResponse)
def ingest_tickets():
    logger.info("Ingest started")
    logger.info("Loading dataset from %s with sample size %s", settings.dataset_path, settings.rag_sample_size)
    df = load_dataset(
        settings.dataset_path,
        sample_size=settings.rag_sample_size,
        random_state=42,
    )
    logger.info("Dataset loaded: %s rows", len(df))

    logger.info("Building chunks")
    chunks = build_issue_response_chunks(df, normalize_mentions=False)
    logger.info("Chunks built: %s", len(chunks))

    total_embeddings_created = 0
    total_stored = 0
    batches_processed = 0
    skipped_existing_chunks = 0
    collection_name = settings.chroma_collection_name

    for batch_number, chunk_batch in enumerate(_batched(chunks, settings.ingestion_batch_size), start=1):
        logger.info(
            "Processing batch %s: %s chunks",
            batch_number,
            len(chunk_batch),
        )
        batch_ids = [chunk["id"] for chunk in chunk_batch]
        existing_ids = get_existing_ids(batch_ids, collection_name=collection_name)

        if existing_ids:
            logger.info(
                "Batch %s already has %s stored chunks; skipping them",
                batch_number,
                len(existing_ids),
            )

        chunks_to_store = [chunk for chunk in chunk_batch if chunk["id"] not in existing_ids]
        skipped_existing_chunks += len(existing_ids)

        if not chunks_to_store:
            logger.info("Batch %s skipped entirely; all chunks already exist", batch_number)
            batches_processed = batch_number
            continue

        embeddings = embed_chunk_customer_texts(chunks_to_store)
        logger.info("Batch %s embeddings generated: %s", batch_number, len(embeddings))

        store_result = store_chunks(
            chunks=chunks_to_store,
            embeddings=embeddings,
            collection_name=collection_name,
        )
        logger.info("Batch %s stored: %s chunks", batch_number, store_result["stored_count"])

        total_embeddings_created += len(embeddings)
        total_stored += store_result["stored_count"]
        batches_processed = batch_number

    logger.info(
        "Ingest completed: %s batches, %s embeddings, %s stored",
        batches_processed,
        total_embeddings_created,
        total_stored,
    )
    # results logged: 319 batches, 159122 embeddings, 159122 stored

    return IngestResponse(
        status="completed",
        dataset_path=settings.dataset_path,
        rows_loaded=len(df),
        chunks_created=len(chunks),
        batches_processed=batches_processed,
        skipped_existing_chunks=skipped_existing_chunks,
        embeddings_created=total_embeddings_created,
        collection_name=collection_name,
        stored_count=total_stored,
    )


def _batched(items: list[dict], batch_size: int) -> Iterable[list[dict]]:
    if batch_size <= 0:
        raise ValueError("INGESTION_BATCH_SIZE must be greater than 0")

    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


# some logic to be explained:
# before embedding each batch, it now:
# collects that batch’s chunk IDs
# asks Chroma which IDs already exist
# filters them out
# embeds and stores only the missing chunks
# if a whole batch is already stored, it skips that batch entirely
# added logs for skipped existing chunks
