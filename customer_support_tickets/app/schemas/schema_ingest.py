from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    dataset_path: str
    rows_loaded: int
    chunks_created: int
    batches_processed: int
    skipped_existing_chunks: int
    embeddings_created: int
    collection_name: str
    stored_count: int
