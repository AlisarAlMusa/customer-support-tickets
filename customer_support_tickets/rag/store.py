from typing import Any, cast

from app.config import settings


MetadataValue = str | int | float | bool
ChromaMetadata = dict[str, MetadataValue]


def get_chroma_client():
    try:
        import chromadb
    except ImportError as exc:
        raise ImportError("ChromaDB package is missing. Install requirements.txt first.") from exc

    return chromadb.PersistentClient(path=settings.chroma_path)


def get_or_create_collection(name: str | None = None):
    client = get_chroma_client()
    collection_name = name or settings.chroma_collection_name
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )


def get_existing_ids(ids: list[str], collection_name: str | None = None) -> set[str]:
    if not ids:
        return set()

    collection = get_or_create_collection(collection_name)
    result = collection.get(
        ids=cast(Any, ids),
        include=cast(Any, []),
    )
    return set(result["ids"])


def store_chunks(
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    collection_name: str | None = None,
):
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")
    if not chunks:
        raise ValueError("No chunks provided for storage")

    collection = get_or_create_collection(collection_name)

    ids = [chunk["id"] for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas: list[ChromaMetadata] = [_prepare_metadata(chunk["metadata"]) for chunk in chunks]
    embeddings = [[float(v) for v in emb] for emb in embeddings]
    

    collection.add(
        ids=cast(Any, ids),
        documents=cast(Any, documents),
        embeddings=cast(Any, embeddings),
        metadatas=cast(Any, metadatas),
    )

    return {
        "collection_name": collection.name,
        "stored_count": len(chunks),
        "ids": ids,
    }


def _prepare_metadata(metadata: dict[str, Any]) -> ChromaMetadata:
    prepared: ChromaMetadata = {}
    for key, value in metadata.items():
        prepared[key] = _normalize_metadata_value(value)
    return prepared


def _normalize_metadata_value(value: Any) -> MetadataValue:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
