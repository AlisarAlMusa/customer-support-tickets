# ------------------------------------------------
# clean data 
# - remove URLs, mentions, and extra whitespace
# - handle null-like values
# 
# build chunks
# - pair customer tweets with company responses 
#   - it drops customer tweets without responses and company tweets without customer queries)
#   - it drops duplicates and empty texts
# - create metadata for each chunk
#   - metadata should include source, company, created_at, and tweet IDs
# 
# expected output: list of dicts with 'text' and 'metadata'
# ------------------------------------------------


import logging
import re
from typing import Any

import pandas as pd


SOURCE_NAME = "customer support data from twitter"
NULL_LIKE_VALUES = {"", "nan", "none", "null", "na", "n/a", "<na>"}
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
SPACE_PATTERN = re.compile(r"\s+")
logger = logging.getLogger(__name__)


def clean_text(value: Any, normalize_mentions: bool = False) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if text.lower() in NULL_LIKE_VALUES:
        return ""

    text = URL_PATTERN.sub("", text)

    if normalize_mentions:
        text = MENTION_PATTERN.sub("@user", text)

    return SPACE_PATTERN.sub(" ", text).strip()


def build_issue_response_chunks(
    df: pd.DataFrame,
    normalize_mentions: bool = False,
) -> list[dict[str, Any]]:
    raw_rows_count = len(df)
    tweets_by_id = df.set_index("tweet_id").to_dict(orient="index")
    chunks: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    matched_pairs_count = 0

    customer_rows = df[df["inbound"] == True]  # noqa: E712
    response_rows = df[df["inbound"] == False]  # noqa: E712
    customer_rows_count = len(customer_rows)
    response_rows_count = len(response_rows)

    logger.info("Chunking stats: raw_rows_count=%s", raw_rows_count)
    logger.info("Chunking stats: customer_rows_count=%s", customer_rows_count)
    logger.info("Chunking stats: response_rows_count=%s", response_rows_count)

    for _, customer_row in customer_rows.iterrows():
        customer_text = clean_text(customer_row["text"], normalize_mentions=normalize_mentions)
        if not customer_text:
            continue

        response_ids = _parse_response_ids(customer_row.get("response_tweet_id"))
        if not response_ids:
            continue

        for response_id in response_ids:
            response_row = tweets_by_id.get(response_id)
            if not response_row or bool(response_row.get("inbound")):
                continue

            response_text = clean_text(response_row.get("text"), normalize_mentions=normalize_mentions)
            if not response_text:
                continue

            matched_pairs_count += 1
            pair_key = (customer_text, response_text)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            chunks.append(
                {
                    "id": f"chunk_{len(chunks)}",
                    "text": customer_text,
                    "metadata": {
                        "response_text": response_text,
                        "source": SOURCE_NAME,
                        "company": response_row.get("author_id"),
                        "created_at": customer_row.get("created_at"),
                        "customer_tweet_id": customer_row.get("tweet_id"),
                        "company_response_id": response_id,
                    },
                }
            )

    logger.info("Chunking stats: matched_pairs_count=%s", matched_pairs_count)
    logger.info("Chunking stats: final_chunks_count=%s", len(chunks))

    return chunks


def _parse_response_ids(value: Any) -> list[int]:
    if pd.isna(value):
        return []

    ids = []
    for item in str(value).split(","):
        item = item.strip()
        if not item or item.lower() in NULL_LIKE_VALUES:
            continue
        try:
            ids.append(int(float(item)))
        except ValueError:
            continue

    return ids
