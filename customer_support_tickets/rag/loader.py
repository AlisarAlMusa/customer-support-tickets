from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "tweet_id",
    "author_id",
    "inbound",
    "created_at",
    "text",
    "response_tweet_id",
    "in_response_to_tweet_id",
}


def load_dataset(
    csv_path: str | Path,
    sample_size: int = 300_000,
    random_state: int = 42,
    required_columns: set[str] | None = None,
) -> pd.DataFrame:
    path = Path(csv_path)
    columns = required_columns or REQUIRED_COLUMNS

    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    missing_columns = columns - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Dataset is missing required column(s): {missing}")

    if sample_size and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_state)

    return df.reset_index(drop=True)
