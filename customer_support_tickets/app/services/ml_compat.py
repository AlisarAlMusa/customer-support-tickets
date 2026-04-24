import __main__
import re

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


urgent_keywords = [
    "refund",
    "charged",
    "cancel",
    "broken",
    "not working",
    "down",
    "failed",
    "error",
    "critical",
    "unresolved",
    "quit",
    "money back",
    "legal",
    "cancellation",
    "manager",
    "supervisor",
    "urgent",
    "emergency",
    "asap",
    "locked out",
    "fraud",
    "dispute",
    "disappointed",
    "immediately"
]

escalation_keywords = [
    "still",
    "again",
    "waiting",
    "no response",
    "second time",
    "third time",
    "useless",
    "complicated",
    "difficult",
]

upset_keywords = [
    "fuck",
    "fucking",
    "freaking",
    "ridiculous",
    "annoying",
    "frustrating",
    "disappointed",
    "unacceptable",
    "terrible",
    "hate",
    "upset",
    "awful",
]


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def count_keywords(text, keywords):
    return sum(1 for kw in keywords if kw in str(text))


def count_exclamations(text):
    return str(text).count("!")


def count_questions(text):
    return str(text).count("?")


def caps_ratio(text):
    letters = [c for c in str(text) if c.isalpha()]
    if len(letters) == 0:
        return 0.0
    caps = sum(1 for c in letters if c.isupper())
    return caps / len(letters)


class TicketTextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        cleaned_data = _to_frame(X).copy()
        cleaned_data["text"] = cleaned_data["text"].astype(str).str.strip()
        cleaned_data["text_clean"] = cleaned_data["text"].apply(clean_text)
        return cleaned_data


class TicketFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = _to_frame(X).copy()
        features["word_count"] = features["text"].str.split().str.len()
        features["question_count"] = features["text"].apply(count_questions).clip(upper=10)
        features["exclamation_count"] = features["text"].apply(count_exclamations).clip(upper=10)
        features["caps_ratio"] = features["text"].apply(caps_ratio)
        features["urgent_kw_count"] = features["text_clean"].apply(lambda text: count_keywords(text, urgent_keywords))
        features["escalation_kw_count"] = features["text_clean"].apply(lambda text: count_keywords(text, escalation_keywords))
        features["upset_kw_count"] = features["text_clean"].apply(lambda text: count_keywords(text, upset_keywords))
        return features


def register_pickle_compat_classes() -> None:
    setattr(__main__, "TicketTextCleaner", TicketTextCleaner)
    setattr(__main__, "TicketFeatureEngineer", TicketFeatureEngineer)


def _to_frame(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    elif isinstance(X, pd.Series):
        frame = X.to_frame(name="text")
    else:
        frame = pd.DataFrame({"text": list(X)})

    if "text" not in frame.columns:
        first_column = frame.columns[0]
        frame = frame.rename(columns={first_column: "text"})

    return frame
