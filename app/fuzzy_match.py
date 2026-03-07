import pandas as pd
from rapidfuzz import process, fuzz
from pathlib import Path

_artists_df: pd.DataFrame | None = None
_artist_names: list[str] = []

DATA_PATH = Path(__file__).parent.parent / "data" / "artists.csv"


def _load():
    global _artists_df, _artist_names
    if _artists_df is None:
        _artists_df = pd.read_csv(DATA_PATH)
        _artist_names = _artists_df["name"].tolist()


def get_artist_names() -> list[str]:
    _load()
    return _artist_names


def match_artist(name: str, threshold: int = 80) -> dict | None:
    """
    Fuzzy-match `name` against the 50 artists in artists.csv.
    Returns the matching row as a dict, or None if no match above threshold.
    """
    _load()
    result = process.extractOne(name, _artist_names, scorer=fuzz.token_sort_ratio)
    if result is None:
        return None
    matched_name, score, _ = result
    if score < threshold:
        return None
    row = _artists_df[_artists_df["name"] == matched_name].iloc[0]
    return row.to_dict()
