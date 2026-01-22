import ast
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "Data")
CACHE_DIR  = os.path.join(BASE_DIR, "cache")

MOVIES_CSV  = os.path.join(DATA_DIR, "tmdb_5000_movies.csv")
CREDITS_CSV = os.path.join(DATA_DIR, "tmdb_5000_credits.csv")

MATRIX_PATH = os.path.join(CACHE_DIR, "cosine_sim.pkl")
INDEX_PATH  = os.path.join(CACHE_DIR, "title_index.pkl")
DF_PATH     = os.path.join(CACHE_DIR, "movies_df.pkl")

os.makedirs(CACHE_DIR, exist_ok=True)


def _parse_names(json_str: str, max_items: int | None = None) -> list[str]:
    try:
        items = ast.literal_eval(json_str)
        names = [item["name"] for item in items if "name" in item]
        return names[:max_items] if max_items else names
    except (ValueError, TypeError):
        return []


def _get_director(json_str: str) -> str:
    try:
        crew = ast.literal_eval(json_str)
        for member in crew:
            if member.get("job") == "Director":
                return member.get("name", "")
        return ""
    except (ValueError, TypeError):
        return ""


def _clean_name(name: str) -> str:
    return name.replace(" ", "")


def _build_soup(row: pd.Series) -> str:
    genres   = [_clean_name(g) for g in row["genres_list"]]
    cast     = [_clean_name(c) for c in row["cast_list"]]
    keywords = [_clean_name(k) for k in row["keywords_list"]]
    director = _clean_name(row["director"]) if row["director"] else ""

    parts = genres + cast + keywords + [director, director] + [row["overview"]]
    return " ".join(filter(None, parts)).lower()


def load_and_merge() -> pd.DataFrame:
    movies  = pd.read_csv(MOVIES_CSV)
    credits = pd.read_csv(CREDITS_CSV)

    movies  = movies.rename(columns={"id": "movie_id"})
    df = movies.merge(credits, on="movie_id", how="inner")

    if "title_x" in df.columns:
        df = df.rename(columns={"title_x": "title"})
        df = df.drop(columns=["title_y"], errors="ignore")

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title"])
    df["overview"] = df["overview"].fillna("")

    df["genres_list"]   = df["genres"].apply(lambda x: _parse_names(x))
    df["keywords_list"] = df["keywords"].apply(lambda x: _parse_names(x, max_items=5))
    df["cast_list"]     = df["cast"].apply(lambda x: _parse_names(x, max_items=3))
    df["director"]      = df["crew"].apply(_get_director)

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["soup"] = df.apply(_build_soup, axis=1)
    return df


def compute_similarity(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english", max_features=15_000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])

    print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")

    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

    return cosine_sim, indices, df


def save_cache(cosine_sim, indices, df: pd.DataFrame) -> None:
    joblib.dump(cosine_sim, MATRIX_PATH)
    joblib.dump(indices,    INDEX_PATH)
    joblib.dump(df,         DF_PATH)
    print(f"  Saved similarity matrix -> {MATRIX_PATH}")
    print(f"  Saved title index       -> {INDEX_PATH}")
    print(f"  Saved dataframe         -> {DF_PATH}")


def load_cache():
    cosine_sim = joblib.load(MATRIX_PATH)
    indices    = joblib.load(INDEX_PATH)
    df         = joblib.load(DF_PATH)
    return cosine_sim, indices, df


def cache_exists() -> bool:
    return all(os.path.exists(p) for p in [MATRIX_PATH, INDEX_PATH, DF_PATH])


def run_preprocessing() -> None:
    print("Loading and merging datasets…")
    df = load_and_merge()
    print(f"  {len(df)} movies loaded.")

    print("Cleaning data…")
    df = clean(df)
    print(f"  {len(df)} movies after cleaning.")

    print("Building feature soup…")
    df = build_features(df)

    print("Computing TF-IDF and cosine similarity matrix…")
    cosine_sim, indices, df = compute_similarity(df)
    print(f"  Similarity matrix shape: {cosine_sim.shape}")

    print("Saving cache…")
    save_cache(cosine_sim, indices, df)

    print("\nDone! Preprocessing complete. Cache saved to ./cache/")


if __name__ == "__main__":
    run_preprocessing()
