import ast
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    genres   = [_clean_name(g) for g in row.get("genres_list", [])]
    cast     = [_clean_name(c) for c in row.get("cast_list", [])]
    keywords = [_clean_name(k) for k in row.get("keywords_list", [])]
    director = _clean_name(row.get("director", "")) if row.get("director") else ""

    parts = genres + cast + keywords + [director, director] + [row["overview"]]
    return " ".join(filter(None, parts)).lower()


def load_and_merge() -> pd.DataFrame:
    # Check if files exist with the common Kaggle names (prioritize the larger 45k metadata)
    m_path = os.path.join(DATA_DIR, "movies_metadata.csv") if os.path.exists(os.path.join(DATA_DIR, "movies_metadata.csv")) else MOVIES_CSV
    c_path = os.path.join(DATA_DIR, "credits.csv") if os.path.exists(os.path.join(DATA_DIR, "credits.csv")) else CREDITS_CSV
    
    if not os.path.exists(m_path):
        raise FileNotFoundError(f"Missing movie dataset. Please ensure {m_path} exists in Data/ folder.")


    movies  = pd.read_csv(m_path, low_memory=False)
    
    # Standardize column names for the 45k dataset (which uses 'id' instead of 'movie_id')
    if "id" in movies.columns and "movie_id" not in movies.columns:
        movies = movies.rename(columns={"id": "movie_id"})
    
    # Some datasets have 'id' as a string with junk, clean it
    movies["movie_id"] = pd.to_numeric(movies["movie_id"], errors="coerce")
    movies = movies.dropna(subset=["movie_id"]).astype({"movie_id": int})

    if os.path.exists(c_path):
        credits = pd.read_csv(c_path)
        # Handle 45k credits column name 'id'
        if "id" in credits.columns and "movie_id" not in credits.columns:
            credits = credits.rename(columns={"id": "movie_id"})
        
        credits["movie_id"] = pd.to_numeric(credits["movie_id"], errors="coerce")
        credits = credits.dropna(subset=["movie_id"]).astype({"movie_id": int})
        
        df = movies.merge(credits, on="movie_id", how="inner")
    else:
        # Fallback if credits file isn't provided (some larger datasets combine them)
        print(f"  Warning: {c_path} not found. Proceeding with movies only.")
        df = movies

    # Standardize Title column
    for col in ["title", "original_title", "title_x"]:
        if col in df.columns:
            df = df.rename(columns={col: "title"})
            break

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.dropna(subset=["title"])
    df["overview"] = df["overview"].fillna("")

    # Clean popularity column (45k dataset has strings)
    if "popularity" in df.columns:
        df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce").fillna(0)
    
    if "genres" in df.columns:
        df["genres_list"] = df["genres"].apply(lambda x: _parse_names(x))
    else:
        df["genres_list"] = [[] for _ in range(len(df))]

    if "keywords" in df.columns:
        df["keywords_list"] = df["keywords"].apply(lambda x: _parse_names(x, max_items=5))
    else:
        df["keywords_list"] = [[] for _ in range(len(df))]

    if "cast" in df.columns:
        df["cast_list"] = df["cast"].apply(lambda x: _parse_names(x, max_items=3))
    else:
        df["cast_list"] = [[] for _ in range(len(df))]

    if "crew" in df.columns:
        df["director"] = df["crew"].apply(_get_director)
    else:
        df["director"] = ["" for _ in range(len(df))]

    df["year"] = pd.to_datetime(df["release_date"], errors="coerce").dt.year

    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df["soup"] = df.apply(_build_soup, axis=1)
    return df


def compute_similarity(df: pd.DataFrame):
    tfidf = TfidfVectorizer(stop_words="english", max_features=15_000)
    tfidf_matrix = tfidf.fit_transform(df["soup"])
    
    print(f"  TF-IDF sparse matrix shape: {tfidf_matrix.shape}")

    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

    return tfidf_matrix, indices, df


def save_cache(tfidf_matrix, indices, df: pd.DataFrame) -> None:
    joblib.dump(tfidf_matrix, MATRIX_PATH)
    joblib.dump(indices,    INDEX_PATH)
    joblib.dump(df,         DF_PATH)
    print(f"  Saved sparse matrix  -> {MATRIX_PATH}")
    print(f"  Saved title index    -> {INDEX_PATH}")
    print(f"  Saved dataframe      -> {DF_PATH}")


def load_cache():
    tfidf_matrix = joblib.load(MATRIX_PATH)
    indices    = joblib.load(INDEX_PATH)
    df         = joblib.load(DF_PATH)
    return tfidf_matrix, indices, df


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

    print("Computing TF-IDF sparse matrix…")
    tfidf_matrix, indices, df = compute_similarity(df)
    
    print("Saving cache…")
    save_cache(tfidf_matrix, indices, df)

    print("\nDone! Preprocessing complete. Cache saved to ./cache/")


if __name__ == "__main__":
    run_preprocessing()
