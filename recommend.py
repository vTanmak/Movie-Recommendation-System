import difflib
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from preprocess import load_cache, cache_exists, run_preprocessing

if not cache_exists():
    print("Cache not found. Running preprocessing pipeline first...")
    run_preprocessing()

tfidf_matrix, indices, movies_df = load_cache()

def get_recommendations(title: str, n: int = 10) -> list[dict]:
    matched_title = None

    if title in indices:
        matched_title = title
    else:
        all_titles = indices.index.tolist()
        close = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.5)
        if close:
            matched_title = close[0]

    if matched_title is None:
        return []

    idx = indices[matched_title]
    if isinstance(idx, (pd.Series, np.ndarray)):
        idx = idx.iloc[0] if hasattr(idx, 'iloc') else idx[0]
    
    # Compute similarity for this specific movie against all others on-the-fly

    movie_vector = tfidf_matrix[idx]
    sim_scores_array = cosine_similarity(movie_vector, tfidf_matrix).flatten()
    sim_scores = list(enumerate(sim_scores_array))

    import math
    hybrid_scores = []
    for idx_match, score in sim_scores:
        if idx_match == idx: continue
        if score < 0.05: continue
        
        pop = movies_df.iloc[idx_match].get("popularity", 1)
        pop_weight = math.log10(max(10, pop))
        hybrid_score = score * pop_weight
        hybrid_scores.append((idx_match, score, hybrid_score))

    hybrid_scores = sorted(hybrid_scores, key=lambda x: x[2], reverse=True)
    hybrid_scores = hybrid_scores[:n]

    results = []
    for movie_idx, raw_score, hybrid_score in hybrid_scores:
        row = movies_df.iloc[movie_idx]
        results.append({
            "title":             row["title"],
            "movie_id":          int(row["movie_id"]),
            "year":              int(row["year"]) if not __import__("math").isnan(row["year"] if row["year"] is not None else float("nan")) else None,
            "genres":            row["genres_list"],
            "vote_average":      float(row.get("vote_average", 0)),
            "popularity":        float(row.get("popularity", 0)),
            "original_language": row.get("original_language", ""),
            "score":             round(float(raw_score), 4),
            "hybrid_score":      round(float(hybrid_score), 4),
        })

    return results


def get_matched_title(title: str) -> str | None:
    if title in indices:
        return title
    all_titles = indices.index.tolist()
    close = difflib.get_close_matches(title, all_titles, n=1, cutoff=0.5)
    return close[0] if close else None


#Quick test
if __name__ == "__main__":
    test_title = "The Dark Knight"
    print(f"Recommendations for '{test_title}':\n")
    recs = get_recommendations(test_title, n=10)
    if recs:
        for i, r in enumerate(recs, 1):
            print(f"  {i:>2}. {r['title']:<40} ({r['year']})  sim={r['score']}  pop={r['popularity']:<6} hybrid={r['hybrid_score']} rating={r['vote_average']}")
    else:
        print("  No recommendations found.")
