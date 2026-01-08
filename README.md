# Movie Recommendation System

A content-based movie recommendation system built with Python, Streamlit, and the TMDB API. 

## Features
- **Content-Based Filtering:** Recommends movies using TF-IDF vectorization and cosine similarity on movie overviews, genres, keywords, cast, and crew.
- **Hybrid Scoring:** Combines similarity score with TMDB popularity metrics for robust, high-quality recommendations.
- **Dynamic UI:** built in Streamlit with dynamic genre/language filtering and intelligent sorting.
- **Live Metadata:** Fetches real-time movie posters and data using the TMDB API. 

## Setup
1. Clone the repository.
2. Download the `tmdb_5000_movies.csv` and `tmdb_5000_credits.csv` datasets from Kaggle into a `Data/` folder.
3. Install dependencies: `pip install -r requirements.txt`
4. Run the preprocessing script to build the cache: `python preprocess.py`
5. Add your TMDB API key to `.streamlit/secrets.toml`:
   ```toml
   TMDB_API_KEY = "your_api_key_here"
   ```
6. Run the app: `streamlit run app.py`
