import math
import requests
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.express as px
import concurrent.futures

from recommend import get_recommendations, get_matched_title, movies_df

st.set_page_config(
    page_title="Movie Recommendations",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: linear-gradient(135deg, #0f0c29, #1a1a2e, #16213e); }

.hero-title {
    font-size: 3.2rem; font-weight: 800;
    background: linear-gradient(90deg, #e94560, #f5a623, #e94560);
    background-size: 200%;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: shimmer 3s infinite linear; text-align: center; margin-bottom: 0;
}
.hero-sub {
    text-align: center; color: #aaa; font-size: 1.1rem; margin-top: 0.3rem;
}
@keyframes shimmer { 0%{background-position:0%} 100%{background-position:200%} }

.movie-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px; padding: 14px; text-align: center;
    transition: transform 0.25s, box-shadow 0.25s;
    backdrop-filter: blur(10px);
}
.movie-card:hover {
    transform: translateY(-6px);
    box-shadow: 0 14px 40px rgba(233,69,96,0.35);
}
.movie-title { font-weight: 700; font-size: 0.92rem; color: #fff; margin-top: 10px; }
.movie-meta  { font-size: 0.78rem; color: #bbb; margin-top: 4px; }
.score-badge {
    display: inline-block; background: linear-gradient(90deg,#e94560,#f5a623);
    color: #fff; border-radius: 20px; padding: 2px 10px;
    font-size: 0.72rem; font-weight: 600; margin-top: 6px;
}
.rating-star { color: #f5a623; }
.section-header {
    font-size: 1.5rem; font-weight: 700; color: #fff;
    border-left: 4px solid #e94560; padding-left: 12px; margin: 2rem 0 1rem;
}
</style>
""", unsafe_allow_html=True)


TMDB_BASE    = "https://api.themoviedb.org/3"
POSTER_BASE  = "https://image.tmdb.org/t/p/w500"
PLACEHOLDER  = (
    "data:image/svg+xml;charset=utf-8,%3Csvg xmlns='http://www.w3.org/2000/svg' "
    "width='200' height='300' viewBox='0 0 200 300'%3E"
    "%3Crect width='200' height='300' fill='%231a1a2e'/%3E"
    "%3Ctext x='50%25' y='45%25' dominant-baseline='middle' text-anchor='middle' "
    "font-family='sans-serif' font-size='14' fill='%23666'%3ENo Poster%3C/text%3E"
    "%3Ctext x='50%25' y='58%25' dominant-baseline='middle' text-anchor='middle' "
    "font-family='sans-serif' font-size='28' fill='%23444'%3E%F0%9F%8E%AC%3C/text%3E"
    "%3C/svg%3E"
)

def _get_api_key() -> str | None:
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_tmdb(movie_id: int) -> dict:
    api_key = _get_api_key()
    if not api_key or api_key == "your_api_key_here":
        return {}
    try:
        url = f"{TMDB_BASE}/movie/{movie_id}?api_key={api_key}&language=en-US"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def get_poster_url(movie_id: int) -> str:
    data = fetch_tmdb(movie_id)
    path = data.get("poster_path")
    return f"{POSTER_BASE}{path}" if path else PLACEHOLDER


def get_overview(movie_id: int, fallback: str = "") -> str:
    data = fetch_tmdb(movie_id)
    return data.get("overview") or fallback


@st.cache_data
def get_all_titles() -> list[str]:
    return sorted(movies_df["title"].dropna().unique().tolist())


def safe_year(y) -> str:
    try:
        if y and not math.isnan(float(y)):
            return str(int(y))
    except Exception:
        pass
    return "N/A"

st.markdown('<p class="hero-title">Movie Recommendation System</p>', unsafe_allow_html=True)
st.markdown("---")

with st.sidebar:
    st.markdown("### Settings")
    n_recs = st.slider("Number of recommendations", min_value=5, max_value=20, value=10)
    st.markdown("---")
    st.markdown("**About**")
    st.caption("A simple content-based movie recommendation system.")

    api_key = _get_api_key()
    if not api_key or api_key == "your_api_key_here":
        st.warning("TMDB API key not set. Posters & live metadata will be unavailable.\n\n"
                   "Add your key to `.streamlit/secrets.toml`.")

all_titles = get_all_titles()
col_search, col_btn = st.columns([5, 1])
with col_search:
    selected = st.selectbox("Search for a movie", options=[""] + all_titles,
                            index=0, placeholder="Type a movie title…")
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    search_clicked = st.button("Find Similar", use_container_width=True, type="primary")

if selected and (search_clicked or True):
    matched = get_matched_title(selected)

    if matched is None:
        st.error(f"No match found for '{selected}'. Please try a different title.")
    else:
        if matched.lower() != selected.lower():
            st.info(f"Showing results for: **{matched}**")

        recs = get_recommendations(matched, n=n_recs)

        if not recs:
            st.warning("No recommendations found for this movie.")
        else:
            query_row = movies_df[movies_df["title"] == matched].iloc[0]
            q_id      = int(query_row["movie_id"])
            q_year    = safe_year(query_row.get("year"))
            q_genres  = ", ".join(query_row["genres_list"][:4]) or "N/A"
            q_rating  = query_row.get("vote_average", 0)
            q_overview = get_overview(q_id, fallback=str(query_row.get("overview", "")))
            q_poster  = get_poster_url(q_id)

            st.markdown(f'<p class="section-header">Selected Movie</p>', unsafe_allow_html=True)
            c1, c2 = st.columns([1, 4])
            with c1:
                try:
                    st.image(q_poster, width=160)
                except Exception:
                    st.markdown(
                        "<div style='width:160px;height:240px;background:#1a1a2e;"
                        "border-radius:10px;display:flex;align-items:center;"
                        "justify-content:center;font-size:40px;'>🎬</div>",
                        unsafe_allow_html=True
                    )
            with c2:
                st.markdown(f"### {matched}")
                st.markdown(f"**Year:** {q_year} &nbsp;|&nbsp; **Genres:** {q_genres} &nbsp;|&nbsp; "
                            f"**Rating:** {q_rating}/10")
                st.markdown(q_overview)

            st.markdown("---")

            st.markdown(f'<p class="section-header">Top {n_recs} Recommendations</p>', unsafe_allow_html=True)

            f_col1, f_col2, f_col3 = st.columns(3)
            
            all_genres = sorted({g for r in recs for g in r["genres"]})
            
            lang_map = {
                "en": "English", "fr": "French", "es": "Spanish", "de": "German", 
                "it": "Italian", "ja": "Japanese", "ko": "Korean", "zh": "Chinese",
                "cn": "Chinese", "hi": "Hindi", "ru": "Russian", "pt": "Portuguese",
                "sv": "Swedish", "da": "Danish", "nl": "Dutch", "no": "Norwegian"
            }
            raw_langs = sorted({r["original_language"] for r in recs if r.get("original_language")})
            display_langs = [lang_map.get(l, l.upper()) for l in raw_langs]
            lang_options = dict(zip(display_langs, raw_langs))
            
            with f_col1:
                sort_by = st.selectbox("Sort By", ["Match Score", "Highest Rated", "Most Popular"])
            with f_col2:
                filter_genre = st.selectbox("Filter Genre", ["All"] + all_genres)
            with f_col3:
                selected_display_lang = st.selectbox("Language", ["All"] + list(lang_options.keys()))
                filter_lang = lang_options.get(selected_display_lang, "All") if selected_display_lang != "All" else "All"

            if filter_genre != "All":
                recs = [r for r in recs if filter_genre in r["genres"]]
                
            if filter_lang != "All":
                recs = [r for r in recs if r.get("original_language") == filter_lang]

            if sort_by == "Highest Rated":
                recs = sorted(recs, key=lambda x: x["vote_average"], reverse=True)
            elif sort_by == "Most Popular":
                recs = sorted(recs, key=lambda x: x.get("popularity", 0), reverse=True)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                poster_futures = {rec["movie_id"]: executor.submit(get_poster_url, rec["movie_id"]) for rec in recs}
            
            posters = {m_id: future.result() for m_id, future in poster_futures.items()}

            cols_per_row = 5
            for row_start in range(0, len(recs), cols_per_row):
                cols = st.columns(cols_per_row)
                for col, rec in zip(cols, recs[row_start:row_start + cols_per_row]):
                    with col:
                        poster = posters[rec["movie_id"]]
                        genres_str = ", ".join(rec["genres"][:2]) if rec["genres"] else "N/A"
                        fallback = PLACEHOLDER.replace("'", "\\'")
                        st.markdown(f"""
                        <div class="movie-card">
                            <img src="{poster}" style="width:100%;border-radius:10px;"
                                 onerror="this.onerror=null;this.src='{fallback}';"/>
                            <p class="movie-title">{rec['title']}</p>
                            <p class="movie-meta">{safe_year(rec['year'])} &nbsp;|&nbsp; {genres_str}</p>
                            <p class="movie-meta"><span class="rating-star">★</span> {rec['vote_average']}</p>
                            <span class="score-badge">Score {int(rec['hybrid_score']*100)}</span>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown("---")

            st.markdown('<p class="section-header">Genre Distribution</p>', unsafe_allow_html=True)
            genre_counts: dict[str, int] = {}
            for rec in recs:
                for g in rec["genres"]:
                    genre_counts[g] = genre_counts.get(g, 0) + 1

            if genre_counts:
                gdf = pd.DataFrame(genre_counts.items(), columns=["Genre", "Count"]).sort_values(
                    "Count", ascending=True)
                fig = px.bar(
                    gdf, x="Count", y="Genre", orientation="h",
                    color="Count", color_continuous_scale=["#f5a623", "#e94560"],
                    template="plotly_dark",
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0),
                    font=dict(family="Inter"),
                )
                st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


