import json
import os
import re
from collections import Counter
from statistics import mean, median

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "ui", "assets", "data")
os.makedirs(OUT_DIR, exist_ok=True)

DATA_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "News_Category_Dataset_v3.json"),
    os.path.join(PROJECT_ROOT, "data", "newscategory", "News_Category_Dataset_v3.json"),
]


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return None if np.isnan(obj) else float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.strftime("%Y-%m-%d")
        return super().default(obj)


def save_json(data, filename):
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=NpEncoder)


def _resolve_data_path():
    for path in DATA_CANDIDATES:
        if os.path.exists(path):
            return path
    searched = "\n".join(DATA_CANDIDATES)
    raise FileNotFoundError(f"News_Category_Dataset_v3.json not found. Searched:\n{searched}")


def _load_df():
    df = pd.read_json(_resolve_data_path(), lines=True)
    df["headline"] = df["headline"].fillna("")
    df["short_description"] = df["short_description"].fillna("")
    df["authors"] = df["authors"].fillna("")
    df["category"] = df["category"].fillna("Unknown")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _word_count(series):
    return series.fillna("").astype(str).str.split().str.len()


STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "for", "on", "at", "with",
    "from", "by", "is", "are", "was", "were", "be", "as", "after", "over", "up",
    "into", "about", "this", "that", "it", "its", "new", "says", "say", "amid",
    "why", "how", "what", "who", "your", "you", "have", "not", "but", "more",
    "can", "has", "our", "one", "all", "will", "out", "when", "his", "their",
    "her", "they", "people", "time", "just", "like", "day", "than", "year", "get",
    "some", "life", "he", "she", "we", "if", "no", "do", "did", "does", "so",
    "my", "me", "us", "them", "had", "been", "also", "here", "too", "off", "now",
    "want", "way", "first", "last", "make", "made", "because", "these", "those",
    "news", "week", "u", "s", "it's", "there", "said", "may", "would", "could",
    "don't", "many", "should", "being", "even", "back", "other", "take", "good",
    "two", "things", "only", "help", "while", "think", "old", "see", "much", "look",
}
TOKEN_PATTERN = re.compile(r"[A-Za-z']+")


def _tokenize(text, remove_stopwords=False, min_len=3):
    tokens = []
    for token in TOKEN_PATTERN.findall(str(text).lower()):
        if len(token) < min_len or token.startswith("'"):
            continue
        if remove_stopwords and token in STOPWORDS:
            continue
        tokens.append(token)
    return tokens


def _character_count(series):
    return series.fillna("").astype(str).str.len()


def _keyword_counts(series, top_k=25):
    counts = Counter()

    for text in series.fillna("").astype(str):
        counts.update(_tokenize(text, remove_stopwords=True))

    return [{"word": word, "count": count} for word, count in counts.most_common(top_k)]


def _stopword_counts(series, top_k=20):
    counts = Counter()
    for text in series.fillna("").astype(str):
        for token in _tokenize(text, remove_stopwords=False, min_len=1):
            if token in STOPWORDS:
                counts[token] += 1
    return [{"word": word, "count": count} for word, count in counts.most_common(top_k)]


def _bigram_counts(series, top_k=20):
    counts = Counter()
    for text in series.fillna("").astype(str):
        tokens = _tokenize(text, remove_stopwords=True)
        counts.update(zip(tokens, tokens[1:]))
    return [
        {"bigram": f"{left} {right}", "count": count}
        for (left, right), count in counts.most_common(top_k)
    ]


def _vocabulary_richness(series):
    all_tokens = []
    for text in series.fillna("").astype(str):
        all_tokens.extend(_tokenize(text, remove_stopwords=True))

    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    unique_tokens = len(token_counts)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    dislegomena_count = sum(1 for count in token_counts.values() if count == 2)

    return {
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "type_token_ratio": round(float(unique_tokens / total_tokens), 4) if total_tokens else 0.0,
        "hapax_count": hapax_count,
        "hapax_ratio": round(float(hapax_count / unique_tokens), 4) if unique_tokens else 0.0,
        "dislegomena_count": dislegomena_count,
        "avg_unique_per_article": round(
            float(
                mean(
                    len(set(_tokenize(text, remove_stopwords=True)))
                    for text in series.fillna("").astype(str)
                )
            ),
            2,
        ) if len(series) else 0.0,
        "top_unique_words": [
            {"word": word, "count": count}
            for word, count in token_counts.most_common(15)
        ],
    }


def _top_terms_by_category(df, categories, top_k=8):
    payload = []
    for category in categories:
        subset = df.loc[df["category"] == category, "combined_text"]
        terms = _keyword_counts(subset, top_k=top_k)
        payload.append({
            "category": category,
            "terms": terms,
        })
    return payload


def _tfidf_keywords_by_category(df, categories, top_k=8):
    payload = []
    for category in categories:
        subset = df.loc[df["category"] == category, "combined_text"].astype(str)
        if subset.empty:
            payload.append({"category": category, "terms": []})
            continue

        vectorizer = TfidfVectorizer(stop_words="english", max_features=4000)
        matrix = vectorizer.fit_transform(subset)
        scores = np.asarray(matrix.mean(axis=0)).ravel()
        features = vectorizer.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:top_k]
        payload.append({
            "category": category,
            "terms": [
                {"term": str(features[i]), "score": round(float(scores[i]), 4)}
                for i in top_idx if scores[i] > 0
            ],
        })
    return payload


def run_text_eda():
    df = _load_df()
    df["combined_text"] = (df["headline"].str.strip() + " " + df["short_description"].str.strip()).str.strip()
    df["year"] = df["date"].dt.year

    headline_wc = _word_count(df["headline"])
    desc_wc = _word_count(df["short_description"])
    combined_wc = _word_count(df["combined_text"])
    headline_cc = _character_count(df["headline"])
    desc_cc = _character_count(df["short_description"])
    combined_cc = _character_count(df["combined_text"])

    missing = {
        col: {
            "missing": int(df[col].astype(str).str.strip().eq("").sum()) if col != "date" else int(df[col].isna().sum()),
            "missing_pct": round(float((df[col].astype(str).str.strip().eq("").sum() if col != "date" else df[col].isna().sum()) / len(df) * 100), 2),
        }
        for col in ["headline", "short_description", "authors", "link", "category", "date"]
    }
    missing["headline"]["missing"] = int(df["headline"].str.strip().eq("").sum())
    missing["headline"]["missing_pct"] = round(float(missing["headline"]["missing"] / len(df) * 100), 2)
    missing["short_description"]["missing"] = int(df["short_description"].str.strip().eq("").sum())
    missing["short_description"]["missing_pct"] = round(float(missing["short_description"]["missing"] / len(df) * 100), 2)
    missing["authors"]["missing"] = int(df["authors"].str.strip().eq("").sum())
    missing["authors"]["missing_pct"] = round(float(missing["authors"]["missing"] / len(df) * 100), 2)

    dedup_subset = ["headline", "short_description", "category", "date"]
    duplicate_records = int(df.duplicated(subset=dedup_subset).sum())
    duplicate_links = int(df["link"].duplicated().sum())

    category_counts = df["category"].value_counts()
    year_counts = df["year"].value_counts().sort_index()

    top_authors = (
        df.loc[df["authors"].str.strip().ne(""), "authors"]
        .value_counts()
        .head(15)
    )

    sample_cols = ["date", "category", "headline", "short_description", "authors", "link"]
    sample_rows = df[sample_cols].head(8).copy()
    sample_rows["date"] = sample_rows["date"].dt.strftime("%Y-%m-%d")
    top_categories = category_counts.head(6).index.tolist()

    save_json({
        "dataset_name": "News_Category_Dataset_v3.json",
        "total_articles": int(len(df)),
        "feature_count": int(df.shape[1] - 2),
        "category_count": int(category_counts.shape[0]),
        "date_min": df["date"].min(),
        "date_max": df["date"].max(),
        "headline_mean_words": round(float(mean(headline_wc)), 2),
        "headline_median_words": int(median(headline_wc)),
        "combined_mean_words": round(float(mean(combined_wc)), 2),
        "combined_median_words": int(median(combined_wc)),
        "headline_mean_chars": round(float(mean(headline_cc)), 2),
        "headline_median_chars": int(median(headline_cc)),
        "combined_mean_chars": round(float(mean(combined_cc)), 2),
        "combined_median_chars": int(median(combined_cc)),
        "display_columns": sample_cols,
        "sample_rows": sample_rows.to_dict(orient="records"),
        "columns": [
            {
                "name": col,
                "dtype": str(df[col].dtype),
                "non_null": int(df[col].notna().sum()) if col == "date" else int(df[col].astype(str).str.strip().ne("").sum()),
                "missing": missing[col]["missing"],
                "missing_pct": missing[col]["missing_pct"],
            }
            for col in ["link", "headline", "category", "short_description", "authors", "date"]
        ],
    }, "text_overview.json")

    top10 = category_counts.head(10)
    tail5 = category_counts.tail(5)
    imbalance_ratio = round(float(category_counts.max() / category_counts.min()), 2)
    save_json({
        "labels": top10.index.tolist(),
        "counts": [int(v) for v in top10.values.tolist()],
        "shares": [round(float(v / len(df) * 100), 2) for v in top10.values.tolist()],
        "smallest_labels": tail5.index.tolist(),
        "smallest_counts": [int(v) for v in tail5.values.tolist()],
        "max_count": int(category_counts.max()),
        "min_count": int(category_counts.min()),
        "imbalance_ratio": imbalance_ratio,
    }, "text_category_dist.json")

    bins = [0, 5, 10, 15, 20, 30, 40, 60, 100]
    headline_hist, headline_edges = np.histogram(headline_wc, bins=bins)
    combined_hist, combined_edges = np.histogram(combined_wc, bins=bins)
    char_bins = [0, 25, 50, 75, 100, 150, 200, 300, 500]
    headline_char_hist, headline_char_edges = np.histogram(headline_cc, bins=char_bins)
    combined_char_hist, combined_char_edges = np.histogram(combined_cc, bins=char_bins)
    save_json({
        "headline_bins": [f"{headline_edges[i]}-{headline_edges[i + 1] - 1}" for i in range(len(headline_hist))],
        "headline_counts": [int(v) for v in headline_hist.tolist()],
        "combined_bins": [f"{combined_edges[i]}-{combined_edges[i + 1] - 1}" for i in range(len(combined_hist))],
        "combined_counts": [int(v) for v in combined_hist.tolist()],
        "headline_char_bins": [f"{headline_char_edges[i]}-{headline_char_edges[i + 1] - 1}" for i in range(len(headline_char_hist))],
        "headline_char_counts": [int(v) for v in headline_char_hist.tolist()],
        "combined_char_bins": [f"{combined_char_edges[i]}-{combined_char_edges[i + 1] - 1}" for i in range(len(combined_char_hist))],
        "combined_char_counts": [int(v) for v in combined_char_hist.tolist()],
        "headline_mean": round(float(mean(headline_wc)), 2),
        "headline_median": int(median(headline_wc)),
        "desc_mean": round(float(mean(desc_wc)), 2),
        "desc_median": int(median(desc_wc)),
        "combined_mean": round(float(mean(combined_wc)), 2),
        "combined_median": int(median(combined_wc)),
        "headline_char_mean": round(float(mean(headline_cc)), 2),
        "headline_char_median": int(median(headline_cc)),
        "desc_char_mean": round(float(mean(desc_cc)), 2),
        "desc_char_median": int(median(desc_cc)),
        "combined_char_mean": round(float(mean(combined_cc)), 2),
        "combined_char_median": int(median(combined_cc)),
    }, "text_lengths.json")

    save_json({
        "years": [int(y) for y in year_counts.index.tolist()],
        "counts": [int(v) for v in year_counts.values.tolist()],
        "peak_year": int(year_counts.idxmax()),
        "peak_count": int(year_counts.max()),
        "latest_year": int(year_counts.index.max()),
        "latest_count": int(year_counts.iloc[-1]),
    }, "text_timeline.json")

    save_json({
        "missing": [{"column": col, **vals} for col, vals in missing.items()],
        "duplicate_records": duplicate_records,
        "duplicate_links": duplicate_links,
    }, "text_quality.json")

    save_json({
        "keywords": _keyword_counts(df["combined_text"], top_k=20),
        "top_authors": [{"author": author, "count": int(count)} for author, count in top_authors.items()],
    }, "text_terms.json")

    stopword_counter = Counter()
    for text in df["combined_text"].fillna("").astype(str):
        for token in _tokenize(text, remove_stopwords=False, min_len=1):
            if token in STOPWORDS:
                stopword_counter[token] += 1
    stopword_stats = [{"word": word, "count": count} for word, count in stopword_counter.most_common(20)]
    vocab_stats = _vocabulary_richness(df["combined_text"])
    save_json({
        "stopwords": stopword_stats,
        "total_stopword_tokens": int(sum(stopword_counter.values())),
        "stopword_share_top20": round(
            float(sum(row["count"] for row in stopword_stats) / (sum(stopword_counter.values()) or 1) * 100),
            2,
        ) if stopword_counter else 0.0,
    }, "text_stopwords.json")

    save_json(vocab_stats, "text_vocabulary_richness.json")

    save_json({
        "bigrams": _bigram_counts(df["combined_text"], top_k=20),
    }, "text_bigrams.json")

    category_length_df = (
        df.assign(headline_words=headline_wc, combined_words=combined_wc)
        .groupby("category")
        .agg(
            article_count=("category", "size"),
            headline_mean=("headline_words", "mean"),
            combined_mean=("combined_words", "mean"),
            combined_median=("combined_words", "median"),
        )
        .sort_values("article_count", ascending=False)
        .reset_index()
    )
    save_json({
        "chart_categories": category_length_df["category"].head(10).tolist(),
        "chart_combined_mean": [round(float(v), 2) for v in category_length_df["combined_mean"].head(10).tolist()],
        "categories": category_length_df["category"].tolist(),
        "article_count": [int(v) for v in category_length_df["article_count"].tolist()],
        "headline_mean": [round(float(v), 2) for v in category_length_df["headline_mean"].tolist()],
        "combined_mean": [round(float(v), 2) for v in category_length_df["combined_mean"].tolist()],
        "combined_median": [round(float(v), 2) for v in category_length_df["combined_median"].tolist()],
    }, "text_length_by_category.json")

    trend_pivot = (
        df[df["category"].isin(top_categories)]
        .groupby(["year", "category"])
        .size()
        .unstack(fill_value=0)
        .sort_index()
    )
    save_json({
        "years": [int(v) for v in trend_pivot.index.tolist()],
        "categories": trend_pivot.columns.tolist(),
        "series": {
            category: [int(v) for v in trend_pivot[category].tolist()]
            for category in trend_pivot.columns
        },
    }, "text_yearly_category_trends.json")

    save_json({
        "categories": top_categories,
        "groups": _top_terms_by_category(df, top_categories, top_k=8),
    }, "text_top_terms_by_category.json")

    save_json({
        "categories": top_categories,
        "groups": _tfidf_keywords_by_category(df, top_categories, top_k=8),
    }, "text_tfidf_by_category.json")
