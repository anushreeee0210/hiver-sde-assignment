import pandas as pd
import pickle
import re

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT = Path("data/applesupport_pairs.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Load dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT)

print(f"Total pairs: {len(df)}")


# --------------------------------------------------
# Clean text
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_customer_message"] = (
    df["customer_message"]
    .apply(clean_text)
)


# --------------------------------------------------
# Remove bad/empty messages
# --------------------------------------------------

df = df[
    df["clean_customer_message"].str.len() >= 10
].copy()

df = df.drop_duplicates(
    subset=["clean_customer_message"]
)

print(f"Usable pairs: {len(df)}")


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nBuilding TF-IDF vectorizer...")

vectorizer = TfidfVectorizer(
    max_features=100000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df["clean_customer_message"]
)

print(f"TF-IDF matrix shape: {X.shape}")


# --------------------------------------------------
# Nearest-neighbor retrieval
# --------------------------------------------------

print("\nBuilding nearest-neighbor index...")

retriever = NearestNeighbors(
    n_neighbors=5,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

retriever.fit(X)


# --------------------------------------------------
# Save everything
# --------------------------------------------------

print("\nSaving models...")

with open(
    MODEL_DIR / "tfidf_vectorizer.pkl",
    "wb"
) as f:
    pickle.dump(vectorizer, f)

with open(
    MODEL_DIR / "retriever.pkl",
    "wb"
) as f:
    pickle.dump(retriever, f)

df.to_pickle(
    MODEL_DIR / "retrieval_data.pkl"
)

print("\nDone.")

print("Created:")
print("  models/tfidf_vectorizer.pkl")
print("  models/retriever.pkl")
print("  models/retrieval_data.pkl")