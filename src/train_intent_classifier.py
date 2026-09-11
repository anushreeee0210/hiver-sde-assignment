import pandas as pd
import pickle
import re

from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# --------------------------------------------------
# Paths
# --------------------------------------------------

INPUT = Path("data/golden_set.csv")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# --------------------------------------------------
# Load golden set
# --------------------------------------------------

print("Loading golden set...")

df = pd.read_csv(INPUT)

print(f"Examples: {len(df)}")


# --------------------------------------------------
# Clean text
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove Twitter mentions
    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    # Remove HTML entities
    text = re.sub(
        r"&\w+;",
        " ",
        text
    )

    # Normalize whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df["clean_message"] = (
    df["customer_message"]
    .apply(clean_text)
)


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nTraining TF-IDF...")

vectorizer = TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df["clean_message"]
)

y = df["intent"]


# --------------------------------------------------
# Logistic Regression
# --------------------------------------------------

print("Training Logistic Regression...")

classifier = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

classifier.fit(X, y)


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(
    MODEL_DIR / "intent_vectorizer.pkl",
    "wb"
) as f:
    pickle.dump(vectorizer, f)


with open(
    MODEL_DIR / "intent_classifier.pkl",
    "wb"
) as f:
    pickle.dump(classifier, f)


print("\nDone.")

print(
    "Created:"
)

print(
    "  models/intent_vectorizer.pkl"
)

print(
    "  models/intent_classifier.pkl"
)