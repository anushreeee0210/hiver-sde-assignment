import pandas as pd
import pickle
import re
from pathlib import Path


MODEL_DIR = Path("models")


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading models...")

with open(
    MODEL_DIR / "tfidf_vectorizer.pkl",
    "rb"
) as f:
    vectorizer = pickle.load(f)

with open(
    MODEL_DIR / "retriever.pkl",
    "rb"
) as f:
    retriever = pickle.load(f)

df = pd.read_pickle(
    MODEL_DIR / "retrieval_data.pkl"
)


# --------------------------------------------------
# Text cleaning
# --------------------------------------------------

def clean_text(text):
    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"@\w+",
        " ",
        text
    )

    text = re.sub(
        r"&\w+;",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# --------------------------------------------------
# Retrieval function
# --------------------------------------------------

def retrieve(query, k=5):

    cleaned_query = clean_text(query)

    query_vector = vectorizer.transform(
        [cleaned_query]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=k
    )

    results = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = df.iloc[index]

        results.append({
            "similarity": round(
                1 - distance,
                4
            ),
            "customer_message":
                row["customer_message"],
            "support_response":
                row["support_response"]
        })

    return results


# --------------------------------------------------
# Test queries
# --------------------------------------------------

queries = [
    "Why is my iPhone battery draining so fast?",
    "My iPhone won't connect to WiFi",
    "I can't access my iCloud account",
    "My apps keep crashing after the iOS update",
    "I want a refund for an App Store purchase"
]


for query in queries:

    print("\n" + "=" * 80)
    print("QUERY:", query)
    print("=" * 80)

    results = retrieve(query, k=3)

    for i, result in enumerate(
        results,
        start=1
    ):

        print(f"\n--- Result {i} ---")
        print(
            f"Similarity: "
            f"{result['similarity']}"
        )

        print(
            "Customer:",
            result["customer_message"]
        )

        print(
            "AppleSupport:",
            result["support_response"]
        )