import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans


INPUT_FILE = "data/applesupport_pairs.csv"

N_CLUSTERS = 10


print("Loading AppleSupport pairs...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Total pairs:", len(df))


# --------------------------------------------------
# CLEAN TEXT
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Remove HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Keep letters/numbers
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = (
    df["customer_message"]
    .apply(clean_text)
)


# Remove very short messages
df = df[
    df["clean_text"].str.len() >= 15
].copy()


print("Usable messages:", len(df))


# --------------------------------------------------
# GENERIC WORDS
# --------------------------------------------------

custom_stopwords = [
    "apple",
    "applesupport",
    "iphone",
    "phone",
    "help",
    "please",
    "thanks",
    "thank",
    "support",
    "hey",
    "hi",
    "hello",
    "just",
    "really",
    "like",
    "want",
    "need",
    "know",
    "does",
    "did",
    "got",
    "get",
    "getting",
    "thing",
    "things",
    "working",
    "work",
    "problem",
    "issue",
    "today",
    "time",
]


# --------------------------------------------------
# TF-IDF
# --------------------------------------------------

print("\nCreating improved TF-IDF representation...")

vectorizer = TfidfVectorizer(
    stop_words=custom_stopwords,
    ngram_range=(1, 2),
    min_df=20,
    max_df=0.90,
    max_features=30000,
    sublinear_tf=True
)

X = vectorizer.fit_transform(
    df["clean_text"]
)

print("TF-IDF shape:", X.shape)


# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------

print("\nRunning clustering...")

model = MiniBatchKMeans(
    n_clusters=N_CLUSTERS,
    random_state=42,
    batch_size=4096,
    n_init=10
)

df["cluster"] = model.fit_predict(X)


# --------------------------------------------------
# ANALYZE CLUSTERS
# --------------------------------------------------

terms = vectorizer.get_feature_names_out()

print("\n")
print("=" * 90)
print("IMPROVED INTENT DISCOVERY")
print("=" * 90)


for cluster_id in range(N_CLUSTERS):

    mask = df["cluster"] == cluster_id

    cluster_size = mask.sum()

    center = model.cluster_centers_[cluster_id]

    top_indices = center.argsort()[-20:][::-1]

    top_terms = [
        terms[i]
        for i in top_indices
    ]

    print("\n" + "-" * 90)

    print(
        f"CLUSTER {cluster_id} "
        f"({cluster_size:,} messages)"
    )

    print(
        "Top terms:",
        ", ".join(top_terms)
    )

    print("\nExamples:")

    examples = df.loc[
        mask,
        "customer_message"
    ].sample(
        min(10, cluster_size),
        random_state=42
    )

    for example in examples:
        print(" -", example)


print("\n")
print("=" * 90)
print("DONE")
print("=" * 90)