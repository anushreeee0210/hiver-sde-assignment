import pandas as pd

FILE = "data/twcs.csv"

print("Reading dataset...")

df = pd.read_csv(
    FILE,
    usecols=[
        "tweet_id",
        "author_id",
        "inbound",
        "text",
        "response_tweet_id",
        "in_response_to_tweet_id"
    ],
    low_memory=False
)

print("\nTotal rows:", len(df))

print("\nSample rows:")
print(df.head(20).to_string(index=False))

print("\nUnique author IDs:", df["author_id"].nunique())

print("\nInbound distribution:")
print(df["inbound"].value_counts())