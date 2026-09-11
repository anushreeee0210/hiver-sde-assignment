import pandas as pd

INPUT_FILE = "data/applesupport_conversations.csv"
OUTPUT_FILE = "data/applesupport_pairs.csv"

print("Loading AppleSupport conversations...")

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Convert IDs to strings so they match reliably
df["tweet_id"] = df["tweet_id"].astype(str)

df["parent_id"] = (
    pd.to_numeric(
        df["in_response_to_tweet_id"],
        errors="coerce"
    )
    .astype("Int64")
    .astype(str)
)

# Remove invalid parent IDs
df.loc[
    df["in_response_to_tweet_id"].isna(),
    "parent_id"
] = None

# Create lookup: tweet ID -> tweet information
tweet_lookup = df.set_index("tweet_id")

# Keep AppleSupport responses
apple = df[
    df["author_id"] == "AppleSupport"
].copy()

print("AppleSupport responses:", len(apple))

# Only responses that directly respond to another tweet
apple = apple[
    apple["parent_id"].notna()
].copy()

# Find the parent tweet
apple["customer_message"] = apple["parent_id"].map(
    tweet_lookup["text"]
)

apple["customer_tweet_id"] = apple["parent_id"]

apple["support_response"] = apple["text"]

# Find parent author
apple["customer_author_id"] = apple["parent_id"].map(
    tweet_lookup["author_id"]
)

# Keep only cases where the parent is actually a customer
apple = apple[
    apple["customer_author_id"].notna()
    & (apple["customer_author_id"] != "AppleSupport")
].copy()

# Keep useful columns
pairs = apple[
    [
        "customer_tweet_id",
        "tweet_id",
        "customer_message",
        "support_response",
        "created_at"
    ]
].copy()

# Remove empty messages
pairs["customer_message"] = pairs["customer_message"].fillna("").str.strip()
pairs["support_response"] = pairs["support_response"].fillna("").str.strip()

pairs = pairs[
    (pairs["customer_message"] != "")
    & (pairs["support_response"] != "")
]

# Remove exact duplicate pairs
pairs = pairs.drop_duplicates(
    subset=["customer_message", "support_response"]
)

# Save
pairs.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n==============================")
print("Support pair creation complete")
print("==============================")

print("Total pairs:", len(pairs))

print("\nColumns:")
print(pairs.columns.tolist())

print("\nSample pairs:\n")

for _, row in pairs.head(10).iterrows():
    print("CUSTOMER:")
    print(row["customer_message"])
    print("\nAPPLESUPPORT:")
    print(row["support_response"])
    print("\n" + "-" * 80)

print(f"\nSaved to: {OUTPUT_FILE}")