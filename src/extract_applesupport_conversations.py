import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_FILE = Path("data/applesupport_conversations.csv")

CHUNK_SIZE = 100_000

print("Step 1: Finding AppleSupport tweets...")

# --------------------------------------------------
# PASS 1: Find AppleSupport tweets and related IDs
# --------------------------------------------------

apple_tweet_ids = set()
related_tweet_ids = set()

total_rows = 0
apple_rows = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    apple = chunk[
        chunk["author_id"].astype(str).str.strip().str.lower()
        == "applesupport"
    ]

    if not apple.empty:
        apple_rows += len(apple)

        # AppleSupport tweet IDs
        apple_tweet_ids.update(
            apple["tweet_id"].astype(str)
        )

        # Tweets AppleSupport directly responded to
        parent_ids = apple["in_response_to_tweet_id"].dropna()

        related_tweet_ids.update(
            parent_ids.astype(int).astype(str)
        )

        # Tweets AppleSupport responded with
        for values in apple["response_tweet_id"].dropna():
            for tweet_id in str(values).split(","):
                tweet_id = tweet_id.strip()

                if tweet_id:
                    related_tweet_ids.add(tweet_id)

    print(
        f"Processed {total_rows:,} rows | "
        f"AppleSupport: {apple_rows:,}"
    )

print("\nAppleSupport tweets:", len(apple_tweet_ids))
print("Related tweet IDs:", len(related_tweet_ids))


# --------------------------------------------------
# PASS 2: Extract AppleSupport + directly related
# customer tweets
# --------------------------------------------------

print("\nStep 2: Extracting conversation tweets...")

conversation_chunks = []
total_rows = 0

target_ids = apple_tweet_ids | related_tweet_ids

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    mask = chunk["tweet_id"].astype(str).isin(target_ids)

    matching = chunk[mask]

    if not matching.empty:
        conversation_chunks.append(matching)

    print(f"Processed {total_rows:,} rows")


# --------------------------------------------------
# SAVE
# --------------------------------------------------

if conversation_chunks:

    conversations = pd.concat(
        conversation_chunks,
        ignore_index=True
    )

    # Sort chronologically
    conversations["created_at"] = pd.to_datetime(
        conversations["created_at"],
        errors="coerce"
    )

    conversations = conversations.sort_values(
        "created_at"
    )

    conversations.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===================================")
    print("Extraction complete!")
    print("===================================")

    print("Total conversation tweets:", len(conversations))

    print("\nInbound distribution:")
    print(
        conversations["inbound"]
        .value_counts(dropna=False)
    )

    print("\nAuthors:")
    print(
        conversations["author_id"]
        .value_counts()
        .head(10)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

else:
    print("No conversation tweets found.")