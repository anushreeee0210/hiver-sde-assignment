import pandas as pd

FILE = "data/applesupport_conversations.csv"

df = pd.read_csv(FILE, low_memory=False)

print("===== BASIC INFO =====")
print("Total tweets:", len(df))
print("Customer tweets:", (df["inbound"] == True).sum())
print("AppleSupport tweets:", (df["inbound"] == False).sum())

# Make tweet IDs strings for reliable matching
tweet_ids = set(df["tweet_id"].astype(str))

df["parent_id"] = (
    df["in_response_to_tweet_id"]
    .dropna()
    .astype(int)
    .astype(str)
)

# Does the parent tweet exist in our extracted dataset?
df["parent_exists"] = df["parent_id"].isin(tweet_ids)

print("\n===== PARENT CONNECTIONS =====")

print(
    df.groupby("inbound")["parent_exists"]
    .agg(["count", "sum"])
)

print("\n===== PARENT CONNECTION RATE =====")

for inbound_value, name in [
    (True, "Customer"),
    (False, "AppleSupport")
]:
    subset = df[df["inbound"] == inbound_value]

    with_parent = subset["in_response_to_tweet_id"].notna().sum()
    parent_exists = subset["parent_exists"].sum()

    print(f"\n{name}:")
    print("  Total:", len(subset))
    print("  Has parent ID:", with_parent)
    print("  Parent exists in dataset:", parent_exists)

# --------------------------------------------------
# Customer -> AppleSupport direct pairs
# --------------------------------------------------

print("\n===== DIRECT CUSTOMER → APPLESUPPORT PAIRS =====")

apple_ids = set(
    df[df["author_id"] == "AppleSupport"]["tweet_id"]
    .astype(str)
)

customer_df = df[df["inbound"] == True].copy()

customer_df["parent_is_apple"] = (
    customer_df["parent_id"].isin(apple_ids)
)

print(
    "Customer tweets responding directly to AppleSupport:",
    customer_df["parent_is_apple"].sum()
)

# --------------------------------------------------
# AppleSupport -> Customer direct pairs
# --------------------------------------------------

customer_ids = set(
    df[df["inbound"] == True]["tweet_id"]
    .astype(str)
)

apple_df = df[df["author_id"] == "AppleSupport"].copy()

apple_df["parent_is_customer"] = (
    apple_df["parent_id"].isin(customer_ids)
)

print(
    "AppleSupport tweets responding directly to customer:",
    apple_df["parent_is_customer"].sum()
)

# --------------------------------------------------
# Date range
# --------------------------------------------------

df["created_at"] = pd.to_datetime(
    df["created_at"],
    errors="coerce",
    utc=True
)

print("\n===== DATE RANGE =====")
print("Earliest:", df["created_at"].min())
print("Latest:", df["created_at"].max())

print("\n===== DONE =====")