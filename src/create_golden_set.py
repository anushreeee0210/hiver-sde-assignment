import pandas as pd
import re
from pathlib import Path

INPUT = Path("data/applesupport_pairs.csv")
OUTPUT = Path("data/golden_set.csv")

df = pd.read_csv(INPUT)

# --------------------------------------------------
# Intent keyword rules
# --------------------------------------------------

INTENTS = {
    "software_ios": [
        "ios", "update", "updating", "upgrade", "version",
        "freeze", "freezing", "slow", "lag", "crash",
        "crashing", "software"
    ],

    "app_issues": [
        "app", "apps", "application", "facebook",
        "whatsapp", "spotify", "youtube", "twitter"
    ],

    "battery_charging": [
        "battery", "charge", "charging", "charger",
        "power", "drain", "dies"
    ],

    "hardware_device": [
        "screen", "display", "camera", "keyboard",
        "speaker", "microphone", "touch", "button",
        "iphone", "ipad"
    ],

    "wifi_connectivity": [
        "wifi", "wi-fi", "bluetooth", "network",
        "internet", "connection", "connect"
    ],

    "icloud_apple_id": [
        "icloud", "apple id", "appleid", "password",
        "account", "locked", "login", "sign in"
    ],

    "app_store_purchases": [
        "itunes", "app store", "appstore", "purchase",
        "purchased", "refund", "payment", "billing",
        "subscription"
    ],

    "mac_macos": [
        "macbook", "imac", "macos", "sierra",
        "high sierra", "finder"
    ],

    "notifications_messages": [
        "notification", "notifications", "alert",
        "messages", "message", "imessage"
    ],

    "security_fraud": [
        "phishing", "scam", "hack", "hacked",
        "fraud", "security", "fake email"
    ]
}


def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Twitter mentions
    text = re.sub(r"@\w+", " ", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def predict_intent(text):
    text = clean_text(text)

    scores = {}

    for intent, keywords in INTENTS.items():
        score = 0

        for keyword in keywords:
            if keyword in text:
                score += 1

        scores[intent] = score

    best_intent = max(scores, key=scores.get)

    # No useful keyword
    if scores[best_intent] == 0:
        return "needs_review"

    return best_intent


# --------------------------------------------------
# Clean customer messages
# --------------------------------------------------

df["clean_message"] = df["customer_message"].apply(clean_text)

# Remove extremely short messages
df = df[df["clean_message"].str.len() >= 15]

# Remove duplicates
df = df.drop_duplicates(subset=["clean_message"])

# Predict intents
df["intent"] = df["customer_message"].apply(predict_intent)

# --------------------------------------------------
# Sample approximately 20 examples per intent
# --------------------------------------------------

samples = []

for intent in INTENTS.keys():

    subset = df[df["intent"] == intent]

    n = min(20, len(subset))

    if n > 0:
        samples.append(
            subset.sample(
                n=n,
                random_state=42
            )
        )

golden = pd.concat(samples, ignore_index=True)

# If fewer than 200, fill remaining slots
if len(golden) < 200:

    remaining = df[
        ~df["customer_tweet_id"].isin(
            golden["customer_tweet_id"]
        )
    ]

    extra = remaining.sample(
        n=min(200 - len(golden), len(remaining)),
        random_state=42
    )

    golden = pd.concat(
        [golden, extra],
        ignore_index=True
    )

# Shuffle
golden = golden.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)

# Add ID
golden.insert(
    0,
    "id",
    range(1, len(golden) + 1)
)

# Keep only useful columns
golden = golden[
    [
        "id",
        "customer_tweet_id",
        "customer_message",
        "intent"
    ]
]

OUTPUT.parent.mkdir(
    parents=True,
    exist_ok=True
)

golden.to_csv(
    OUTPUT,
    index=False
)

print("Golden set created.")
print(f"Examples: {len(golden)}")
print(f"Saved to: {OUTPUT}")

print("\nIntent distribution:")
print(golden["intent"].value_counts())