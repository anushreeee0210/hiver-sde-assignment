import pandas as pd
import re
from collections import Counter

INPUT_FILE = "data/applesupport_pairs.csv"

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Total pairs:", len(df))


# --------------------------------------------------
# TEXT CLEANING
# --------------------------------------------------

def clean_text(text):

    text = str(text).lower()

    # URLs
    text = re.sub(r"https?://\S+", " ", text)

    # Mentions
    text = re.sub(r"@\w+", " ", text)

    # HTML entities
    text = re.sub(r"&\w+;", " ", text)

    # Remove punctuation
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Extra spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


df["clean_text"] = (
    df["customer_message"]
    .apply(clean_text)
)


# --------------------------------------------------
# REMOVE GENERIC WORDS
# --------------------------------------------------

stopwords = {
    "the", "and", "for", "that", "this", "with",
    "you", "your", "are", "was", "have", "has",
    "had", "but", "not", "can", "cant", "could",
    "would", "should", "from", "they", "them",
    "their", "there", "here", "what", "when",
    "where", "why", "how", "who", "will",
    "just", "like", "really", "very", "please",
    "thanks", "thank", "help", "need", "want",
    "know", "does", "did", "got", "get",
    "getting", "make", "made", "thing", "things",
    "time", "today", "now", "new", "good",
    "great", "hey", "hi", "hello",

    # Apple/Twitter specific
    "apple", "applesupport", "iphone", "phone",
    "ios", "ipad", "mac", "support",

    # Generic issue words
    "problem", "issue", "working", "work",
    "fix", "fixed", "update", "updated",
}


# --------------------------------------------------
# WORD FREQUENCY
# --------------------------------------------------

word_counter = Counter()

for text in df["clean_text"]:

    words = text.split()

    words = [
        word
        for word in words
        if word not in stopwords
        and len(word) >= 3
        and not word.isdigit()
    ]

    word_counter.update(words)


print("\n")
print("=" * 80)
print("MOST FREQUENT MEANINGFUL WORDS")
print("=" * 80)

for word, count in word_counter.most_common(150):

    print(f"{word:25} {count:,}")


# --------------------------------------------------
# IMPORTANT TOPIC KEYWORDS
# --------------------------------------------------

topic_keywords = {

    "Battery": [
        "battery",
        "drain",
        "charging",
        "charge",
        "charger",
        "power",
        "dies"
    ],

    "iCloud / Apple ID": [
        "icloud",
        "appleid",
        "account",
        "password",
        "locked",
        "login",
        "sign",
        "security"
    ],

    "Wi-Fi / Connectivity": [
        "wifi",
        "wi-fi",
        "bluetooth",
        "network",
        "internet",
        "connection",
        "connect"
    ],

    "App Store / Purchases": [
        "itunes",
        "appstore",
        "store",
        "purchase",
        "purchased",
        "refund",
        "payment",
        "billing",
        "subscription"
    ],

    "Applications": [
        "app",
        "apps",
        "application",
        "crash",
        "crashing",
        "facebook",
        "whatsapp",
        "spotify",
        "youtube"
    ],

    "Notifications": [
        "notification",
        "notifications",
        "alert",
        "alerts",
        "messages"
    ],

    "Hardware": [
        "screen",
        "display",
        "camera",
        "keyboard",
        "speaker",
        "microphone",
        "touch",
        "button",
        "touchbar"
    ],

    "Mac": [
        "macbook",
        "imac",
        "macos",
        "sierra",
        "high",
        "finder"
    ],

    "Software / iOS": [
        "ios",
        "update",
        "updating",
        "version",
        "freeze",
        "freezing",
        "slow",
        "lag",
        "crash"
    ],

    "Security / Phishing": [
        "phishing",
        "scam",
        "hack",
        "hacked",
        "fraud",
        "security"
    ]
}


print("\n")
print("=" * 80)
print("TOPIC KEYWORD COUNTS")
print("=" * 80)


for topic, keywords in topic_keywords.items():

    total = 0

    for keyword in keywords:
        total += word_counter[keyword]

    print(
        f"{topic:30} {total:,}"
    )


# --------------------------------------------------
# EXAMPLES FOR EACH TOPIC
# --------------------------------------------------

print("\n")
print("=" * 80)
print("REPRESENTATIVE EXAMPLES")
print("=" * 80)


for topic, keywords in topic_keywords.items():

    pattern = "|".join(
        re.escape(keyword)
        for keyword in keywords
    )

    matches = df[
        df["clean_text"].str.contains(
            pattern,
            regex=True,
            na=False
        )
    ]

    print("\n" + "-" * 80)
    print(f"{topic} ({len(matches):,} messages)")
    print("-" * 80)

    if len(matches) > 0:

        examples = matches.sample(
            min(5, len(matches)),
            random_state=42
        )

        for message in examples["customer_message"]:
            print("-", message)


print("\n")
print("=" * 80)
print("TOPIC ANALYSIS COMPLETE")
print("=" * 80)