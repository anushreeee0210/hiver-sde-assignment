import pandas as pd
import pickle
import re
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity


# ==================================================
# PATHS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"


# ==================================================
# LOAD MODELS
# ==================================================

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_DIR / "retriever.pkl", "rb") as f:
    retriever = pickle.load(f)

df = pd.read_pickle(
    MODEL_DIR / "retrieval_data.pkl"
)


# ==================================================
# INTENT KEYWORDS
# ==================================================

INTENTS = {

    "software_ios": [
        "ios",
        "update",
        "updating",
        "upgrade",
        "version",
        "freeze",
        "freezing",
        "slow",
        "lag",
        "crash",
        "crashing",
        "software"
    ],

    "app_issues": [
        "app",
        "apps",
        "application",
        "facebook",
        "whatsapp",
        "spotify",
        "youtube",
        "twitter"
    ],

    "battery_charging": [
        "battery",
        "charge",
        "charging",
        "charger",
        "power",
        "drain",
        "dies"
    ],

    "hardware_device": [
        "screen",
        "display",
        "camera",
        "keyboard",
        "speaker",
        "microphone",
        "touch",
        "button",
        "broken",
        "black screen"
    ],

    "wifi_connectivity": [
        "wifi",
        "wi-fi",
        "wireless",
        "network",
        "internet"
    ],

    "icloud_apple_id": [
        "icloud",
        "apple id",
        "appleid",
        "password",
        "account",
        "locked",
        "login",
        "sign in"
    ],

    "app_store_purchases": [
        "itunes",
        "app store",
        "appstore",
        "purchase",
        "purchased",
        "refund",
        "payment",
        "billing",
        "subscription"
    ],

    "mac_macos": [
        "macbook",
        "mac book",
        "imac",
        "macos",
        "mac os",
        "sierra",
        "high sierra",
        "finder"
    ],

    "notifications_messages": [
        "notification",
        "notifications",
        "alert",
        "messages",
        "message",
        "imessage"
    ],

    "security_fraud": [
        "phishing",
        "scam",
        "hack",
        "hacked",
        "fraud",
        "security",
        "fake email"
    ]
}


# ==================================================
# CLEAN TEXT
# ==================================================

def clean_text(text):

    text = str(text).lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    # Remove Twitter usernames
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


# ==================================================
# DETECT INTENT
# ==================================================

def detect_intent(text):

    text = str(text).lower()


    # --------------------------------------------------
    # 1. SECURITY / FRAUD
    # --------------------------------------------------

    security_keywords = [
        "phishing",
        "scam",
        "scammers",
        "fake email",
        "suspicious email",
        "suspicious message",
        "fraud",
        "hacked",
        "hack",
        "steal my apple id",
        "steal my password",
        "credit card",
        "verify payment",
    ]

    if any(keyword in text for keyword in security_keywords):
        return "security_fraud"


    # --------------------------------------------------
    # 2. BATTERY / CHARGING
    # --------------------------------------------------

    battery_keywords = [
        "battery",
        "draining",
        "drain",
        "won't charge",
        "wont charge",
        "not charging",
        "doesn't charge",
        "doesnt charge",
        "charging",
        "charger",
        "charging port",
        "usb c charging",
        "usb-c charging",
        "overheating",
    ]

    if any(keyword in text for keyword in battery_keywords):
        return "battery_charging"


    # --------------------------------------------------
    # 3. WI-FI / CONNECTIVITY
    # --------------------------------------------------

    wifi_keywords = [
        "wifi",
        "wi-fi",
        "wireless",
        "bluetooth",
        "network",
        "internet",
        "disconnected",
        "disconnect",
        "connection",
        "connect to",
        "connecting",
        "not connecting",
        "won't connect",
        "wont connect",
    ]

    if any(keyword in text for keyword in wifi_keywords):
        return "wifi_connectivity"


    # --------------------------------------------------
    # 4. iCLOUD / APPLE ID
    # --------------------------------------------------

    icloud_keywords = [
        "icloud",
        "apple id",
        "appleid",
        "apple account",
        "icloud account",
        "password reset",
        "account locked",
        "locked for security",
    ]

    if any(keyword in text for keyword in icloud_keywords):
        return "icloud_apple_id"


    # --------------------------------------------------
    # 5. APP STORE / PURCHASES
    # --------------------------------------------------

    purchase_keywords = [
        "app store",
        "itunes",
        "purchase",
        "purchased",
        "refund",
        "charged",
        "billing",
        "payment",
        "subscription",
        "music purchase",
    ]

    if any(keyword in text for keyword in purchase_keywords):
        return "app_store_purchases"


    # --------------------------------------------------
    # 6. MAC / macOS
    # --------------------------------------------------

    mac_keywords = [
        "macbook",
        "mac book",
        "imac",
        "macos",
        "mac os",
        "mac os x",
        "sierra",
        "high sierra",
        "finder",
        "spinning pizza",
        "pizza of death",
    ]

    if any(keyword in text for keyword in mac_keywords):
        return "mac_macos"


    # --------------------------------------------------
    # 7. NOTIFICATIONS / MESSAGES
    # --------------------------------------------------

    notification_keywords = [
        "notification",
        "notifications",
        "not receiving notifications",
        "messages not",
        "message not",
        "imessage",
        "text messages",
    ]

    if any(keyword in text for keyword in notification_keywords):
        return "notifications_messages"


    # --------------------------------------------------
    # 8. HARDWARE / DEVICE
    # --------------------------------------------------

    hardware_keywords = [
        "screen",
        "display",
        "camera",
        "photos are blurry",
        "fuzzy photos",
        "keyboard",
        "speaker",
        "speaker phone",
        "microphone",
        "touch",
        "touchscreen",
        "button",
        "broken",
        "black screen",
        "volume",
        "brightness",
        "phone turns off",
        "phone shuts off",
        "iphone turns off",
        "iphone shuts off",
        "pocket dial",
        "repair center",
        "repair",
        "replacement",
        "lost iphone",
        "lost phone",
        "apple tv",
    ]

    if any(keyword in text for keyword in hardware_keywords):
        return "hardware_device"


    # --------------------------------------------------
    # 9. APP ISSUES
    # --------------------------------------------------

    app_keywords = [
        "app crashing",
        "apps crashing",
        "app crashes",
        "apps crash",
        "app won't open",
        "app wont open",
        "application",
        "applications",
        "app not working",
        "apps not working",
        "app freezing",
        "apps freezing",
        "pokemon go",
    ]

    if any(keyword in text for keyword in app_keywords):
        return "app_issues"


    # --------------------------------------------------
    # 10. iOS / SOFTWARE
    # --------------------------------------------------

    software_keywords = [
        "ios",
        "ios update",
        "software update",
        "update",
        "bug",
        "glitch",
        "autocorrect",
        "freezing",
        "slow",
        "lag",
    ]

    if any(keyword in text for keyword in software_keywords):
        return "software_ios"


    return "unknown"


# ==================================================
# RETRIEVE
# ==================================================

def retrieve(query, k=5):

    intent = detect_intent(query)

    cleaned_query = clean_text(query)

    print(f"\nDetected intent: {intent}")


    # --------------------------------------------------
    # FILTER DATASET BY INTENT
    # --------------------------------------------------

    if intent != "unknown":

        keywords = INTENTS[intent]

        pattern = "|".join(
            re.escape(keyword)
            for keyword in keywords
        )

        mask = (
            df["customer_message"]
            .astype(str)
            .str.lower()
            .str.contains(
                pattern,
                regex=True,
                na=False
            )
        )

        filtered = df[mask].copy()

    else:

        filtered = df.copy()


    # --------------------------------------------------
    # FALLBACK
    # --------------------------------------------------

    if len(filtered) < k:
        filtered = df.copy()


    # --------------------------------------------------
    # VECTORIZE QUERY
    # --------------------------------------------------

    query_vector = vectorizer.transform(
        [cleaned_query]
    )


    # --------------------------------------------------
    # VECTORIZE CANDIDATES
    # --------------------------------------------------

    filtered_vectors = vectorizer.transform(
        filtered["clean_customer_message"]
    )


    # --------------------------------------------------
    # COSINE SIMILARITY
    # --------------------------------------------------

    similarities = cosine_similarity(
        query_vector,
        filtered_vectors
    )[0]


    # --------------------------------------------------
    # TOP K
    # --------------------------------------------------

    top_indices = similarities.argsort()[-k:][::-1]


    results = []

    for index in top_indices:

        row = filtered.iloc[index]

        results.append({

            "similarity": round(
                float(similarities[index]),
                4
            ),

            "customer_message":
                row["customer_message"],

            "support_response":
                row["support_response"]

        })


    return intent, results


# ==================================================
# DEMO TEST
# ==================================================

if __name__ == "__main__":

    queries = [

        "Why is my iPhone battery draining so fast?",

        "My iPhone won't connect to WiFi",

        "I can't access my iCloud account",

        "My apps keep crashing after the iOS update",

        "I want a refund for an App Store purchase",

        "My MacBook is running very slowly",

        "I'm not receiving any notifications",

        "My iPhone screen is completely black",

        "I received a suspicious email asking for my Apple ID",

        "My iPhone won't charge anymore"

    ]


    for query in queries:

        print("\n" + "=" * 80)

        print("QUERY:", query)

        print("=" * 80)

        intent, results = retrieve(
            query,
            k=3
        )

        for i, result in enumerate(
            results,
            start=1
        ):

            print(
                f"\n--- Result {i} ---"
            )

            print(
                "Similarity:",
                result["similarity"]
            )

            print(
                "Customer:",
                result["customer_message"]
            )

            print(
                "AppleSupport:",
                result["support_response"]
            )