import pickle
import re

from pathlib import Path


MODEL_DIR = Path("models")


# --------------------------------------------------
# Load classifier
# --------------------------------------------------

with open(
    MODEL_DIR / "intent_vectorizer.pkl",
    "rb"
) as f:
    vectorizer = pickle.load(f)


with open(
    MODEL_DIR / "intent_classifier.pkl",
    "rb"
) as f:
    classifier = pickle.load(f)


# --------------------------------------------------
# Clean text
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
# Predict
# --------------------------------------------------

def predict_intent(message):

    cleaned = clean_text(message)

    vector = vectorizer.transform(
        [cleaned]
    )

    probabilities = classifier.predict_proba(
        vector
    )[0]

    prediction = classifier.classes_[
        probabilities.argmax()
    ]

    confidence = probabilities.max()

    return prediction, confidence


# --------------------------------------------------
# Test messages
# --------------------------------------------------

test_messages = [

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


# --------------------------------------------------
# Run tests
# --------------------------------------------------

print("=" * 80)
print("INTENT CLASSIFIER TEST")
print("=" * 80)


for message in test_messages:

    intent, confidence = predict_intent(
        message
    )

    print("\nMessage:")
    print(message)

    print(
        f"Predicted intent: {intent}"
    )

    print(
        f"Confidence: {confidence:.4f}"
    )