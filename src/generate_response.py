import sys
import re
import pickle
from pathlib import Path

import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

with open(MODEL_DIR / "tfidf_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(MODEL_DIR / "retriever.pkl", "rb") as f:
    retriever = pickle.load(f)

with open(MODEL_DIR / "retrieval_data.pkl", "rb") as f:
    df = pickle.load(f)


def clean_text(text):
    if not text:
        return ""

    text = re.sub(r'@\w+', '', str(text))
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokens(text):
    return set(
        re.findall(
            r'\b[a-z0-9]+\b',
            text.lower()
        )
    )


def domain_score(query, candidate):
    """
    Reward candidates containing the important words
    from the customer's query.
    """

    q = tokens(query)
    c = tokens(candidate)

    if not q:
        return 0.0

    important = {
        "battery",
        "charging",
        "charger",
        "wifi",
        "wireless",
        "internet",
        "icloud",
        "appleid",
        "password",
        "login",
        "refund",
        "purchase",
        "purchased",
        "payment",
        "billing",
        "subscription",
        "itunes",
        "appstore",
        "app",
        "apps",
        "crash",
        "crashing",
        "ios",
        "update",
        "upgrade",
        "macbook",
        "macos",
        "imac",
        "finder",
        "notification",
        "notifications",
        "message",
        "messages",
        "screen",
        "display",
        "camera",
        "keyboard"
    }

    q_important = q.intersection(important)

    if not q_important:
        return 0.0

    overlap = q_important.intersection(c)

    return len(overlap) / len(q_important)


def retrieve_best(query, k=30):

    cleaned_query = clean_text(query)

    query_vector = vectorizer.transform(
        [cleaned_query]
    )

    distances, indices = retriever.kneighbors(
        query_vector,
        n_neighbors=k
    )

    candidates = []

    for distance, index in zip(
        distances[0],
        indices[0]
    ):

        row = df.iloc[index]

        customer_message = str(
            row["customer_message"]
        )

        similarity = 1 - float(distance)

        keyword_score = domain_score(
            cleaned_query,
            customer_message
        )

        # Semantic similarity + important keyword matching
        final_score = (
            0.65 * similarity +
            0.35 * keyword_score
        )

        candidates.append({
            "customer_message": customer_message,
            "support_response": str(
                row["support_response"]
            ),
            "similarity": similarity,
            "keyword_score": keyword_score,
            "final_score": final_score
        })

    candidates.sort(
        key=lambda x: x["final_score"],
        reverse=True
    )

    return candidates


def detect_intent(query):

    q = query.lower()

    if any(x in q for x in [
        "refund",
        "purchase",
        "purchased",
        "payment",
        "billing",
        "subscription",
        "itunes",
        "app store",
        "appstore"
    ]):
        return "app_store_purchases"

    if any(x in q for x in [
        "battery",
        "charging",
        "charger",
        "draining",
        "drain"
    ]):
        return "battery_charging"

    if any(x in q for x in [
        "wifi",
        "wi-fi",
        "wireless",
        "internet",
        "network"
    ]):
        return "wifi_connectivity"

    if any(x in q for x in [
        "icloud",
        "apple id",
        "appleid",
        "password",
        "login",
        "sign in",
        "locked out"
    ]):
        return "icloud_apple_id"

    if any(x in q for x in [
        "macbook",
        "mac book",
        "macos",
        "mac os",
        "imac",
        "finder"
    ]):
        return "mac_macos"

    if any(x in q for x in [
        "notification",
        "notifications",
        "message",
        "messages",
        "imessage"
    ]):
        return "notifications_messages"

    if any(x in q for x in [
        "screen",
        "display",
        "camera",
        "keyboard",
        "speaker",
        "microphone"
    ]):
        return "hardware_device"

    if any(x in q for x in [
        "app",
        "apps",
        "application",
        "crash",
        "crashing"
    ]):
        return "app_issues"

    if any(x in q for x in [
        "ios",
        "update",
        "upgrade",
        "software"
    ]):
        return "software_ios"

    return "unknown"


def generate_response(query):

    intent = detect_intent(query)

    results = retrieve_best(
        query,
        k=30
    )

    if not results:
        return {
            "intent": intent,
            "response": (
                "I'm sorry, but I couldn't find "
                "a relevant support response."
            ),
            "source": None,
            "similarity": 0.0,
            "final_score": 0.0
        }

    best = results[0]

    response = clean_text(
        best["support_response"]
    )

    if not response:
        response = (
            "Thanks for reaching out. "
            "We're here to help. Please contact "
            "Apple Support so we can look into "
            "this issue further."
        )

    return {
        "intent": intent,
        "response": response,
        "source": clean_text(
            best["customer_message"]
        ),
        "similarity": best["similarity"],
        "final_score": best["final_score"]
    }


if __name__ == "__main__":

    query = input("\nCustomer: ")

    result = generate_response(query)

    print("\n" + "=" * 70)
    print("DETECTED INTENT:", result["intent"])
    print("SIMILARITY:", round(result["similarity"], 4))
    print("FINAL SCORE:", round(result["final_score"], 4))
    print("=" * 70)

    print("\nCUSTOMER QUERY:")
    print(query)

    print("\nGENERATED SUPPORT RESPONSE:")
    print(result["response"])

    print("\nRETRIEVED FROM SIMILAR CUSTOMER ISSUE:")
    print(result["source"])