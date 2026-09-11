import os
import sys
import pandas as pd

# Make src importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from intent_retriever import detect_intent


# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOLDEN_FILE = os.path.join(BASE_DIR, "data", "golden_set.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "data", "retriever_evaluation.csv")


df = pd.read_csv(GOLDEN_FILE)

print("=" * 80)
print("GOLDEN SET INTENT EVALUATION")
print("=" * 80)

correct = 0
total = len(df)

results = []

for _, row in df.iterrows():

    query = str(row["customer_message"])
    expected = str(row["intent"])

    predicted = detect_intent(query)

    is_correct = predicted == expected

    if is_correct:
        correct += 1

    results.append({
        "customer_message": query,
        "expected_intent": expected,
        "predicted_intent": predicted,
        "correct": is_correct
    })


accuracy = correct / total if total > 0 else 0


print()
print(f"Total examples: {total}")
print(f"Correct: {correct}")
print(f"Incorrect: {total - correct}")
print(f"Intent accuracy: {accuracy:.2%}")


results_df = pd.DataFrame(results)

results_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print("Saved evaluation results to:")
print(OUTPUT_FILE)