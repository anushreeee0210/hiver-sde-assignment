import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "retriever_evaluation.csv"
)

df = pd.read_csv(INPUT_FILE)

errors = df[df["correct"] == False]

print("=" * 80)
print("INTENT CLASSIFICATION ERROR ANALYSIS")
print("=" * 80)

print(f"\nTotal examples: {len(df)}")
print(f"Errors: {len(errors)}")

print("\n" + "=" * 80)
print("CONFUSION PAIRS")
print("=" * 80)

confusion = (
    errors
    .groupby(["expected_intent", "predicted_intent"])
    .size()
    .reset_index(name="count")
    .sort_values("count", ascending=False)
)

print(confusion.to_string(index=False))

print("\n" + "=" * 80)
print("INCORRECT EXAMPLES")
print("=" * 80)

for i, (_, row) in enumerate(errors.iterrows(), 1):

    print(f"\n--- Error {i} ---")
    print(f"Customer: {row['customer_message']}")
    print(f"Expected : {row['expected_intent']}")
    print(f"Predicted: {row['predicted_intent']}")

print("\n" + "=" * 80)