import pandas as pd

FILE = "data/applesupport_raw.csv"

df = pd.read_csv(FILE, low_memory=False)

print("\n===== DATASET SHAPE =====")
print(df.shape)

print("\n===== COLUMNS =====")
print(df.columns.tolist())

print("\n===== INBOUND DISTRIBUTION =====")
print(df["inbound"].value_counts(dropna=False))

print("\n===== MISSING VALUES =====")
print(df.isna().sum())

print("\n===== CUSTOMER MESSAGES =====")
print(
    df[df["inbound"] == True]["text"]
    .head(10)
    .to_string(index=False)
)

print("\n===== APPLESUPPORT RESPONSES =====")
print(
    df[df["inbound"] == False]["text"]
    .head(10)
    .to_string(index=False)
)