import pandas as pd
from pathlib import Path

INPUT_FILE = Path("data/twcs.csv")
OUTPUT_FILE = Path("data/applesupport_raw.csv")

CHUNK_SIZE = 100_000

print("Starting AppleSupport extraction...")
print(f"Reading: {INPUT_FILE}")

chunks = []
total_rows = 0
apple_rows = 0

for chunk in pd.read_csv(
    INPUT_FILE,
    chunksize=CHUNK_SIZE,
    low_memory=False
):
    total_rows += len(chunk)

    # Keep tweets sent by AppleSupport
    apple_chunk = chunk[
        chunk["author_id"].astype(str).str.strip().str.lower()
        == "applesupport"
    ]

    if not apple_chunk.empty:
        chunks.append(apple_chunk)
        apple_rows += len(apple_chunk)

    print(
        f"Processed: {total_rows:,} rows | "
        f"AppleSupport: {apple_rows:,}"
    )

if chunks:
    applesupport_df = pd.concat(chunks, ignore_index=True)
    applesupport_df.to_csv(OUTPUT_FILE, index=False)

    print("\nExtraction complete!")
    print(f"AppleSupport rows: {len(applesupport_df):,}")
    print(f"Saved to: {OUTPUT_FILE}")
else:
    print("No AppleSupport tweets were found.")