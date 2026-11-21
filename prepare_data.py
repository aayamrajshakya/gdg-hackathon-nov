"""Step 1: load dataset, clean it, and lock the 80/20 split.

Outputs (in data/):
  kb_train.parquet    -> 80% split, includes answers (training + RAG index)
  test_inputs.parquet -> 20% split, ONLY ticket_id/subject/body (what the model sees)
  test_labels.parquet -> 20% split, ONLY labels (used by evaluation, never by the model)
The holdout 'answer' column is dropped entirely, so it can never leak.
"""
import os
import pandas as pd
from common import build_text, PRIORITY_MAP

import glob
import sys

# Always resolve paths relative to this script, so it works from any terminal folder.
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
os.makedirs("data", exist_ok=True)

# 1. Load: use ANY .csv you put in data/ (prefers the 20k file), otherwise try Hugging Face.
csvs = sorted(glob.glob(os.path.join("data", "*.csv")))
if csvs:
    preferred = [c for c in csvs if "20k" in c]
    LOCAL_CSV = preferred[0] if preferred else csvs[0]
    df = pd.read_csv(LOCAL_CSV)
    print(f"Loaded local file {LOCAL_CSV}")
else:
    print(f"No .csv found in {os.path.join(HERE, 'data')}")
    print("Files currently in data/:", os.listdir("data"))
    print("Put the downloaded CSV in that folder and run again.")
    sys.exit(1)

print("Rows:", len(df))
print("Columns:", list(df.columns))

# 2. Clean
df = df.dropna(subset=["body", "type", "queue", "priority"]).copy()
df["priority"] = df["priority"].str.lower().map(PRIORITY_MAP).fillna(df["priority"])
df["subject"] = df["subject"].fillna("")
df["text"] = [build_text(s, b) for s, b in zip(df["subject"], df["body"])]

# 3. Look at the data (useful talking points for the demo)
for col in ["type", "queue", "priority", "language"]:
    if col in df.columns:
        print(f"\n--- {col} ---\n{df[col].value_counts()}")
print("\nEmpty subjects:", (df["subject"] == "").sum())

# 4. Official split (same recipe as the problem statement)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
df["ticket_id"] = [f"TKT-{i:05d}" for i in range(len(df))]
split_idx = int(len(df) * 0.8)
kb_train_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()

# 5. Save, keeping the holdout inputs and labels in separate files
kb_train_df.to_parquet("data/kb_train.parquet", index=False)
test_df[["ticket_id", "subject", "body", "text"]].to_parquet("data/test_inputs.parquet", index=False)
label_cols = ["ticket_id", "type", "queue", "priority"] + (["language"] if "language" in df.columns else [])
test_df[label_cols].to_parquet("data/test_labels.parquet", index=False)

print(f"\nSaved {len(kb_train_df)} KB/train rows and {len(test_df)} holdout rows to data/")
