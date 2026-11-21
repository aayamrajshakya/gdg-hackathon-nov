"""Step 2: train type/queue/priority classifiers and report holdout metrics.

Run after prepare_data.py. Saves models/triage.joblib and models/metrics.json.
"""
import json
import os
import time

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

from common import TARGETS

os.makedirs("models", exist_ok=True)

train = pd.read_parquet("data/kb_train.parquet")
test_X = pd.read_parquet("data/test_inputs.parquet")
test_y = pd.read_parquet("data/test_labels.parquet")
assert (test_X["ticket_id"].values == test_y["ticket_id"].values).all()

# One shared TF-IDF (word 1-2 grams works for both English and German)
t0 = time.time()
vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=200_000,
                             sublinear_tf=True, strip_accents=None, lowercase=True)
Xtr = vectorizer.fit_transform(train["text"])
Xte = vectorizer.transform(test_X["text"])

models, metrics, preds = {}, {}, {}
for target in TARGETS:
    base = LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced")
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)  # calibrated confidences
    clf.fit(Xtr, train[target])
    models[target] = clf
    preds[target] = clf.predict(Xte)
    metrics[target] = {
        "macro_f1": round(f1_score(test_y[target], preds[target], average="macro"), 4),
        "accuracy": round(accuracy_score(test_y[target], preds[target]), 4),
    }
    print(f"\n===== {target} =====")
    print(classification_report(test_y[target], preds[target], digits=3))

# High-priority Incident recall: of the truly high+Incident tickets, how many did we catch?
true_hi = (test_y["priority"] == "high") & (test_y["type"] == "Incident")
pred_hi = (preds["priority"] == "high") & (preds["type"] == "Incident")
metrics["high_priority_incident_recall"] = round((pred_hi & true_hi).sum() / max(true_hi.sum(), 1), 4)

# Also report plain high-priority recall (what the escalation gate relies on most)
true_high = test_y["priority"] == "high"
metrics["high_priority_recall"] = round(((preds["priority"] == "high") & true_high).sum() / max(true_high.sum(), 1), 4)
metrics["train_seconds"] = round(time.time() - t0, 1)
metrics["n_train"], metrics["n_test"] = len(train), len(test_X)

joblib.dump({"vectorizer": vectorizer, "models": models}, "models/triage.joblib")
with open("models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("\n===== SUMMARY =====")
print(json.dumps(metrics, indent=2))
