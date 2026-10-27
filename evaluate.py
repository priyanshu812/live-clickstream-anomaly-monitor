"""Contamination tuning: precision / recall / F1 (answer key is only used here)"""
import argparse
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from features import make_windows, FEATURE_COLS
from train_model import train

ap = argparse.ArgumentParser()
ap.add_argument("--csv", default="click_data.csv")
a = ap.parse_args()

w = make_windows(pd.read_csv(a.csv), with_labels=True)
y = w["is_anomaly"]
print(f"Windows: {len(w)} | truly anomalous windows: {y.sum()} ({y.mean():.1%})\n")

rows = []
for c in [0.04, 0.06, 0.10]:
    m = train(w, c)
    pred = (m.predict(w[FEATURE_COLS]) == -1).astype(int)
    rows.append({"contamination": c,
                 "precision": precision_score(y, pred, zero_division=0),
                 "recall": recall_score(y, pred, zero_division=0),
                 "f1": f1_score(y, pred, zero_division=0)})
    # type-wise recall (bot / spike / slow)
    for t in ["bot", "spike", "slow"]:
        mask = w["anomaly_type"] == t
        if mask.any():
            print(f"  c={c}: {t:5s} recall = {pred[mask].mean():.2f}  ({mask.sum()} windows)")
res = pd.DataFrame(rows).round(3)
print("\n", res.to_string(index=False))
res.to_csv("eval_results.csv", index=False)
