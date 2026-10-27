"""CSV -> windows -> IsolationForest -> anomaly_model.pkl"""
import argparse
import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest
from features import make_windows, FEATURE_COLS


def train(windows: pd.DataFrame, contamination: float) -> IsolationForest:
    model = IsolationForest(n_estimators=200, contamination=contamination,
                            random_state=42)
    model.fit(windows[FEATURE_COLS])          # labels are never used
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="click_data.csv")
    ap.add_argument("--contamination", type=float, default=0.06)
    ap.add_argument("--out", default="anomaly_model.pkl")
    a = ap.parse_args()

    w = make_windows(pd.read_csv(a.csv))      # with_labels=False
    m = train(w, a.contamination)
    joblib.dump({"model": m, "features": FEATURE_COLS}, a.out)
    flagged = (m.predict(w[FEATURE_COLS]) == -1).sum()
    print(f"{len(w)} windows, {flagged} flagged, saved -> {a.out}")
