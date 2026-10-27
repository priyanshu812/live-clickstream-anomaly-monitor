"""
features.py - imported by both the training pipeline and the live processor.
There is only one way to compute window features: window_features().
"""
import pandas as pd

WINDOW_SEC = 5
FEATURE_COLS = ["clicks", "unique_users", "avg_response_ms",
                "error_rate", "max_user_clicks"]

# --- CSV column names. Match these to your CSV; change only here ---
COL_TS = "timestamp"
COL_USER = "user_id"
COL_RESP = "response_ms"
COL_ERROR = "is_error"          # 0/1
COL_LABEL = "label"             # used only for evaluation
COL_ATYPE = "anomaly_type"      # used only for evaluation


def window_features(clicks: pd.DataFrame) -> dict:
    """Given a window of clicks (DataFrame), return a dict of 5 features.
    The live processor also builds a DataFrame of clicks and calls this."""
    n = len(clicks)
    if n == 0:
        return {c: 0.0 for c in FEATURE_COLS}
    return {
        "clicks": float(n),
        "unique_users": float(clicks[COL_USER].nunique()),
        "avg_response_ms": float(clicks[COL_RESP].mean()),
        "error_rate": float(clicks[COL_ERROR].mean()),
        "max_user_clicks": float(clicks[COL_USER].value_counts().max()),
    }


def window_id(ts) -> int:
    """Timestamp -> window number (5-second bucket). Used in the live pipeline too."""
    return int(pd.Timestamp(ts).timestamp()) // WINDOW_SEC


def make_windows(df: pd.DataFrame, with_labels: bool = False) -> pd.DataFrame:
    """Full CSV -> window table (for training and evaluation)."""
    df = df.copy()
    df["_wid"] = pd.to_datetime(df[COL_TS]).astype("int64") // 10**9 // WINDOW_SEC
    rows = []
    for wid, g in df.groupby("_wid"):
        r = window_features(g)
        r["window_start"] = int(wid) * WINDOW_SEC
        if with_labels:
            r["is_anomaly"] = int((g[COL_LABEL] == 1).any())
            bad = g.loc[g[COL_LABEL] == 1, COL_ATYPE]
            r["anomaly_type"] = bad.mode().iloc[0] if len(bad) else "normal"
        rows.append(r)
    return pd.DataFrame(rows)
