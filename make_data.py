import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
SECONDS = 1800
start = pd.Timestamp("2026-10-26 10:00:00")

pages = ["/home", "/menu", "/product", "/cart", "/checkout"]
page_p = [0.30, 0.30, 0.25, 0.10, 0.05]
devices = ["mobile", "desktop", "tablet"]
dev_p = [0.60, 0.35, 0.05]
users = [f"u{i:03d}" for i in range(200)]

# (start_second, duration_seconds, anomaly_type)
episodes = [(300, 20, "bot"), (600, 15, "spike"), (900, 20, "slow"),
            (1200, 20, "bot"), (1450, 15, "spike"), (1650, 20, "slow")]

rows = []
for s in range(SECONDS):
    kind = "normal"
    for st, ln, k in episodes:
        if st <= s < st + ln:
            kind = k

    rate = 40 if kind == "spike" else 6
    for _ in range(rng.poisson(rate)):
        rt = rng.lognormal(np.log(120), 0.25)
        err_p = 0.01
        if kind == "slow":
            rt *= 8
            err_p = 0.25
        rows.append({
            "timestamp": start + pd.Timedelta(seconds=s + rng.random()),
            "user_id": rng.choice(users),
            "page": rng.choice(pages, p=page_p),
            "device": rng.choice(devices, p=dev_p),
            "response_ms": round(rt, 1),
            "is_error": int(rng.random() < err_p),
            "label": 0 if kind == "normal" else 1,
            "anomaly_type": kind,
        })

    if kind == "bot":  # single user generating 40 clicks/sec
        for _ in range(40):
            rows.append({
                "timestamp": start + pd.Timedelta(seconds=s + rng.random()),
                "user_id": "bot_77", "page": "/menu", "device": "desktop",
                "response_ms": round(rng.lognormal(np.log(60), 0.1), 1),
                "is_error": 0, "label": 1, "anomaly_type": "bot",
            })

df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
df.loc[rng.choice(len(df), 5, replace=False), "response_ms"] = np.nan  # simulate sensor gaps

df.to_csv("click_data.csv", index=False)
print(df.head())
print("Rows:", len(df))
print(df.anomaly_type.value_counts())