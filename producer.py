import json
import time
import pandas as pd
from kafka import KafkaProducer

TOPIC = "click-events"
SLEEP_PER_SECOND = 0.4   # 1 second of data replayed in 0.4 real seconds

producer = KafkaProducer(
    bootstrap_servers="127.0.0.1:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8"),
)

df = pd.read_csv("click_data.csv", parse_dates=["timestamp"])
df["sec"] = df["timestamp"].dt.floor("s")   # group each click into its second

sent = 0
for sec, group in df.groupby("sec"):
    for row in group.itertuples():
        event = {
            "timestamp": row.timestamp.isoformat(),
            "user_id": row.user_id,
            "page": row.page,
            "device": row.device,
            "response_ms": None if pd.isna(row.response_ms) else float(row.response_ms),
            "is_error": int(row.is_error),
            "label": int(row.label),
            "anomaly_type": row.anomaly_type,
        }
        producer.send(TOPIC, value=event)
        sent += 1
    time.sleep(SLEEP_PER_SECOND)
    if sec.second == 0:
        print(f"{sec} -> total sent: {sent}")

producer.flush()
print("Done. Total clicks sent:", sent)