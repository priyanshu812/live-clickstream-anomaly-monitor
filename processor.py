import json
import os
import time

import joblib
import pandas as pd
import psycopg2
from dotenv import load_dotenv
from kafka import KafkaConsumer

from features import window_id, window_features, WINDOW_SEC

load_dotenv()

TOPIC = "click-events"
IDLE_FLUSH_SEC = 5
# label and anomaly_type are deliberately excluded: they are the answer key
CLICK_COLS = ["timestamp", "user_id", "page", "device", "response_ms", "is_error"]

# --- Model ---
bundle = joblib.load("anomaly_model.pkl")
model, FEATURES = bundle["model"], bundle["features"]

# --- PostgreSQL ---
conn = psycopg2.connect(
    host="127.0.0.1", port=5433,
    dbname="clickstream", user="monitor", password=os.environ["POSTGRES_PASSWORD"],
)
conn.autocommit = True
with conn.cursor() as cur:
    cur.execute("""
        CREATE TABLE IF NOT EXISTS window_results (
            window_start     TIMESTAMP PRIMARY KEY,
            clicks           INTEGER,
            unique_users     INTEGER,
            avg_response_ms  DOUBLE PRECISION,
            error_rate       DOUBLE PRECISION,
            max_user_clicks  INTEGER,
            is_anomaly       BOOLEAN,
            score            DOUBLE PRECISION
        )
    """)

INSERT_SQL = """
INSERT INTO window_results
(window_start, clicks, unique_users, avg_response_ms, error_rate,
 max_user_clicks, is_anomaly, score)
VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
ON CONFLICT (window_start) DO NOTHING
"""


def process_window(rows):
    df = pd.DataFrame(rows)[CLICK_COLS]
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    feats = window_features(df)                      # same feature file used for training
    X = pd.DataFrame([[feats[c] for c in FEATURES]], columns=FEATURES)
    pred = model.predict(X)[0]                       # -1 = anomaly
    score = float(model.decision_function(X)[0])     # more negative = more anomalous
    window_start = df["timestamp"].min().floor(f"{WINDOW_SEC}s").to_pydatetime()
    with conn.cursor() as cur:
        cur.execute(INSERT_SQL, (
            window_start, int(feats["clicks"]), int(feats["unique_users"]),
            float(feats["avg_response_ms"]), float(feats["error_rate"]),
            int(feats["max_user_clicks"]), bool(pred == -1), score,
        ))
    print(f"{window_start} | anomaly={pred == -1} | score={score:.3f}")


# --- Kafka ---
consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="127.0.0.1:9092",
    value_deserializer=lambda m: json.loads(m.decode("utf-8")),
    group_id=None,
    auto_offset_reset="latest",
)
while not consumer.assignment():
    consumer.poll(timeout_ms=500)
print("Processor ready. Start the producer now.")

current_wid, buffer = None, []
last_msg = time.time()

while True:
    batch = consumer.poll(timeout_ms=1000)
    if not batch:
        # no clicks for a while -> score the remaining window
        if buffer and time.time() - last_msg > IDLE_FLUSH_SEC:
            process_window(buffer)
            current_wid, buffer = None, []
        continue
    last_msg = time.time()
    for msgs in batch.values():
        for msg in msgs:
            click = msg.value
            wid = window_id(pd.Timestamp(click["timestamp"]))
            if current_wid is None:
                current_wid = wid
            if wid < current_wid:                    # late click, ignore
                continue
            if wid > current_wid:                    # new window -> close the old one
                process_window(buffer)
                current_wid, buffer = wid, []
            buffer.append(click)