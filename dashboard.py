import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

# ---- Postgres settings (read password from .env file) ----
load_dotenv()
PG_PASSWORD = os.environ["POSTGRES_PASSWORD"]

st.set_page_config(page_title="Clickstream Anomaly Monitor", layout="wide")

# Fonts: Allura (subtitle), Playfair Display (headings), Inter (body)
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Allura&family=Inter:wght@400;500&family=Playfair+Display:wght@600&display=swap');
html, body, [class*="css"], .stMarkdown, .stDataFrame { font-family: 'Inter', sans-serif; }
h1, h2, h3 { font-family: 'Playfair Display', serif !important; }
.sub { font-family: 'Allura', cursive; font-size: 2rem; color: #e0a458; margin-top: -10px; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_engine():
    url = URL.create("postgresql+psycopg2", username="monitor", password=PG_PASSWORD,
                     host="127.0.0.1", port=5433, database="clickstream")
    return create_engine(url)


st.title("Live Clickstream Anomaly Monitor")
st.markdown('<div class="sub">Kafka · IsolationForest · PostgreSQL</div>', unsafe_allow_html=True)

n_windows = st.slider("Number of latest windows to show", 30, 360, 120, step=10)


@st.fragment(run_every=2)  # only this fragment refreshes every 2 seconds
def live_view():
    q = text("SELECT * FROM window_results ORDER BY window_start DESC LIMIT :n")
    df = pd.read_sql(q, get_engine(), params={"n": n_windows})
    if df.empty:
        st.info("Table is empty. Start the processor and producer first.")
        return
    df = df.sort_values("window_start")
    anom = df[df["is_anomaly"]]

    c1, c2, c3 = st.columns(3)
    c1.metric("Windows shown", len(df))
    c2.metric("Anomalies", len(anom))
    c3.metric("Latest score", f"{df['score'].iloc[-1]:.3f}")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["window_start"], y=df["score"], mode="lines",
                             name="Anomaly score", line=dict(color="#e0a458", width=2)))
    fig.add_trace(go.Scatter(x=anom["window_start"], y=anom["score"], mode="markers",
                             name="Anomaly", marker=dict(color="#e5534b", size=10)))
    fig.add_hline(y=0, line_dash="dash", line_color="#8a7a6a",
                  annotation_text="score < 0 => anomaly")
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", height=380,
                      xaxis_title="Window start (event time)", yaxis_title="Score")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Anomaly alerts")
    if anom.empty:
        st.success("No anomalies in this range.")
    else:
        cols = ["window_start", "clicks", "unique_users", "avg_response_ms",
                "error_rate", "max_user_clicks", "score"]
        st.dataframe(anom.sort_values("window_start", ascending=False)[cols],
                     use_container_width=True, hide_index=True)


live_view()