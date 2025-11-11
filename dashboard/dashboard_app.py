"""Streamlit dashboard for FedShield."""
import os
import sys
import streamlit as st
import requests
import pandas as pd
import time

# Ensure project root is on sys.path so `dashboard.*` imports resolve when run via Streamlit
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.components import metrics_cards, threat_table, model_charts, alerts_feed

st.set_page_config(page_title='FedShield Dashboard', layout='wide', initial_sidebar_state='collapsed')
st.markdown('<style>body{background:#0A0A0A;color:#e6f7ff;}</style>', unsafe_allow_html=True)

API_BASE = 'http://localhost:5000/api'


@st.cache_data(ttl=5)
def fetch_threats():
    try:
        r = requests.get(f'{API_BASE}/threats', timeout=3)
        return pd.DataFrame(r.json()) if r.status_code == 200 else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_summary():
    try:
        r = requests.get(f'{API_BASE}/system_summary', timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


st.title('FedShield — Federated Threat Isolation & Detection')
summary_col, table_col = st.columns([1, 3])

with summary_col:
    summary = fetch_summary()
    metrics_cards.render(summary)

with table_col:
    df = fetch_threats()
    threat_table.render(df)

st.markdown('---')
chart_col, alerts_col = st.columns([2,1])
with chart_col:
    threats_df = fetch_threats()
    model_charts.render(threats_df)

with alerts_col:
    alerts_feed.render()

st.markdown('<div style="position:fixed;bottom:10px;right:10px;color:#0ef">Auto-refresh every 5s</div>', unsafe_allow_html=True)

# Replace blocking infinite loop with a session-state timed rerun to refresh every 5 seconds.
# This avoids a busy/blocking loop and is more stable in Streamlit.
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()
else:
    elapsed = time.time() - st.session_state['last_refresh']
    if elapsed > 5:
        st.session_state['last_refresh'] = time.time()
        st.experimental_rerun()
