import streamlit as st
import requests
import time

API_BASE = 'http://localhost:5000/api'


def render():
    st.header('Alerts')
    try:
        r = requests.get(f'{API_BASE}/threats', timeout=3)
        alerts = r.json() if r.status_code == 200 else []
    except Exception:
        alerts = []
    if not alerts:
        st.write('No alerts')
        return
    for a in alerts[-10:][::-1]:
        st.markdown(f"**{a.get('timestamp')}** — {a.get('client_id')} — Threat={a.get('is_threat')} — {a.get('action')}")
