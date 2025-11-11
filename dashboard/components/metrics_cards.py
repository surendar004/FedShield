import streamlit as st

def render(summary: dict):
    a, b, c, d = st.columns(4)
    a.metric('Active Clients', summary.get('clients', 0))
    b.metric('Threats', summary.get('threats', 0))
    c.metric('Isolations', summary.get('isolations', 0))
    d.metric('Accuracy (est)', '90%')
