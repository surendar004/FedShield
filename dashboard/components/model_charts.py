import streamlit as st
import plotly.express as px
import pandas as pd

def render(df: pd.DataFrame):
    if df is None or df.empty:
        st.info('No data for charts')
        return
    df2 = df.copy()
    df2['timestamp'] = pd.to_datetime(df2['timestamp'], errors='coerce')
    fig = px.histogram(df2, x='cpu_pct', nbins=20, title='CPU usage distribution')
    st.plotly_chart(fig, use_container_width=True)
