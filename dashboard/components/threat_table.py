import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    if df is None or df.empty:
        st.info('No threats detected yet')
        return
    df_display = df.copy()
    if 'quarantined_path' in df_display.columns:
        df_display['quarantined_path'] = df_display['quarantined_path'].fillna('')
    st.dataframe(df_display)
