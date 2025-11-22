import streamlit as st
import pandas as pd

def render(df: pd.DataFrame):
    if df is None or df.empty:
        st.info('No signature contributions detected yet.')
        return
    df_display = df.copy()
    if 'quarantined_path' in df_display.columns:
        df_display['quarantined_path'] = df_display['quarantined_path'].fillna('')
    # Extract client metrics (stored as dicts under '_client_metrics' or 'client_metrics')
    def _get_metric(row, key, default=None):
        # row may be dict-like or a scalar
        val = None
        if isinstance(row, dict):
            val = row.get('_client_metrics') or row.get('client_metrics')
        else:
            val = row
        if isinstance(val, dict):
            return val.get(key, default)
        return default

    if '_client_metrics' in df_display.columns or 'client_metrics' in df_display.columns:
        df_display['access_count'] = df_display.apply(lambda r: _get_metric(r.to_dict(), 'access_count', None), axis=1)
        df_display['cpu_percent'] = df_display.apply(lambda r: _get_metric(r.to_dict(), 'cpu_percent', None), axis=1)
        df_display['memory_mb'] = df_display.apply(lambda r: _get_metric(r.to_dict(), 'memory_mb', None), axis=1)

    st.dataframe(df_display)
