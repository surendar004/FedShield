import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Dict

def render(df: pd.DataFrame, key: Optional[str] = None):
    if df is None or df.empty:
        st.info('No data for charts')
        return

    # Derive threat categories client-side without changing backend logic.
    # These lightweight heuristics only affect visualization.
    df2 = df.copy()
    for col in ['cpu_pct', 'net_bytes', 'file_access_count']:
        if col in df2.columns:
            df2[col] = pd.to_numeric(df2[col], errors='coerce')
    if 'file_path' not in df2.columns:
        df2['file_path'] = ''
    if 'is_threat' not in df2.columns:
        df2['is_threat'] = False

    is_threat = df2['is_threat'] == True
    file_path = df2['file_path'].astype(str).str.lower()
    net_bytes = df2.get('net_bytes', pd.Series([0]*len(df2))).fillna(0)
    file_access = df2.get('file_access_count', pd.Series([0]*len(df2))).fillna(0)

    # Heuristic categorization
    malware = is_threat & (
        file_path.str.contains(r'\.exe|malware|virus|trojan', regex=True) | (file_access > 25)
    )
    unauthorized = is_threat & (net_bytes.between(2_00_000, 1_000_000, inclusive='left'))
    data_leak = is_threat & (net_bytes >= 1_000_000)
    phishing = is_threat & file_path.str.contains(r'\.htm|\.html|email|phish', regex=True)
    # Anything else flagged becomes system anomaly
    remaining = is_threat & ~(malware | unauthorized | phishing | data_leak)

    categories: Dict[str, int] = {
        'Malware detection': int(malware.sum()),
        'Unauthorized access': int(unauthorized.sum()),
        'Phishing alerts': int(phishing.sum()),
        'Data leaks': int(data_leak.sum()),
        'System anomalies': int(remaining.sum()),
    }

    total = sum(categories.values())
    if total == 0:
        st.info('No suspicious activity detected yet.')
        return

    labels = list(categories.keys())
    values = list(categories.values())

    # Specified neon palette
    colors = ['#00FFFF', '#FFA500', '#FF1E56', '#00FF88', '#B517FF']

    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=values,
                hole=0.45,
                marker=dict(colors=colors, line=dict(color='#0b1226', width=2)),
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>',
                sort=False,
                direction='clockwise',
                textinfo='none',
                texttemplate='<span style="color:#000000;text-shadow:0 0 8px rgba(0,255,255,0.6); font-weight:600;">%{value}<br>%{percent}</span>',
                textfont=dict(color='#000000', size=12, family='Inter'),
                insidetextfont=dict(color='#000000', size=16, family='Inter'),
                outsidetextfont=dict(color='#000000', size=13, family='Inter'),
                pull=[0.04 if v == max(values) else 0 for v in values],
            )
        ]
    )
    fig.update_traces(textposition='inside', selector=dict(type='pie'))

    fig.update_layout(
        title=dict(
            text='Suspicious Activity Segmentation',
            font=dict(color='#E0F7FF', size=16, family='Inter'),
            x=0.02,
        ),
        showlegend=True,
        legend=dict(
            font=dict(color='#E0F7FF', size=11),
            bgcolor='rgba(2,6,23,0.0)',
            bordercolor='rgba(148,163,184,0.2)',
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
    )

    st.plotly_chart(fig, use_container_width=True, key=key or 'model_charts_pie')
    
    # Add time-series chart if timestamp data is available
    time_col = None
    for col in ['timestamp', 'received_at']:
        if col in df2.columns:
            time_col = col
            break
    
    if time_col:
        try:
            # Convert timestamp to datetime
            df2[time_col] = pd.to_datetime(df2[time_col], errors='coerce', utc=True)
            df2 = df2.dropna(subset=[time_col])
            
            if not df2.empty:
                # Group by time (hourly)
                df2['hour'] = df2[time_col].dt.floor('H')
                hourly_threats = df2[df2['is_threat'] == True].groupby('hour').size().reset_index(name='count')
                hourly_all = df2.groupby('hour').size().reset_index(name='total')
                
                # Create time-series chart
                fig_time = go.Figure()
                
                # Add threat line
                if not hourly_threats.empty:
                    fig_time.add_trace(go.Scatter(
                        x=hourly_threats['hour'],
                        y=hourly_threats['count'],
                        mode='lines+markers',
                        name='Threats',
                        line=dict(color='#FF1E56', width=3),
                        marker=dict(size=8, color='#FF1E56'),
                        fill='tozeroy',
                        fillcolor='rgba(255, 30, 86, 0.1)'
                    ))
                
                # Add total events line
                if not hourly_all.empty:
                    fig_time.add_trace(go.Scatter(
                        x=hourly_all['hour'],
                        y=hourly_all['total'],
                        mode='lines+markers',
                        name='Total Events',
                        line=dict(color='#00FFFF', width=2, dash='dash'),
                        marker=dict(size=6, color='#00FFFF')
                    ))
                
                fig_time.update_layout(
                    title=dict(
                        text='Threat Activity Over Time',
                        font=dict(color='#E0F7FF', size=16, family='Inter'),
                        x=0.02,
                    ),
                    xaxis=dict(
                        title='Time',
                        title_font=dict(color='#B0BEC5', size=12),
                        tickfont=dict(color='#B0BEC5', size=10),
                        gridcolor='rgba(71, 85, 105, 0.2)',
                    ),
                    yaxis=dict(
                        title='Count',
                        title_font=dict(color='#B0BEC5', size=12),
                        tickfont=dict(color='#B0BEC5', size=10),
                        gridcolor='rgba(71, 85, 105, 0.2)',
                    ),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    legend=dict(
                        font=dict(color='#E0F7FF', size=11),
                        bgcolor='rgba(2,6,23,0.0)',
                    ),
                    margin=dict(l=50, r=20, t=50, b=50),
                    hovermode='x unified',
                    height=350,
                )
                
                st.plotly_chart(fig_time, use_container_width=True, key=(key or 'model_charts') + '_timeseries')
        except Exception as e:
            # Silently fail if time-series chart can't be created
            pass
