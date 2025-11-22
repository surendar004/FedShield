"""Streamlit dashboard for FedShield (dark cybersecurity theme, logic preserved)."""
import os
import sys
import time
import requests
import pandas as pd
import streamlit as st    # pyright: ignore[reportMissingImports]

# Ensure project root is on sys.path so `dashboard.*` imports resolve when run via Streamlit
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dashboard.components import metrics_cards, threat_table, model_charts, alerts_feed, chatbot_widget

# Page config must be first Streamlit call
st.set_page_config(
    page_title="FedShield | Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Global styling (dark cybersecurity theme) — purely presentational
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        background: radial-gradient(circle at top left, #020617, #050a18, #0b1120, #0f172a);
        font-family: 'Inter', sans-serif;
        color: #E0E0E0;
        text-shadow: 0 0 4px rgba(0, 255, 255, 0.15);
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Top nav styled like hero header strip */
    .top-nav {
        display: flex; align-items: center; justify-content: space-between;
        padding: 10px 18px; margin-bottom: 8px;
        border-bottom: 1px solid rgba(34, 211, 238, 0.15);
        background: linear-gradient(180deg, rgba(10,16,40,0.6), rgba(10,16,40,0));
    }
    .nav-links a {
        color: #E0F7FF; text-decoration: none; margin: 0 12px; font-weight: 600; position: relative;
        letter-spacing: 0.3px;
    }
    .nav-links a:hover { color: #00FFFF; text-shadow: 0 0 8px rgba(0,255,255,0.6); }
    .nav-links a:after {
        content: ""; position: absolute; left: 0; bottom: -4px; height: 2px; width: 0;
        background: linear-gradient(90deg,#00FFFF,#0078FF,#FF1E56);
        transition: width .25s ease;
        box-shadow: 0 0 10px rgba(0,255,255,.6);
    }
    .nav-links a:hover:after { width: 100%; }

    /* Hero banner inspired by provided layout */
    .main-header {
        background: rgba(15, 23, 42, 0.6);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(0, 255, 255, 0.22);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 0 40px rgba(0, 255, 255, 0.12);
        position: relative;
    }
    .main-title {
        font-size: 46px;
        font-weight: 800;
        background: linear-gradient(135deg, #00FFFF 0%, #66E0FF 40%, #B517FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        text-align: center;
        letter-spacing: 0.5px;
        text-shadow: 0 0 16px rgba(0,255,255,0.35);
    }
    .main-subtitle {
        color: #B0BEC5;
        text-align: center;
        font-size: 15px;
        margin-top: 10px;
        letter-spacing: 0.4px;
        text-shadow: 0 0 10px rgba(0, 120, 255, 0.4);
    }
    .hero-cta {
        display:flex; gap: 12px; justify-content:center; margin-top: 18px;
    }
    .glow-btn {
        border: 1px solid rgba(0, 255, 255, 0.45);
        color: #E0F7FF; background: rgba(6, 182, 212, 0.12);
        padding: 10px 16px; border-radius: 10px; font-weight: 600; text-decoration: none;
        transition: all .25s ease;
        box-shadow: 0 0 18px rgba(0,255,255,0.25) inset, 0 0 12px rgba(0,255,255,0.2);
        letter-spacing: 0.4px;
    }
    .glow-btn:hover {
        background: rgba(0, 255, 255, 0.2);
        box-shadow: 0 0 22px rgba(0,255,255,0.45);
        transform: translateY(-1px);
    }
    .glow-btn.red {
        border-color: rgba(255,60,60,.45); background: rgba(255,60,60,.1);
        box-shadow: 0 0 18px rgba(255,60,60,.25) inset, 0 0 12px rgba(255,60,60,.2);
        color: #FFE5E5;
    }
    .glow-btn.red:hover { background: rgba(255,60,60,.2); box-shadow: 0 0 22px rgba(255,60,60,.45); }

    /* Fun cybersecurity stickers ribbon */
    .sticker-ribbon {
        display:flex; gap:14px; justify-content:center; margin-top:14px; opacity:.9;
    }
    .sticker {
        background: rgba(2,6,23,.65);
        border: 1px dashed rgba(0,255,255,.35);
        border-radius: 10px; padding: 6px 10px; font-size:12px; color:#E0F7FF;
        box-shadow: 0 0 12px rgba(0,255,255,.18);
        text-shadow: 0 0 6px rgba(0,255,255,.4);
    }

    .section {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(71, 85, 105, 0.25);
        border-radius: 12px;
        padding: 16px 16px 4px 16px;
        box-shadow: inset 0 0 18px rgba(0,255,255,0.07);
    }

    .section-header {
        color: #E0F7FF;
        font-size: 21px;
        font-weight: 700;
        margin: 0 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(71, 85, 105, 0.3);
        letter-spacing: 0.5px;
        text-shadow: 0 0 12px rgba(0,255,255,0.4);
    }

    .hint {
        color: #B0BEC5;
        font-size: 12px;
        text-shadow: 0 0 8px rgba(0, 120, 255, 0.3);
    }

    .threat-amber {
        color: #FFA200 !important;
        text-shadow: 0 0 14px rgba(255, 162, 0, 0.5) !important;
    }

    /* Table & metric readability - bright vibrant colors */
    .stDataFrame thead tr th {
        background: linear-gradient(135deg, rgba(0, 255, 255, 0.25), rgba(0, 120, 255, 0.25)) !important;
        color: #00FFFF !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(0,255,255,0.7), 0 0 20px rgba(0,255,255,0.4);
        border: 1px solid rgba(0, 255, 255, 0.4) !important;
    }
    .stDataFrame tbody tr td {
        color: #E0F7FF !important;
        text-shadow: 0 0 8px rgba(0, 255, 255, 0.3);
        border-color: rgba(0, 255, 255, 0.15) !important;
    }
    /* Metrics cards container enhancement */
    div[data-testid="stMetric"] {
        background: rgba(232, 17, 35, 0.05) !important;
        border: 1px solid rgba(255, 241, 0, 0.2) !important;
        border-radius: 12px !important;
        padding: 16px !important;
        box-shadow: 0 0 20px rgba(255, 241, 0, 0.15), 0 0 40px rgba(232, 17, 35, 0.1), inset 0 0 10px rgba(255, 241, 0, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stMetric"]:hover {
        background: rgba(232, 17, 35, 0.1) !important;
        border-color: rgba(255, 241, 0, 0.4) !important;
        box-shadow: 0 0 30px rgba(255, 241, 0, 0.25), 0 0 60px rgba(232, 17, 35, 0.15), inset 0 0 15px rgba(255, 241, 0, 0.1) !important;
        transform: translateY(-2px);
    }
    
    div[data-testid="metric"] label {
        background: linear-gradient(135deg, #E81123, #FF6B35, #FFF100) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 700 !important;
        text-shadow: 0 0 12px rgba(232, 17, 35, 0.6), 0 0 20px rgba(255, 241, 0, 0.4);
        filter: drop-shadow(0 0 8px rgba(232, 17, 35, 0.5));
        font-size: 13px !important;
    }
    div[data-testid="metric"] .stMetricValue {
        background: linear-gradient(135deg, #E81123, #FF6B35, #FFF100) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 0 20px rgba(232, 17, 35, 0.8), 0 0 30px rgba(255, 241, 0, 0.6), 0 0 40px rgba(255, 241, 0, 0.4);
        font-weight: 900;
        font-size: 2.5rem !important;
        filter: drop-shadow(0 0 12px rgba(232, 17, 35, 0.7)) drop-shadow(0 0 20px rgba(255, 241, 0, 0.5));
        letter-spacing: 0.5px;
    }
    div[data-testid="metric"] .stMetricDelta {
        background: linear-gradient(135deg, #E81123, #FFF100) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        text-shadow: 0 0 15px rgba(232, 17, 35, 0.7), 0 0 25px rgba(255, 241, 0, 0.5);
        font-weight: 700;
        filter: drop-shadow(0 0 10px rgba(255, 241, 0, 0.6));
    }
    
    /* Download button - bright vibrant styling */
    div[data-testid="stDownloadButton"] button,
    div[data-testid="stDownloadButton"] button > span {
        background: linear-gradient(135deg, #00FFFF, #0078FF) !important;
        color: #000000 !important;
        border: 2px solid #00FFFF !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.3), inset 0 0 10px rgba(0,255,255,0.2) !important;
        text-shadow: 0 0 8px rgba(0,0,0,0.8) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stDownloadButton"] button:hover {
        background: linear-gradient(135deg, #00FFFF, #00BFFF) !important;
        box-shadow: 0 0 30px rgba(0,255,255,0.8), 0 0 60px rgba(0,255,255,0.5), inset 0 0 15px rgba(0,255,255,0.3) !important;
        transform: translateY(-2px) scale(1.05) !important;
    }
    
    /* Style all secondary buttons that might be download buttons */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #00FFFF, #0078FF) !important;
        color: #000000 !important;
        border: 2px solid #00FFFF !important;
        border-radius: 12px !important;
        padding: 10px 20px !important;
        font-weight: 700 !important;
        font-size: 14px !important;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        box-shadow: 0 0 20px rgba(0,255,255,0.6), 0 0 40px rgba(0,255,255,0.3), inset 0 0 10px rgba(0,255,255,0.2) !important;
        text-shadow: 0 0 8px rgba(0,0,0,0.8) !important;
        transition: all 0.3s ease !important;
    }
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #00FFFF, #00BFFF) !important;
        box-shadow: 0 0 30px rgba(0,255,255,0.8), 0 0 60px rgba(0,255,255,0.5), inset 0 0 15px rgba(0,255,255,0.3) !important;
        transform: translateY(-2px) scale(1.05) !important;
    }
    .status-positive { color: #00FF88; text-shadow: 0 0 10px rgba(0,255,136,0.4); font-weight: 600; }
    .status-alert { color: #FF3C3C; text-shadow: 0 0 12px rgba(255,60,60,0.5); font-weight: 600; }

    /* Plotly dark compatibility */
    .js-plotly-plot { background: transparent !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 8px; height: 8px; }
    ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.4); }
    ::-webkit-scrollbar-thumb { background: rgba(6, 182, 212, 0.3); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: rgba(6, 182, 212, 0.5); }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(10, 14, 39, 0.95);
        border-right: 1px solid rgba(34, 211, 238, 0.2);
        color: #FFA200;
    }
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
        color: #FFA200;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        text-shadow: 0 0 10px rgba(255, 162, 0, 0.45);
        font-weight: 700;
    }
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p {
        color: #FFD27A !important;
        text-shadow: 0 0 8px rgba(255, 162, 0, 0.35);
    }
    section[data-testid="stSidebar"] .stCheckbox,
    section[data-testid="stSidebar"] .stRadio,
    section[data-testid="stSidebar"] .stSlider {
        color: #FFA200;
    }
    section[data-testid="stSidebar"] input[type="range"]::-webkit-slider-thumb {
        background: #FFA200;
        box-shadow: 0 0 12px rgba(255, 162, 0, 0.6);
    }
    section[data-testid="stSidebar"] .stCheckbox [data-testid="stWidgetLabel"] > span {
        color: #FFD27A !important;
    }
    
    /* Mobile Responsiveness */
    @media (max-width: 768px) {
        .main-title {
            font-size: 28px !important;
            line-height: 1.2 !important;
        }
        .main-subtitle {
            font-size: 12px !important;
        }
        .section {
            padding: 12px !important;
            margin-bottom: 16px !important;
        }
        .section-header {
            font-size: 18px !important;
        }
        .hero-cta {
            flex-direction: column !important;
            gap: 8px !important;
        }
        .glow-btn {
            width: 100% !important;
            text-align: center !important;
        }
        .sticker-ribbon {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        .sticker {
            font-size: 10px !important;
            padding: 4px 8px !important;
        }
        div[data-testid="stMetric"] {
            padding: 12px !important;
            margin-bottom: 12px !important;
        }
        div[data-testid="metric"] .stMetricValue {
            font-size: 1.8rem !important;
        }
        .floating-chatbot-wrapper {
            width: 100% !important;
            max-width: 100% !important;
            right: 0 !important;
            bottom: 0 !important;
            border-radius: 0 !important;
        }
        .floating-chatbot-wrapper .chat-window {
            height: 60vh !important;
            max-height: 60vh !important;
        }
    }
    
    @media (max-width: 480px) {
        .main-title {
            font-size: 22px !important;
        }
        .section-header {
            font-size: 16px !important;
        }
        div[data-testid="metric"] label {
            font-size: 11px !important;
        }
        div[data-testid="metric"] .stMetricValue {
            font-size: 1.5rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

API_BASE = 'http://localhost:5000/api'

@st.cache_data(ttl=5)
def fetch_threats(page=1, per_page=1000):
    """Fetch threats with pagination support."""
    try:
        params = {'page': page, 'per_page': per_page}
        r = requests.get(f'{API_BASE}/threats', params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # Handle both old format (list) and new format (dict with 'data')
            if isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            elif isinstance(data, list):
                return pd.DataFrame(data)
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch threats: {str(e)}")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return pd.DataFrame()

def fetch_summary():
    """Fetch system summary with error handling."""
    try:
        r = requests.get(f'{API_BASE}/system_summary', timeout=5)
        if r.status_code == 200:
            return r.json()
        return {}
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch summary: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return {}

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛡️ FedSIG+: Trust-Aware Federated Mapping of Suspicious File Behaviour</h1>
    <p class="main-subtitle">Federated Learning Security Dashboard • Real-Time Monitoring</p>
    <div class="hero-cta">
        <a class="glow-btn" href="#system-summary">View Summary</a>
        <a class="glow-btn red" href="#threats">Inspect Threats</a>
        <a class="glow-btn" href="http://localhost:5000/api/health" target="_blank">API Health</a>
    </div>
    <div class="sticker-ribbon">
        <div class="sticker">🛡️ Shield</div>
        <div class="sticker">🔐 Lock</div>
        <div class="sticker">🕷️ Malware</div>
        <div class="sticker">🧪 Sandbox</div>
        <div class="sticker">🧬 Anomaly</div>
        <div class="sticker">💾 Hash</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar controls (pure UI; does not alter fetch/render logic)
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_interval = st.slider("Refresh interval (seconds)", 1, 10, 5)
    st.markdown("---")
    st.markdown("### ℹ️ Info")
    st.caption("Welcome to FedSIG+: Trust-Aware Federated Mapping of Suspicious File Behaviour, your intelligent cybersecurity companion. This dashboard helps you monitor, detect, and respond to threats in real time — all while keeping your data private and secure through federated learning. Stay informed, stay protected, and let FedSIG+ safeguard your System.")

# Core layout stacked vertically (System Summary → Threats → Analytics)
with st.container():
    st.markdown('<div id="system-summary" class="section"><div class="section-header">System Summary</div>', unsafe_allow_html=True)
    summary = fetch_summary()
    metrics_cards.render(summary)
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div id="threats" class="section"><div class="section-header">Threats</div>', unsafe_allow_html=True)
    df_full = fetch_threats()
    if not df_full.empty:
        csv_bytes = df_full.to_csv(index=False).encode('utf-8')
        st.download_button("Download threats as CSV", data=csv_bytes, file_name="threats.csv", mime="text/csv")
    threat_table.render(df_full)
    st.markdown('</div>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section"><div class="section-header threat-amber">Threat Analytics</div>', unsafe_allow_html=True)
    threats_df = fetch_threats()
    model_charts.render(threats_df, key="charts_vertical")
    st.markdown('</div>', unsafe_allow_html=True)

# Small on-screen note
st.markdown(
    '<div style="position:fixed;bottom:10px;right:10px;color:#FFA200;font-size:12px;" class="hint">'
    f'Auto-refresh every {refresh_interval}s'
    '</div>',
    unsafe_allow_html=True
)

# Floating Chatbot Widget (bottom-right corner)
chatbot_widget.render()

# Non-blocking refresh preserving your original session-state cadence
if 'last_refresh' not in st.session_state:
    st.session_state['last_refresh'] = time.time()
else:
    elapsed = time.time() - st.session_state['last_refresh']
    if auto_refresh and elapsed > float(refresh_interval):
        st.session_state['last_refresh'] = time.time()
        # Streamlit ≥1.27: st.rerun(); older: experimental_rerun
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()
