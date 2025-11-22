"""Floating chatbot widget component for FedShield dashboard."""
import streamlit as st
import requests
import pandas as pd
from typing import Dict

API_BASE = 'http://localhost:5000/api'

def fetch_threats_for_chat():
    """Fetch threats data for chatbot responses with pagination."""
    try:
        params = {'page': 1, 'per_page': 100}  # Get recent 100 threats
        r = requests.get(f'{API_BASE}/threats', params=params, timeout=5)
        if r.status_code == 200:
            data = r.json()
            # Handle both old format (list) and new format (dict with 'data')
            if isinstance(data, dict) and 'data' in data:
                return pd.DataFrame(data['data'])
            elif isinstance(data, list):
                return pd.DataFrame(data)
        return pd.DataFrame()
    except requests.exceptions.RequestException:
        return pd.DataFrame()
    except Exception:
        return pd.DataFrame()

def fetch_summary_for_chat():
    """Fetch summary data for chatbot responses."""
    try:
        r = requests.get(f'{API_BASE}/system_summary', timeout=5)
        if r.status_code == 200:
            return r.json()
        return {}
    except requests.exceptions.RequestException:
        return {}
    except Exception:
        return {}

def fuzzy_match(query: str, patterns: list) -> float:
    """Fuzzy matching for better query understanding."""
    from difflib import SequenceMatcher
    query_lower = query.lower()
    max_ratio = 0.0
    for pattern in patterns:
        ratio = SequenceMatcher(None, query_lower, pattern.lower()).ratio()
        max_ratio = max(max_ratio, ratio)
    return max_ratio

def process_user_query(query: str, df: pd.DataFrame, summary: Dict) -> str:
    """Process user query and return context-aware response with enhanced matching."""
    text = query.lower().strip()
    
    # Enhanced pattern matching with fuzzy matching
    latest_patterns = ["latest", "recent", "new", "show threats", "view threats", "recent threats", "new threats"]
    risk_patterns = ["highest risk", "which client", "most threats", "riskiest", "highest-risk", "worst client", "most dangerous"]
    health_patterns = ["health", "status", "system", "how is", "system status", "overall health"]
    summary_patterns = ["summary", "summarize", "alerts", "overview", "latest alerts", "dashboard summary"]
    help_patterns = ["help", "what can", "commands", "guide", "help guide", "what do you do", "capabilities"]
    
    # Latest threats with enhanced matching
    if any(word in text for word in latest_patterns) or fuzzy_match(text, latest_patterns) > 0.6:
        if not df.empty:
            # Sort by timestamp if available, otherwise use last rows
            if 'timestamp' in df.columns or 'received_at' in df.columns:
                time_col = 'timestamp' if 'timestamp' in df.columns else 'received_at'
                df_sorted = df.sort_values(time_col, ascending=False)
            else:
                df_sorted = df.tail(10)
            
            latest = df_sorted.head(5)
            if 'client_id' in latest.columns and 'file_path' in latest.columns:
                records = latest[['client_id', 'file_path', 'is_threat']].to_dict('records')
                lines = [f"• **{r['client_id']}**: {r.get('file_path', 'N/A')} {'⚠️ THREAT' if r.get('is_threat') else '✓ Safe'}" for r in records]
                return "🛡️ **Latest 5 Events:**\n\n" + "\n".join(lines)
            else:
                return f"🛡️ **Latest Events:**\n\nFound {len(latest)} recent events."
        return "No events recorded yet."
    
    # Highest risk client with enhanced matching
    if any(phrase in text for phrase in risk_patterns) or fuzzy_match(text, risk_patterns) > 0.6:
        if not df.empty and 'client_id' in df.columns:
            # Count threats per client
            if 'is_threat' in df.columns:
                risk = df[df['is_threat'] == True].groupby('client_id').size().sort_values(ascending=False)
            else:
                risk = df.groupby('client_id').size().sort_values(ascending=False)
            
            if not risk.empty:
                top = risk.index[0]
                count = int(risk.iloc[0])
                total_events = len(df[df['client_id'] == top]) if 'client_id' in df.columns else 0
                risk_pct = (count / total_events * 100) if total_events > 0 else 0
                return f"🔴 **Highest Risk Client:** {top}\n\n• Threats Detected: {count}\n• Total Events: {total_events}\n• Risk Rate: {risk_pct:.1f}%"
        return "No threat data available yet."
    
    # System health with enhanced matching
    if any(word in text for word in health_patterns) or fuzzy_match(text, health_patterns) > 0.6:
        threats = int(summary.get('threats', 0))
        clients = int(summary.get('clients', 0))
        isolations = int(summary.get('isolations', 0))
        recent_threats = int(summary.get('recent_threats_24h', 0))
        total_events = int(summary.get('total_events', 0))
        
        if threats < 5:
            status = "🟢 Healthy"
        elif threats < 10:
            status = "🟡 Moderate"
        else:
            status = "🔴 Critical"
        
        return f"**System Status:** {status}\n\n• Active Clients: {clients}\n• Total Threats: {threats}\n• Recent (24h): {recent_threats}\n• Isolations: {isolations}\n• Total Events: {total_events}"
    
    # Summary/Alerts with enhanced matching
    if any(word in text for word in summary_patterns) or fuzzy_match(text, summary_patterns) > 0.6:
        threats = int(summary.get('threats', 0))
        clients = int(summary.get('clients', 0))
        isolations = int(summary.get('isolations', 0))
        recent_threats = int(summary.get('recent_threats_24h', 0))
        return f"📊 **Dashboard Summary:**\n\n• Active Clients: {clients}\n• Total Threats: {threats}\n• Recent Threats (24h): {recent_threats}\n• Isolations: {isolations}"
    
    # Help with enhanced matching
    if any(word in text for word in help_patterns) or fuzzy_match(text, help_patterns) > 0.6:
        return """**Available Commands:**\n\n• "Show latest threats" - View recent events\n• "Which client has highest risk?" - Find riskiest client\n• "System health" - Check overall status\n• "Summarize alerts" - Get overview\n• "Help" - Show this guide\n\n**Tips:**\nYou can use natural language - I'll understand variations of these commands!"""
    
    # Default response with suggestions
    return "I can help with: latest threats, system health, highest risk client, or alerts summary. Try asking:\n\n• 'Show latest threats'\n• 'System health'\n• 'Which client has highest risk?'\n• 'Summarize today's alerts'\n\nType 'help' for more options."

def render():
    """Render floating chatbot widget."""
    # Initialize chat history
    if 'chatbot_messages' not in st.session_state:
        st.session_state.chatbot_messages = [
            {"role": "assistant", "content": "Hi! I'm your FedShield Assistant. How can I help you today?"}
        ]
    
    if 'chatbot_open' not in st.session_state:
        st.session_state.chatbot_open = False
    
    # Floating widget CSS and structure
    st.markdown("""
    <style>
        /* Floating Chatbot Container */
        .floating-chatbot-wrapper {
            position: fixed;
            bottom: 20px;
            right: 20px;
            z-index: 1000;
            font-family: 'Inter', sans-serif;
        }
        
        /* Toggle Button */
        .chatbot-toggle-btn {
            width: 60px;
            height: 60px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00FFFF 0%, #0078FF 100%);
            border: 2px solid rgba(0, 255, 255, 0.5);
            box-shadow: 0 0 20px rgba(0, 255, 255, 0.4), 0 0 40px rgba(0, 255, 255, 0.2);
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 28px;
            transition: all 0.3s ease;
            animation: pulse-glow 2s infinite;
        }
        
        .chatbot-toggle-btn:hover {
            transform: scale(1.1);
            box-shadow: 0 0 30px rgba(0, 255, 255, 0.6), 0 0 60px rgba(0, 255, 255, 0.3);
        }
        
        @keyframes pulse-glow {
            0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 255, 0.4), 0 0 40px rgba(0, 255, 255, 0.2); }
            50% { box-shadow: 0 0 30px rgba(0, 255, 255, 0.6), 0 0 60px rgba(0, 255, 255, 0.4); }
        }
        
        /* Chat Window */
        .chatbot-window-wrapper {
            width: 380px;
            max-height: 600px;
            background: rgba(15, 23, 42, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(0, 255, 255, 0.3);
            border-radius: 16px;
            box-shadow: 0 0 40px rgba(0, 255, 255, 0.2), 0 8px 32px rgba(0, 0, 0, 0.4);
            animation: slide-up 0.3s ease;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        @keyframes slide-up {
            from { transform: translateY(20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
        
        .chatbot-header-custom {
            background: linear-gradient(135deg, rgba(0, 255, 255, 0.15), rgba(0, 120, 255, 0.15));
            border-bottom: 1px solid rgba(0, 255, 255, 0.2);
            padding: 14px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chatbot-header-title {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #00FFFF;
            font-weight: 600;
            font-size: 16px;
        }
        
        .chatbot-close-btn {
            width: 32px;
            height: 32px;
            border: 2px solid #FF3C3C;
            background: linear-gradient(135deg, rgba(255, 60, 60, 0.2), rgba(255, 30, 86, 0.2));
            color: #FF3C3C;
            cursor: pointer;
            border-radius: 8px;
            font-size: 20px;
            font-weight: 700;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.2s ease;
            box-shadow: 0 0 12px rgba(255, 60, 60, 0.4);
            text-shadow: 0 0 8px rgba(255, 60, 60, 0.6);
        }
        
        .chatbot-close-btn:hover {
            background: linear-gradient(135deg, rgba(255, 60, 60, 0.4), rgba(255, 30, 86, 0.4));
            box-shadow: 0 0 20px rgba(255, 60, 60, 0.7), 0 0 30px rgba(255, 60, 60, 0.4);
            transform: scale(1.15);
        }
        
        .chatbot-messages-area {
            flex: 1;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
            max-height: 350px;
        }
        
        .chatbot-quick-actions-area {
            padding: 12px 16px;
            border-top: 1px solid rgba(71, 85, 105, 0.3);
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            background: rgba(2, 6, 23, 0.6);
        }
        
        .chatbot-quick-btn-custom {
            padding: 8px 14px;
            background: linear-gradient(135deg, #00FFFF, #0078FF);
            border: 2px solid #00FFFF;
            border-radius: 10px;
            color: #000000;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 0 15px rgba(0,255,255,0.5), 0 0 30px rgba(0,255,255,0.3), inset 0 0 8px rgba(0,255,255,0.2);
            text-shadow: 0 0 6px rgba(0,0,0,0.7);
        }
        
        .chatbot-quick-btn-custom:hover {
            background: linear-gradient(135deg, #00FFFF, #00BFFF);
            box-shadow: 0 0 25px rgba(0,255,255,0.7), 0 0 50px rgba(0,255,255,0.4), inset 0 0 12px rgba(0,255,255,0.3);
            transform: translateY(-2px) scale(1.05);
        }
        
        /* Style Streamlit chat messages for floating widget - bright vibrant colors */
        .floating-chatbot-wrapper .stChatMessage {
            background: transparent !important;
            padding: 0 !important;
        }
        
        .floating-chatbot-wrapper .stChatMessage[data-testid="user"] .stChatMessageContent {
            background: linear-gradient(135deg, rgba(0, 120, 255, 0.25), rgba(0, 78, 255, 0.25)) !important;
            border: 2px solid rgba(0, 120, 255, 0.5) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            color: #E0F7FF !important;
            text-shadow: 0 0 8px rgba(0, 255, 255, 0.4) !important;
            box-shadow: 0 0 12px rgba(0, 120, 255, 0.3) !important;
        }
        
        .floating-chatbot-wrapper .stChatMessage[data-testid="assistant"] .stChatMessageContent {
            background: linear-gradient(135deg, rgba(232, 17, 35, 0.25), rgba(255, 107, 53, 0.2), rgba(255, 241, 0, 0.15)) !important;
            border: 2px solid rgba(255, 241, 0, 0.5) !important;
            border-radius: 12px !important;
            padding: 10px 14px !important;
            color: #FFF100 !important;
            text-shadow: 0 0 12px rgba(255, 241, 0, 0.8), 0 0 20px rgba(232, 17, 35, 0.6), 0 0 30px rgba(255, 241, 0, 0.4) !important;
            box-shadow: 0 0 20px rgba(255, 241, 0, 0.4), 0 0 30px rgba(232, 17, 35, 0.3), inset 0 0 10px rgba(255, 241, 0, 0.1) !important;
            font-weight: 600 !important;
        }
        
        /* Special styling for greeting message */
        .floating-chatbot-wrapper .stChatMessage[data-testid="assistant"] .stChatMessageContent p,
        .floating-chatbot-wrapper .stChatMessage[data-testid="assistant"] .stChatMessageContent div {
            background: linear-gradient(135deg, #E81123, #FF6B35, #FFF100) !important;
            -webkit-background-clip: text !important;
            -webkit-text-fill-color: transparent !important;
            background-clip: text !important;
            text-shadow: 0 0 15px rgba(255, 241, 0, 0.9), 0 0 25px rgba(232, 17, 35, 0.7), 0 0 35px rgba(255, 241, 0, 0.5) !important;
            filter: drop-shadow(0 0 10px rgba(255, 241, 0, 0.8)) drop-shadow(0 0 15px rgba(232, 17, 35, 0.6)) !important;
            font-weight: 700 !important;
            font-size: 15px !important;
            letter-spacing: 0.3px !important;
        }
        
        .floating-chatbot-wrapper .stChatInput {
            background: rgba(15, 23, 42, 0.9) !important;
            border: 2px solid rgba(0, 255, 255, 0.4) !important;
            border-radius: 12px !important;
            color: #E0F7FF !important;
            font-weight: 500 !important;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.2) !important;
        }
        
        .floating-chatbot-wrapper .stChatInput:focus {
            border-color: rgba(0, 255, 255, 0.7) !important;
            box-shadow: 0 0 25px rgba(0, 255, 255, 0.5), 0 0 40px rgba(0, 255, 255, 0.3) !important;
            background: rgba(15, 23, 42, 0.95) !important;
        }
        
        .floating-chatbot-wrapper .stChatInput::placeholder {
            color: #94a3b8 !important;
            text-shadow: 0 0 6px rgba(0, 255, 255, 0.3) !important;
        }
        
        /* Style Streamlit buttons in chatbot (quick action buttons) */
        .floating-chatbot-wrapper button[kind="secondary"],
        .floating-chatbot-wrapper button[data-testid="baseButton-secondary"] {
            background: linear-gradient(135deg, #00FFFF, #0078FF) !important;
            color: #000000 !important;
            border: 2px solid #00FFFF !important;
            border-radius: 10px !important;
            padding: 8px 14px !important;
            font-weight: 700 !important;
            font-size: 12px !important;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            box-shadow: 0 0 15px rgba(0,255,255,0.5), 0 0 30px rgba(0,255,255,0.3), inset 0 0 8px rgba(0,255,255,0.2) !important;
            text-shadow: 0 0 6px rgba(0,0,0,0.7) !important;
            transition: all 0.3s ease !important;
        }
        
        .floating-chatbot-wrapper button[kind="secondary"]:hover,
        .floating-chatbot-wrapper button[data-testid="baseButton-secondary"]:hover {
            background: linear-gradient(135deg, #00FFFF, #00BFFF) !important;
            box-shadow: 0 0 25px rgba(0,255,255,0.7), 0 0 50px rgba(0,255,255,0.4), inset 0 0 12px rgba(0,255,255,0.3) !important;
            transform: translateY(-2px) scale(1.05) !important;
        }
        
        /* Close button styling */
        .floating-chatbot-wrapper button[aria-label*="Close"],
        .floating-chatbot-wrapper button[title*="Close"],
        .floating-chatbot-wrapper button[key*="chatbot_close"] {
            background: linear-gradient(135deg, rgba(255, 60, 60, 0.2), rgba(255, 30, 86, 0.2)) !important;
            border: 2px solid #FF3C3C !important;
            color: #FF3C3C !important;
            box-shadow: 0 0 12px rgba(255, 60, 60, 0.4) !important;
            text-shadow: 0 0 8px rgba(255, 60, 60, 0.6) !important;
            font-weight: 700 !important;
        }
        
        .floating-chatbot-wrapper button[aria-label*="Close"]:hover,
        .floating-chatbot-wrapper button[title*="Close"]:hover,
        .floating-chatbot-wrapper button[key*="chatbot_close"]:hover {
            background: linear-gradient(135deg, rgba(255, 60, 60, 0.4), rgba(255, 30, 86, 0.4)) !important;
            box-shadow: 0 0 20px rgba(255, 60, 60, 0.7), 0 0 30px rgba(255, 60, 60, 0.4) !important;
            transform: scale(1.15) !important;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Render toggle button or chat window
    if not st.session_state.chatbot_open:
        # Toggle button
        if st.button("🤖", key="chatbot_toggle_btn", help="Open FedShield Assistant"):
            st.session_state.chatbot_open = True
            st.rerun()
        # Position the button using CSS
        st.markdown("""
        <script>
            (function() {
                const btn = document.querySelector('[data-testid="baseButton-secondary"][aria-label*="Open FedShield"]');
                if (btn) {
                    btn.style.position = 'fixed';
                    btn.style.bottom = '20px';
                    btn.style.right = '20px';
                    btn.style.width = '60px';
                    btn.style.height = '60px';
                    btn.style.borderRadius = '50%';
                    btn.style.background = 'linear-gradient(135deg, #00FFFF 0%, #0078FF 100%)';
                    btn.style.border = '2px solid rgba(0, 255, 255, 0.5)';
                    btn.style.boxShadow = '0 0 20px rgba(0, 255, 255, 0.4), 0 0 40px rgba(0, 255, 255, 0.2)';
                    btn.style.zIndex = '1000';
                    btn.style.fontSize = '28px';
                    btn.style.padding = '0';
                }
            })();
        </script>
        """, unsafe_allow_html=True)
    else:
        # Chat window
        with st.container():
            st.markdown('<div class="floating-chatbot-wrapper">', unsafe_allow_html=True)
            st.markdown('<div class="chatbot-window-wrapper">', unsafe_allow_html=True)
            
            # Header
            st.markdown("""
            <div class="chatbot-header-custom">
                <div class="chatbot-header-title">
                    <span>🛡️</span>
                    <span>FedShield Assistant</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns([1, 20])
            with col2:
                if st.button("×", key="chatbot_close_btn", help="Close"):
                    st.session_state.chatbot_open = False
                    st.rerun()
            
            # Quick action buttons
            st.markdown('<div class="chatbot-quick-actions-area">', unsafe_allow_html=True)
            col1, col2, col3, col4 = st.columns(4)
            
            quick_actions = [
                ("View Threats", "Show latest threats"),
                ("System Health", "System health"),
                ("Latest Alerts", "Summarize alerts"),
                ("Help Guide", "Help")
            ]
            
            for i, (label, query) in enumerate(quick_actions):
                with [col1, col2, col3, col4][i]:
                    if st.button(label, key=f"quick_{i}", use_container_width=True):
                        df = fetch_threats_for_chat()
                        summary = fetch_summary_for_chat()
                        response = process_user_query(query, df, summary)
                        st.session_state.chatbot_messages.append({"role": "user", "content": query})
                        st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                        st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat messages
            st.markdown('<div class="chatbot-messages-area">', unsafe_allow_html=True)
            for msg in st.session_state.chatbot_messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Chat input
            user_input = st.chat_input("Ask about threats, clients, or alerts...", key="chatbot_input_main")
            if user_input:
                df = fetch_threats_for_chat()
                summary = fetch_summary_for_chat()
                response = process_user_query(user_input, df, summary)
                st.session_state.chatbot_messages.append({"role": "user", "content": user_input})
                st.session_state.chatbot_messages.append({"role": "assistant", "content": response})
                st.rerun()
            
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Position the window using CSS
            st.markdown("""
            <script>
                (function() {
                    const wrapper = document.querySelector('.floating-chatbot-wrapper');
                    if (wrapper) {
                        wrapper.style.position = 'fixed';
                        wrapper.style.bottom = '20px';
                        wrapper.style.right = '20px';
                        wrapper.style.zIndex = '1000';
                    }
                })();
            </script>
            """, unsafe_allow_html=True)
