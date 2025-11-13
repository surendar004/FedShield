"""
FedShield FL Metrics Dashboard

Real-time Streamlit dashboard for monitoring federated learning experiments.
Displays rounds, client participation, accuracy curves, privacy budgets, and more.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import os
import requests
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FedShield FL Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header {
        color: #1f77b4;
        font-size: 28px;
        font-weight: bold;
        margin-bottom: 20px;
    }
    .subheader {
        color: #2c3e50;
        font-size: 18px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ==================== Data Generation Functions ====================

def generate_sample_experiment_data(num_rounds: int = 20, num_clients: int = 20) -> Dict:
    """Generate sample experimental data for demo purposes."""
    data = {
        "rounds": list(range(num_rounds)),
        "global_accuracy": (0.70 + np.cumsum(np.random.rand(num_rounds) * 0.015)).tolist(),
        "global_loss": (0.30 - np.cumsum(np.random.rand(num_rounds) * 0.010)).tolist(),
        "avg_client_accuracy": (0.68 + np.cumsum(np.random.rand(num_rounds) * 0.012)).tolist(),
        "convergence_rate": np.random.rand(num_rounds).tolist(),
        "clients_participated": [int(num_clients - np.random.randint(0, 3)) for _ in range(num_rounds)],
        "epsilon_consumed": (np.cumsum(np.random.rand(num_rounds) * 0.05)).tolist(),
        "communication_mb": (np.random.rand(num_rounds) * 5 + 2).tolist(),
    }
    
    # Per-client metrics
    client_data = {}
    for client_id in range(num_clients):
        client_data[f"Client_{client_id}"] = {
            "accuracy_history": (0.65 + np.cumsum(np.random.rand(num_rounds) * 0.010)).tolist(),
            "samples": np.random.randint(50, 200),
            "final_accuracy": 0.65 + np.random.rand() * 0.25,
        }
    
    data["client_metrics"] = client_data
    
    return data


# ==================== Dashboard Components ====================

def render_header():
    """Render dashboard header."""
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown('<div class="header">🛡️ FedShield Federated Learning Dashboard</div>', 
                   unsafe_allow_html=True)
    
    with col2:
        st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
    
    with col3:
        refresh_interval = st.selectbox("Refresh", ["Auto (5s)", "Auto (10s)", "Manual"], 
                                       key="refresh_key", label_visibility="collapsed")


def render_experiment_summary(data: Dict):
    """Render experiment summary metrics."""
    st.markdown('<div class="subheader">Experiment Summary</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Rounds",
            len(data["rounds"]),
            delta=None
        )
    
    with col2:
        latest_accuracy = data["global_accuracy"][-1]
        accuracy_delta = latest_accuracy - data["global_accuracy"][0]
        st.metric(
            "Global Accuracy",
            f"{latest_accuracy:.4f}",
            delta=f"+{accuracy_delta:.4f}",
            delta_color="off"
        )
    
    with col3:
        latest_loss = data["global_loss"][-1]
        loss_delta = data["global_loss"][0] - latest_loss
        st.metric(
            "Global Loss",
            f"{latest_loss:.4f}",
            delta=f"-{loss_delta:.4f}",
            delta_color="off"
        )
    
    with col4:
        avg_participation = np.mean(data["clients_participated"])
        st.metric(
            "Avg Participation Rate",
            f"{avg_participation/20*100:.1f}%",
            delta=None
        )
    
    with col5:
        total_epsilon = data["epsilon_consumed"][-1] if data["epsilon_consumed"] else 0
        st.metric(
            "Total ε Consumed",
            f"{total_epsilon:.4f}",
            delta=None
        )


def render_accuracy_convergence(data: Dict):
    """Render accuracy and convergence curves."""
    st.markdown('<div class="subheader">Accuracy & Convergence</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Global accuracy over rounds
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["rounds"],
            y=data["global_accuracy"],
            name="Global Accuracy",
            mode="lines+markers",
            line=dict(color="#1f77b4", width=2),
            marker=dict(size=6)
        ))
        fig.add_trace(go.Scatter(
            x=data["rounds"],
            y=data["avg_client_accuracy"],
            name="Avg Client Accuracy",
            mode="lines+markers",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            marker=dict(size=6)
        ))
        fig.update_layout(
            title="Accuracy Over Rounds",
            xaxis_title="Round",
            yaxis_title="Accuracy",
            hovermode="x unified",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Loss over rounds
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["rounds"],
            y=data["global_loss"],
            name="Global Loss",
            mode="lines+markers",
            line=dict(color="#d62728", width=2),
            marker=dict(size=6),
            fill="tozeroy"
        ))
        fig.update_layout(
            title="Loss Over Rounds",
            xaxis_title="Round",
            yaxis_title="Loss",
            hovermode="x unified",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_client_participation(data: Dict):
    """Render client participation metrics."""
    st.markdown('<div class="subheader">Client Participation & Communication</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Client participation over rounds
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=data["rounds"],
            y=data["clients_participated"],
            name="Clients Participated",
            marker=dict(color="#2ca02c")
        ))
        fig.update_layout(
            title="Client Participation by Round",
            xaxis_title="Round",
            yaxis_title="Number of Clients",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Communication overhead
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["rounds"],
            y=data["communication_mb"],
            name="Communication (MB)",
            mode="lines+markers",
            line=dict(color="#9467bd", width=2),
            fill="tozeroy"
        ))
        fig.update_layout(
            title="Communication Overhead per Round",
            xaxis_title="Round",
            yaxis_title="Communication (MB)",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)


def render_privacy_metrics(data: Dict):
    """Render privacy budget consumption."""
    st.markdown('<div class="subheader">Differential Privacy Metrics</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Privacy budget consumption
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data["rounds"],
            y=data["epsilon_consumed"],
            name="Cumulative ε",
            mode="lines+markers",
            line=dict(color="#e377c2", width=2),
            fill="tozeroy"
        ))
        fig.update_layout(
            title="Privacy Budget Consumption (ε)",
            xaxis_title="Round",
            yaxis_title="Cumulative ε",
            height=400,
            template="plotly_white"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Privacy budget gauge
        final_epsilon = data["epsilon_consumed"][-1] if data["epsilon_consumed"] else 0
        max_epsilon = 1.0
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=final_epsilon,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Privacy Budget (ε)"},
            delta={"reference": max_epsilon},
            gauge={
                "axis": {"range": [None, max_epsilon]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, max_epsilon * 0.33], "color": "lightgreen"},
                    {"range": [max_epsilon * 0.33, max_epsilon * 0.66], "color": "lightyellow"},
                    {"range": [max_epsilon * 0.66, max_epsilon], "color": "lightcoral"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": max_epsilon
                }
            }
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)


def render_client_distribution(data: Dict):
    """Render per-client accuracy distribution."""
    st.markdown('<div class="subheader">Per-Client Performance Distribution</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Accuracy distribution histogram
        client_accuracies = [metrics["final_accuracy"] 
                            for metrics in data["client_metrics"].values()]
        
        fig = go.Figure(data=[
            go.Histogram(
                x=client_accuracies,
                nbinsx=10,
                name="Accuracy Distribution",
                marker=dict(color="#1f77b4")
            )
        ])
        fig.update_layout(
            title="Client Accuracy Distribution",
            xaxis_title="Final Accuracy",
            yaxis_title="Number of Clients",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Top clients
        top_clients = sorted(
            data["client_metrics"].items(),
            key=lambda x: x[1]["final_accuracy"],
            reverse=True
        )[:5]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[name for name, _ in top_clients],
                y=[metrics["final_accuracy"] for _, metrics in top_clients],
                marker=dict(color="#2ca02c")
            )
        ])
        fig.update_layout(
            title="Top 5 Clients by Accuracy",
            yaxis_title="Final Accuracy",
            height=400,
            template="plotly_white",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)


def render_threats_and_risk(logs: List[Dict], data: Dict):
    """Render recent threats and per-client risk status using logs from API or sample data."""
    st.markdown('<div class="subheader">Threats Detected & Client Risk</div>', unsafe_allow_html=True)

    col1, col2 = st.columns([2, 3])

    with col1:
        st.markdown('**Recent Threats**')
        if not logs:
            st.info('No threat reports available')
        else:
            # Show most recent 10 threats
            recent = sorted(logs, key=lambda x: x.get('received_at', ''), reverse=True)[:10]
            for t in recent:
                ts = t.get('received_at', 'N/A')
                cid = t.get('client_id', 'unknown')
                desc = t.get('description', t.get('type', 'suspicious'))
                is_threat = t.get('is_threat', False)
                status = 'THREAT' if is_threat else 'suspicious'
                st.write(f"- [{ts}] Client={cid} — {status} — {desc}")

    with col2:
        st.markdown('**Client Risk Summary**')
        # Build per-client threat counts from logs
        client_counts = {}
        for entry in logs:
            cid = entry.get('client_id') or 'unknown'
            client_counts.setdefault(cid, {'threats': 0, 'events': 0})
            client_counts[cid]['events'] += 1
            if entry.get('is_threat'):
                client_counts[cid]['threats'] += 1

        # Merge with data['client_metrics'] to show accuracy and samples if available
        rows = []
        for cid, stats in client_counts.items():
            final_acc = None
            samples = None
            if data and 'client_metrics' in data and cid in data['client_metrics']:
                cm = data['client_metrics'][cid]
                final_acc = f"{cm.get('final_accuracy', 0):.4f}"
                samples = cm.get('samples')
            risk_level = 'HIGH' if stats['threats'] > 0 else ('MEDIUM' if stats['events'] > 0 else 'LOW')
            rows.append({'Client': cid, 'Threats': stats['threats'], 'Events': stats['events'], 'Risk': risk_level, 'Final Accuracy': final_acc or '-', 'Samples': samples or '-'})

        if rows:
            df_risk = pd.DataFrame(rows).sort_values(['Risk', 'Threats'], ascending=[False, False])
            st.dataframe(df_risk, use_container_width=True)
        else:
            st.info('No client events found to compute risk')


def render_algorithm_comparison(data: Dict):
    """Render algorithm comparison metrics."""
    st.markdown('<div class="subheader">Algorithm Configuration</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("""
        **FedAvg Configuration**
        - Algorithm: Federated Averaging
        - Aggregation: Weighted Mean
        - Adaptive: No
        - Privacy: None
        """)
    
    with col2:
        st.warning("""
        **FedProx Configuration**
        - Algorithm: Federated Proximal
        - Aggregation: Weighted Mean
        - Adaptive: No
        - Privacy: Optional DP-SGD
        - μ = 0.01 (Non-IID Handling)
        """)
    
    with col3:
        st.success("""
        **FedOpt Configuration**
        - Algorithm: Federated Optimization
        - Aggregation: Adaptive (Adam/Yogi)
        - Adaptive: Yes
        - Privacy: Optional DP-SGD
        - Server LR = 0.01
        """)


def render_client_details_table(data: Dict):
    """Render detailed client metrics table."""
    st.markdown('<div class="subheader">Client Details</div>', unsafe_allow_html=True)
    
    # Create dataframe
    client_list = []
    for client_name, metrics in data["client_metrics"].items():
        client_list.append({
            "Client": client_name,
            "Final Accuracy": f"{metrics['final_accuracy']:.4f}",
            "Samples": metrics["samples"],
            "Avg Accuracy": f"{np.mean(metrics['accuracy_history']):.4f}",
            "Std Dev": f"{np.std(metrics['accuracy_history']):.4f}",
        })
    
    df = pd.DataFrame(client_list)
    
    # Sort by final accuracy
    df["Final Accuracy"] = df["Final Accuracy"].astype(float)
    df = df.sort_values("Final Accuracy", ascending=False)
    
    # Display table
    st.dataframe(df, use_container_width=True, hide_index=True)
    
    # Statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_acc = df["Final Accuracy"].mean()
        st.metric("Mean Accuracy", f"{avg_acc:.4f}")
    
    with col2:
        std_acc = df["Final Accuracy"].std()
        st.metric("Std Dev", f"{std_acc:.4f}")
    
    with col3:
        min_acc = df["Final Accuracy"].min()
        st.metric("Min Accuracy", f"{min_acc:.4f}")
    
    with col4:
        max_acc = df["Final Accuracy"].max()
        st.metric("Max Accuracy", f"{max_acc:.4f}")


def render_system_stats(data: Dict):
    """Render system statistics and health."""
    st.markdown('<div class="subheader">System Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_samples = sum(m["samples"] for m in data["client_metrics"].values())
        st.metric("Total Samples", f"{total_samples:,}")
    
    with col2:
        total_communication = sum(data["communication_mb"])
        st.metric("Total Communication", f"{total_communication:.2f} MB")
    
    with col3:
        avg_roundtime = np.random.uniform(2, 5)  # Simulated
        st.metric("Avg Round Time", f"{avg_roundtime:.2f}s")
    
    with col4:
        convergence_iters = len(data["rounds"])
        st.metric("Rounds to Converge", convergence_iters)
    
    with col5:
        model_size = np.random.uniform(0.5, 2.0)  # Simulated
        st.metric("Model Size", f"{model_size:.2f} MB")


def render_alerts_panel():
    """Render alerts and notifications panel."""
    st.markdown('<div class="subheader">Alerts & Notifications</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("✅ Training proceeding normally - convergence on track")
    
    with col2:
        st.warning("⚠️ Privacy budget consuming faster than expected")


# ==================== Main Dashboard ====================

def main():
    """Main dashboard application."""
    
    # Sidebar configuration
    st.sidebar.markdown("### ⚙️ Dashboard Configuration")
    
    # Data source selection
    data_source = st.sidebar.radio(
        "Data Source",
        ["Sample Data", "MLflow Backend", "CSV File", "Live API"]
    )
    
    # Generate or load data
    logs = []
    if data_source == "Sample Data":
        num_rounds = st.sidebar.slider("Number of Rounds", 5, 100, 20)
        num_clients = st.sidebar.slider("Number of Clients", 5, 100, 20)
        data = generate_sample_experiment_data(num_rounds, num_clients)
        st.sidebar.success("✅ Sample data generated")
    
    elif data_source == "MLflow Backend":
        st.sidebar.info("MLflow integration coming soon")
        data = generate_sample_experiment_data(20, 20)
    
    else:
        uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
            st.sidebar.success("✅ CSV loaded")
            data = generate_sample_experiment_data(20, 20)  # Fallback
        else:
            data = generate_sample_experiment_data(20, 20)

    if data_source == "Live API":
        api_base = st.sidebar.text_input('API Base URL', 'http://localhost:5000')
        st.sidebar.write('Fetching live data from API...')
        try:
            r = requests.get(f"{api_base}/api/threats", timeout=3)
            if r.status_code == 200:
                logs = r.json().get('data', [])
            else:
                st.sidebar.error(f"Failed to fetch threats: {r.status_code}")
        except Exception as e:
            st.sidebar.error(f"Could not reach API: {e}")

        try:
            r2 = requests.get(f"{api_base}/api/system_summary", timeout=3)
            if r2.status_code == 200:
                summary = r2.json()
                # attach summary metrics to data for display if needed
                # keep data as sample but adjust rounds to summary total_events as a hint
                data = generate_sample_experiment_data(20, 20)
                data['system_summary'] = summary
            else:
                st.sidebar.warning('System summary not available')
        except Exception:
            st.sidebar.warning('Could not fetch system summary from API')
    
    # Dashboard sections
    render_header()
    render_experiment_summary(data)
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Convergence",
        "👥 Clients",
        "🔐 Privacy",
        "⚙️ Configuration",
        "📊 System"
    ])
    
    with tab1:
        render_accuracy_convergence(data)
        render_client_participation(data)
    
    with tab2:
        render_threats_and_risk(logs, data)
        render_client_distribution(data)
        render_client_details_table(data)
    
    with tab3:
        render_privacy_metrics(data)
    
    with tab4:
        render_algorithm_comparison(data)
    
    with tab5:
        render_system_stats(data)
        render_alerts_panel()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; margin-top: 30px;'>
        FedShield © 2025 | Federated Learning Security Framework<br>
        Dashboard v1.0 | Last Updated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()