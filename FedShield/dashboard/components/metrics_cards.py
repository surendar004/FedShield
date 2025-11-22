import streamlit as st

def render(summary: dict):
    import json
    from pathlib import Path
    # Load actual accuracy from global_model_info.json
    info_path = Path(__file__).parent.parent.parent / 'server' / 'models' / 'global_model_info.json'
    try:
        with open(info_path, 'r', encoding='utf8') as f:
            info = json.load(f)
            acc = info.get('simulated_accuracy', None)
            if acc is not None:
                acc_str = f"{acc*100:.2f}%"
            else:
                acc_str = 'N/A'
    except Exception:
        acc_str = 'N/A'
    a, b, c, d = st.columns(4)
    a.metric('Trusted Clients', summary.get('clients', 0))
    b.metric('Signatures Logged', summary.get('threats', 0))
    c.metric('Quarantines', summary.get('isolations', 0))
    d.metric('Detection Accuracy (FedSIG+)', acc_str)
