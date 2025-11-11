import json
import os
from datetime import datetime
from flask import Flask, request, jsonify

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, 'global_logs.json')
MODEL_INFO_FILE = os.path.join(BASE_DIR, 'models', 'global_model_info.json')

os.makedirs(os.path.join(BASE_DIR, 'models'), exist_ok=True)

app = Flask(__name__)


def _load_logs():
    if not os.path.exists(DATA_FILE):
        return []
    with open(DATA_FILE, 'r') as f:
        try:
            return json.load(f)
        except Exception:
            return []


def _save_logs(logs):
    with open(DATA_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


@app.route('/api/report_threat', methods=['POST'])
def report_threat():
    payload = request.get_json(force=True)
    logs = _load_logs()
    payload['received_at'] = datetime.utcnow().isoformat() + 'Z'
    logs.append(payload)
    _save_logs(logs)
    return jsonify({'status': 'ok', 'saved': True}), 201


@app.route('/api/threats', methods=['GET'])
def get_threats():
    return jsonify(_load_logs())


@app.route('/api/system_summary', methods=['GET'])
def system_summary():
    logs = _load_logs()
    clients = len(set(l.get('client_id') for l in logs))
    threats = len([l for l in logs if l.get('is_threat')])
    isolations = len([l for l in logs if l.get('action') == 'quarantine'])
    return jsonify({'clients': clients, 'threats': threats, 'isolations': isolations})


@app.route('/api/global_model_info', methods=['GET'])
def global_model_info():
    if not os.path.exists(MODEL_INFO_FILE):
        return jsonify({'status': 'no-model'})
    with open(MODEL_INFO_FILE, 'r') as f:
        return jsonify(json.load(f))


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok', 'timestamp': datetime.utcnow().isoformat()+'Z'})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)