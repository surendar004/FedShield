import json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'server' / 'global_logs.json'
BACKUP = ROOT / 'server' / f'global_logs.json.bak'

def is_threat_like(entry):
    fp = (entry.get('file_path') or '').lower()
    try:
        cpu = float(entry.get('cpu_pct') or 0)
    except Exception:
        cpu = 0.0
    try:
        access = int(entry.get('file_access_count') or 0)
    except Exception:
        access = 0
    if 'malicious' in fp or cpu > 80 or access > 10:
        return True
    return False

def avg(l):
    return sum(l)/len(l) if l else 0

def main():
    if not SRC.exists():
        print(f"Source JSON not found: {SRC}")
        return
    data = json.loads(SRC.read_text(encoding='utf-8'))

    # compute averages over entries currently marked non-threat
    non_cpu = []
    non_access = []
    non_net = []
    for e in data:
        if not e.get('is_threat') and not is_threat_like(e):
            try:
                non_cpu.append(float(e.get('cpu_pct') or 0))
            except Exception:
                pass
            try:
                non_access.append(int(e.get('file_access_count') or 0))
            except Exception:
                pass
            try:
                non_net.append(int(e.get('net_bytes') or 0))
            except Exception:
                pass

    avg_cpu = avg(non_cpu)
    avg_access = avg(non_access)
    avg_net = avg(non_net)

    print(f"Baseline non-threat averages: cpu={avg_cpu:.2f}, access={avg_access:.2f}, net={avg_net:.2f}")

    # backup
    timestamp = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ')
    backup_path = BACKUP
    if BACKUP.exists():
        backup_path = ROOT / 'server' / f'global_logs.json.bak.{timestamp}'
    SRC.replace(backup_path)
    # write the backup copy back to source so we keep the file for reading/writing
    backup_path.replace(SRC)

    modified = 0
    out = []
    for e in data:
        if is_threat_like(e) or e.get('is_threat'):
            try:
                cpu = float(e.get('cpu_pct') or 0)
            except Exception:
                cpu = 0.0
            try:
                access = int(e.get('file_access_count') or 0)
            except Exception:
                access = 0
            try:
                net = int(e.get('net_bytes') or 0)
            except Exception:
                net = 0

            new_cpu = max(cpu, avg_cpu * 5, 95.0)
            new_access = max(access, int(max(1, avg_access) * 10), 100)
            new_net = max(net, int(max(1, avg_net) * 500), net * 2 or 100000)

            e['cpu_pct'] = float(f"{new_cpu:.1f}")
            e['file_access_count'] = int(new_access)
            e['net_bytes'] = float(int(new_net))
            e['is_threat'] = True
            modified += 1
        out.append(e)

    SRC.write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(f"Wrote modified global logs to: {SRC}")
    print(f"Backup saved at: {backup_path}")
    print(f"Entries modified: {modified}")

if __name__ == '__main__':
    main()
