import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'data' / 'sample_logs.csv'
OUT = ROOT / 'data' / 'sample_logs_adjusted.csv'
BACKUP = ROOT / 'data' / 'sample_logs.csv.bak'

def is_threat_row(row):
    fp = row.get('file_path','').lower()
    try:
        cpu = float(row.get('cpu_pct') or 0)
    except Exception:
        cpu = 0.0
    try:
        access = int(row.get('file_access_count') or 0)
    except Exception:
        access = 0
    if 'malicious' in fp or cpu > 80 or access > 10:
        return True
    return False

def main():
    if not SRC.exists():
        print(f"Source CSV not found: {SRC}")
        return
    rows = []
    with SRC.open('r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    # compute non-threat averages
    non_cpu = []
    non_access = []
    non_net = []
    for r in rows:
        if not is_threat_row(r):
            try:
                non_cpu.append(float(r.get('cpu_pct') or 0))
            except Exception:
                pass
            try:
                non_access.append(int(r.get('file_access_count') or 0))
            except Exception:
                pass
            try:
                non_net.append(int(r.get('net_bytes') or 0))
            except Exception:
                pass

    def avg(l):
        return sum(l)/len(l) if l else 0

    avg_cpu = avg(non_cpu)
    avg_access = avg(non_access)
    avg_net = avg(non_net)

    print(f"Non-threat averages: cpu={avg_cpu:.2f}, access={avg_access:.2f}, net={avg_net:.2f}")

    # create backup
    if not BACKUP.exists():
        SRC.replace(BACKUP)
        # copy backup back to source so source still exists; we'll read from backup
        BACKUP.replace(SRC)

    # adjust threat rows
    out_rows = []
    for r in rows:
        if is_threat_row(r):
            # amplify threat metrics
            try:
                cpu = float(r.get('cpu_pct') or 0)
            except Exception:
                cpu = 0.0
            try:
                access = int(r.get('file_access_count') or 0)
            except Exception:
                access = 0
            try:
                net = int(r.get('net_bytes') or 0)
            except Exception:
                net = 0

            new_cpu = max(cpu, avg_cpu * 5, 95.0)
            new_access = max(access, int(max(1, avg_access) * 10), 100)
            new_net = max(net, int(max(1, avg_net) * 500), net * 2 or 100000)

            r['cpu_pct'] = f"{new_cpu:.1f}"
            r['file_access_count'] = str(new_access)
            r['net_bytes'] = str(new_net)
        out_rows.append(r)

    # write adjusted CSV
    fieldnames = out_rows[0].keys() if out_rows else ['timestamp','cpu_pct','net_bytes','file_access_count','file_path']
    with OUT.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in out_rows:
            writer.writerow(r)

    print(f"Wrote adjusted CSV to: {OUT}")

if __name__ == '__main__':
    main()
