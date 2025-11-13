# FedShield Data Schema & Objectives

## Threat Classification Labels
```
Label Categories (Multi-class):
- 0: NORMAL - Normal network/system activity
- 1: MALWARE - Malicious software/ransomware
- 2: PHISHING - Phishing attempts/social engineering
- 3: UNAUTHORIZED_ACCESS - Unauthorized access/intrusion attempts
- 4: DATA_LEAK - Data exfiltration/data leak attempts
- 5: ANOMALY - General anomalies not fitting other categories
```

## Feature Schema

### Network Features
- `src_ip` (string): Source IP address (hashed for privacy)
- `dst_ip` (string): Destination IP address (hashed)
- `src_port` (int): Source port
- `dst_port` (int): Destination port
- `protocol` (int): Protocol (TCP=6, UDP=17, ICMP=1, etc.)
- `packet_count` (int): Number of packets in flow
- `byte_count` (int): Total bytes transferred
- `duration_sec` (float): Flow duration in seconds
- `flow_rate` (float): Bytes per second

### System Features
- `cpu_usage_percent` (float): CPU usage 0-100
- `memory_usage_percent` (float): Memory usage 0-100
- `disk_io_read_mb` (float): Disk read MB/s
- `disk_io_write_mb` (float): Disk write MB/s
- `network_in_mbps` (float): Network input Mbps
- `network_out_mbps` (float): Network output Mbps
- `process_count` (int): Number of running processes
- `open_connections` (int): Number of open network connections
- `failed_login_attempts` (int): Failed SSH/RDP login attempts

### Application Features
- `http_requests` (int): HTTP request count in window
- `https_requests` (int): HTTPS request count
- `dns_queries` (int): DNS query count
- `unique_domains` (int): Number of unique domains contacted
- `suspicious_ports_contacted` (int): Count of unusual ports
- `encryption_weak_tls` (int): Weak TLS version usage count

### Derived Features (computed locally)
- `packet_size_avg` (float): Average packet size
- `packet_size_std` (float): Packet size std dev
- `inter_arrival_time_avg` (float): Avg time between packets
- `entropy_src_port` (float): Entropy of source ports
- `entropy_dst_port` (float): Entropy of destination ports

## Data Format

### Input CSV/JSON per Client
```
timestamp,src_ip_hash,dst_ip_hash,src_port,dst_port,protocol,packet_count,byte_count,
duration_sec,flow_rate,cpu_usage_percent,memory_usage_percent,disk_io_read_mb,
disk_io_write_mb,network_in_mbps,network_out_mbps,process_count,open_connections,
failed_login_attempts,http_requests,https_requests,dns_queries,unique_domains,
suspicious_ports_contacted,encryption_weak_tls,label

2024-01-01T10:00:00,hash_192_168_1_1,hash_10_0_0_1,54321,443,6,100,50000,30.5,1637.2,
25.3,45.2,10.5,15.3,100.2,85.5,42,12,0,5,3,2,3,0,0,0
```

### Tensor Format
- Shape: (batch_size, 26) for 26 features
- Dtype: float32
- Normalization: Per-client z-score normalization

## Privacy & Security Considerations
- IP addresses and sensitive data hashed using SHA256
- Features normalized per-client before transmission
- Client IDs anonymized (Client_A, Client_B, etc.)
- Model updates encrypted in transit with TLS 1.3
- Per-client differential privacy budget tracked

## Non-IID Scenarios
1. **IID Baseline**: Balanced distribution across all clients (20% each class)
2. **Skewed Distribution**: Client 1 has 80% normal traffic, Client 2 has 60% malware
3. **Feature Scarcity**: Some clients missing certain feature types (e.g., no DNS data)
4. **Temporal Drift**: Distribution changes over training rounds
