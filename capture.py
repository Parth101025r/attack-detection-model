#!/usr/bin/env python3
"""
Network Packet Capture and Flow Feature Extraction
Extracts UNSW-NB15 dataset compatible features from network traffic
"""

import subprocess
import sys
import os
import time
from datetime import datetime
from collections import defaultdict
import csv

try:
    from scapy.all import *
except ImportError:
    print("Error: Scapy not installed. Installing...")
    subprocess.run([sys.executable, "-m", "pip", "install", "scapy"], check=True)
    from scapy.all import *

class FlowExtractor:
    def __init__(self):
        self.flows = defaultdict(lambda: {
            'spkts': 0, 'dpkts': 0, 'sbytes': 0, 'dbytes': 0,
            'start_time': None, 'end_time': None,
            'src_ttl': [], 'dst_ttl': [],
            'src_pkt_times': [], 'dst_pkt_times': [],
            'src_win': [], 'dst_win': [],
            'tcp_flags': [], 'service': 'other'
        })
    
    def get_flow_key(self, pkt):
        """Generate unique flow identifier"""
        if IP in pkt:
            if TCP in pkt:
                proto = 'tcp'
                sport = pkt[TCP].sport
                dport = pkt[TCP].dport
            elif UDP in pkt:
                proto = 'udp'
                sport = pkt[UDP].sport
                dport = pkt[UDP].dport
            else:
                proto = 'other'
                sport = 0
                dport = 0
            
            src = pkt[IP].src
            dst = pkt[IP].dst
            
            # Create bidirectional flow key (sorted to group both directions)
            flow_tuple = tuple(sorted([(src, sport), (dst, dport)]))
            return (flow_tuple, proto)
        return None
    
    def detect_service(self, pkt):
        """Detect service based on port numbers"""
        if TCP in pkt:
            port = pkt[TCP].dport
        elif UDP in pkt:
            port = pkt[UDP].dport
        else:
            return 'other'
        
        services = {
            20: 'ftp-data', 21: 'ftp', 22: 'ssh', 23: 'telnet',
            25: 'smtp', 53: 'dns', 80: 'http', 110: 'pop3',
            143: 'imap', 443: 'https', 445: 'smb', 3389: 'rdp'
        }
        return services.get(port, 'other')
    
    def process_packet(self, pkt):
        """Process individual packet and update flow statistics"""
        flow_key = self.get_flow_key(pkt)
        if not flow_key:
            return
        
        flow = self.flows[flow_key]
        timestamp = float(pkt.time)
        
        # Initialize timestamps
        if flow['start_time'] is None:
            flow['start_time'] = timestamp
        flow['end_time'] = timestamp
        
        # Detect service
        flow['service'] = self.detect_service(pkt)
        
        if IP in pkt:
            src = pkt[IP].src
            dst = pkt[IP].dst
            
            # Determine direction (first seen IP is source)
            if 'src_ip' not in flow:
                flow['src_ip'] = src
                flow['dst_ip'] = dst
            
            is_forward = (src == flow['src_ip'])
            
            # Update packet and byte counts
            pkt_size = len(pkt)
            if is_forward:
                flow['spkts'] += 1
                flow['sbytes'] += pkt_size
                flow['src_ttl'].append(pkt[IP].ttl)
                flow['src_pkt_times'].append(timestamp)
            else:
                flow['dpkts'] += 1
                flow['dbytes'] += pkt_size
                flow['dst_ttl'].append(pkt[IP].ttl)
                flow['dst_pkt_times'].append(timestamp)
            
            # TCP specific features
            if TCP in pkt:
                if is_forward:
                    flow['src_win'].append(pkt[TCP].window)
                else:
                    flow['dst_win'].append(pkt[TCP].window)
                
                flow['tcp_flags'].append(pkt[TCP].flags)
    
    def calculate_features(self):
        """Calculate all required features for each flow"""
        results = []
        
        for flow_key, flow in self.flows.items():
            if flow['start_time'] is None:
                continue
            
            # Duration
            dur = flow['end_time'] - flow['start_time']
            dur = max(dur, 0.000001)  # Avoid division by zero
            
            # Protocol
            proto = flow_key[1]
            
            # Service
            service = flow['service']
            
            # State (simplified)
            state = 'CON' if flow['dpkts'] > 0 else 'INT'
            
            # Packet counts
            spkts = flow['spkts']
            dpkts = flow['dpkts']
            
            # Byte counts
            sbytes = flow['sbytes']
            dbytes = flow['dbytes']
            
            # Rate
            rate = (sbytes + dbytes) / dur if dur > 0 else 0
            
            # TTL
            sttl = int(sum(flow['src_ttl']) / len(flow['src_ttl'])) if flow['src_ttl'] else 0
            dttl = int(sum(flow['dst_ttl']) / len(flow['dst_ttl'])) if flow['dst_ttl'] else 0
            
            # Load (bits per second)
            sload = (sbytes * 8) / dur if dur > 0 else 0
            dload = (dbytes * 8) / dur if dur > 0 else 0
            
            # Loss (simplified - not directly available from packets)
            sloss = 0
            dloss = 0
            
            # Inter-packet times
            sinpkt = self.calc_mean_interval(flow['src_pkt_times'])
            dinpkt = self.calc_mean_interval(flow['dst_pkt_times'])
            
            # Jitter
            sjit = self.calc_jitter(flow['src_pkt_times'])
            djit = self.calc_jitter(flow['dst_pkt_times'])
            
            # Window sizes
            swin = int(sum(flow['src_win']) / len(flow['src_win'])) if flow['src_win'] else 0
            dwin = int(sum(flow['dst_win']) / len(flow['dst_win'])) if flow['dst_win'] else 0
            
            # TCP base sequence numbers (simplified)
            stcpb = 0
            dtcpb = 0
            
            # TCP RTT (simplified)
            tcprtt = 0
            synack = 0
            ackdat = 0
            
            # Mean packet sizes
            smean = sbytes / spkts if spkts > 0 else 0
            dmean = dbytes / dpkts if dpkts > 0 else 0
            
            # HTTP/Application layer features (simplified)
            trans_depth = 0
            response_body_len = 0
            
            # Connection state features (simplified)
            ct_srv_src = 1
            ct_state_ttl = 1
            ct_dst_ltm = 1
            ct_src_dport_ltm = 1
            ct_dst_sport_ltm = 1
            ct_dst_src_ltm = 1
            
            # FTP features
            is_ftp_login = 1 if service == 'ftp' else 0
            ct_ftp_cmd = 0
            
            # HTTP methods
            ct_flw_http_mthd = 0
            
            # Time-based connection counts (simplified)
            ct_src_ltm = 1
            ct_srv_dst = 1
            
            # IP/Port similarity
            is_sm_ips_ports = 0
            
            results.append({
                'dur': round(dur, 6),
                'proto': proto,
                'service': service,
                'state': state,
                'spkts': spkts,
                'dpkts': dpkts,
                'sbytes': sbytes,
                'dbytes': dbytes,
                'rate': round(rate, 2),
                'sttl': sttl,
                'dttl': dttl,
                'sload': round(sload, 2),
                'dload': round(dload, 2),
                'sloss': sloss,
                'dloss': dloss,
                'sinpkt': round(sinpkt, 6),
                'dinpkt': round(dinpkt, 6),
                'sjit': round(sjit, 6),
                'djit': round(djit, 6),
                'swin': swin,
                'stcpb': stcpb,
                'dtcpb': dtcpb,
                'dwin': dwin,
                'tcprtt': tcprtt,
                'synack': synack,
                'ackdat': ackdat,
                'smean': round(smean, 2),
                'dmean': round(dmean, 2),
                'trans_depth': trans_depth,
                'response_body_len': response_body_len,
                'ct_srv_src': ct_srv_src,
                'ct_state_ttl': ct_state_ttl,
                'ct_dst_ltm': ct_dst_ltm,
                'ct_src_dport_ltm': ct_src_dport_ltm,
                'ct_dst_sport_ltm': ct_dst_sport_ltm,
                'ct_dst_src_ltm': ct_dst_src_ltm,
                'is_ftp_login': is_ftp_login,
                'ct_ftp_cmd': ct_ftp_cmd,
                'ct_flw_http_mthd': ct_flw_http_mthd,
                'ct_src_ltm': ct_src_ltm,
                'ct_srv_dst': ct_srv_dst,
                'is_sm_ips_ports': is_sm_ips_ports
            })
        
        return results
    
    def calc_mean_interval(self, times):
        """Calculate mean inter-arrival time"""
        if len(times) < 2:
            return 0
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        return sum(intervals) / len(intervals) if intervals else 0
    
    def calc_jitter(self, times):
        """Calculate jitter (variance in inter-arrival times)"""
        if len(times) < 3:
            return 0
        intervals = [times[i+1] - times[i] for i in range(len(times)-1)]
        if not intervals:
            return 0
        mean = sum(intervals) / len(intervals)
        variance = sum((x - mean) ** 2 for x in intervals) / len(intervals)
        return variance ** 0.5


def capture_packets(interface='eth0', duration=60, output_file='capture.pcap'):
    """Capture network packets using tcpdump"""
    print(f"[*] Capturing packets on {interface} for {duration} seconds...")
    print(f"[*] Output file: {output_file}")
    
    cmd = ['tcpdump', '-i', interface, '-w', output_file, '-s', '0']
    
    try:
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(duration)
        process.terminate()
        process.wait()
        print(f"[+] Capture complete! Packets saved to {output_file}")
        return True
    except PermissionError:
        print("[!] Error: Need root privileges. Run with sudo!")
        return False
    except FileNotFoundError:
        print("[!] Error: tcpdump not found. Install with: sudo apt install tcpdump")
        return False
    except Exception as e:
        print(f"[!] Error during capture: {e}")
        return False


def main():
    if len(sys.argv) < 2:
        print("Usage: sudo python3 capture.py <interface> [duration] [output_dir]")
        print("Example: sudo python3 capture.py eth0 60 ./output")
        sys.exit(1)
    
    interface = sys.argv[1]
    duration = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    output_dir = sys.argv[3] if len(sys.argv) > 3 else f"./capture"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    pcap_file = os.path.join(output_dir, 'capture.pcap')
    csv_file = os.path.join(output_dir, 'features.csv')
    
    # Step 1: Capture packets
    if not capture_packets(interface, duration, pcap_file):
        sys.exit(1)
    
    # Step 2: Extract features
    print(f"\n[*] Extracting flow features from {pcap_file}...")
    extractor = FlowExtractor()
    
    try:
        packets = rdpcap(pcap_file)
        print(f"[*] Processing {len(packets)} packets...")
        
        for i, pkt in enumerate(packets):
            extractor.process_packet(pkt)
            if (i + 1) % 1000 == 0:
                print(f"[*] Processed {i + 1} packets...", end='\r')
        
        print(f"\n[*] Calculating features for {len(extractor.flows)} flows...")
        results = extractor.calculate_features()
        
        # Step 3: Write to CSV
        if results:
            fieldnames = [
                'dur', 'proto', 'service', 'state', 'spkts', 'dpkts', 'sbytes', 'dbytes',
                'rate', 'sttl', 'dttl', 'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt',
                'sjit', 'djit', 'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat',
                'smean', 'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src', 'ct_state_ttl',
                'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 'ct_dst_src_ltm',
                'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd', 'ct_src_ltm', 'ct_srv_dst',
                'is_sm_ips_ports'
            ]
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)
            
            print(f"\n[+] Success! Features extracted to {csv_file}")
            print(f"[+] Total flows: {len(results)}")
        else:
            print("[!] No flows extracted!")
    
    except Exception as e:
        print(f"[!] Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
