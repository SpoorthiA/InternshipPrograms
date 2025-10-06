import os
import re
import json
import csv
from datetime import datetime
from collections import defaultdict

def read_log_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as file:
        return file.readlines()

def parse_log_line(line):
    pattern = re.compile(
        r'(?P<month>\w{3})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+'
        r'(?P<host>\S+)\s+\S+:\s+(?P<status>Failed|Accepted)\s+\S+\s+for\s+user\s+'
        r'(?P<user>\S+)\s+from\s+(?P<ip>\d{1,3}(?:\.\d{1,3}){3})'
    )
    match = pattern.search(line)
    if match:
        data = match.groupdict()
        try:
            timestamp = datetime.strptime(
                f"{data['month']} {data['day']} {data['time']}", "%b %d %H:%M:%S"
            ).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            timestamp = None
        return {
            "timestamp": timestamp,
            "host": data["host"],
            "status": data["status"],
            "user": data["user"],
            "ip": data["ip"],
        }
    return None

def parse_all_logs(lines):
    parsed = []
    for line in lines:
        entry = parse_log_line(line)
        if entry:
            parsed.append(entry)
    return parsed

def filter_failed_attempts(entries):
    return [e for e in entries if e["status"].lower() == "failed"]

def filter_by_date(entries, start_time, end_time):
    filtered = []
    for e in entries:
        if e["timestamp"]:
            t = datetime.strptime(e["timestamp"], "%Y-%m-%d %H:%M:%S")
            if start_time <= t <= end_time:
                filtered.append(e)
    return filtered

def count_failed_by_ip(entries):
    ip_counts = defaultdict(int)
    for e in entries:
        ip_counts[e["ip"]] += 1
    sorted_counts = sorted(ip_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts

def count_failed_by_user(entries):
    user_counts = defaultdict(int)
    for e in entries:
        user_counts[e["user"]] += 1
    sorted_counts = sorted(user_counts.items(), key=lambda x: x[1], reverse=True)
    return sorted_counts

def export_to_json(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

def export_to_csv(data, filename):
    if not data:
        return
    keys = data[0].keys()
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(data)

def export_summary_to_csv(ip_counts, user_counts, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Category", "Value", "Count"])
        for ip, count in ip_counts:
            writer.writerow(["IP", ip, count])
        for user, count in user_counts:
            writer.writerow(["User", user, count])

def display_summary(ip_counts, user_counts):
    print("=" * 60)
    print("Top IPs with Most Failed Attempts")
    print("-" * 60)
    for ip, count in ip_counts[:10]:
        print(f"{ip:<20}{count:>5}")
    print("\nTop Users with Most Failed Attempts")
    print("-" * 60)
    for user, count in user_counts[:10]:
        print(f"{user:<20}{count:>5}")
    print("=" * 60)

def generate_log_summary(entries):
    total = len(entries)
    failed = len([e for e in entries if e["status"].lower() == "failed"])
    accepted = len([e for e in entries if e["status"].lower() == "accepted"])
    hosts = len(set(e["host"] for e in entries))
    ips = len(set(e["ip"] for e in entries))
    users = len(set(e["user"] for e in entries))
    summary = {
        "total_entries": total,
        "failed_logins": failed,
        "accepted_logins": accepted,
        "unique_hosts": hosts,
        "unique_ips": ips,
        "unique_users": users,
        "failure_rate": round((failed / total * 100), 2) if total > 0 else 0,
    }
    return summary

def print_log_summary(summary):
    print("\nLog File Summary")
    print("-" * 60)
    for key, val in summary.items():
        print(f"{key.replace('_', ' ').title():<25}: {val}")
    print("-" * 60)

def create_sample_log(file_path):
    sample_logs = [
        "Jan 12 07:24:32 server sshd[1001]: Failed password for user root from 192.168.1.101 port 22 ssh2",
        "Jan 12 07:25:12 server sshd[1002]: Accepted password for user admin from 192.168.1.55 port 22 ssh2",
        "Jan 12 07:26:10 server sshd[1003]: Failed password for user guest from 192.168.1.102 port 22 ssh2",
        "Jan 12 07:27:45 server sshd[1004]: Failed password for user root from 192.168.1.101 port 22 ssh2",
        "Jan 12 07:29:01 server sshd[1005]: Failed password for user test from 10.0.0.7 port 22 ssh2",
        "Jan 12 07:30:15 server sshd[1006]: Accepted password for user admin from 10.0.0.9 port 22 ssh2",
        "Jan 12 07:32:01 server sshd[1007]: Failed password for user user1 from 172.16.0.3 port 22 ssh2",
        "Jan 12 07:33:22 server sshd[1008]: Failed password for user root from 192.168.1.101 port 22 ssh2",
        "Jan 12 07:35:11 server sshd[1009]: Accepted password for user dev from 192.168.1.77 port 22 ssh2",
        "Jan 12 07:36:22 server sshd[1010]: Failed password for user guest from 192.168.1.102 port 22 ssh2",
        "Jan 12 07:38:01 server sshd[1011]: Failed password for user test from 10.0.0.7 port 22 ssh2",
        "Jan 12 07:39:44 server sshd[1012]: Failed password for user root from 192.168.1.101 port 22 ssh2",
        "Jan 12 07:41:01 server sshd[1013]: Accepted password for user dev from 10.0.0.5 port 22 ssh2",
        "Jan 12 07:43:10 server sshd[1014]: Failed password for user guest from 192.168.1.102 port 22 ssh2",
        "Jan 12 07:44:55 server sshd[1015]: Failed password for user root from 192.168.1.101 port 22 ssh2",
        "Jan 12 07:46:22 server sshd[1016]: Failed password for user test from 10.0.0.7 port 22 ssh2",
        "Jan 12 07:47:30 server sshd[1017]: Accepted password for user admin from 10.0.0.9 port 22 ssh2",
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sample_logs))

def save_summary_json(summary, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

def interactive_filter(entries):
    ips = list(set(e["ip"] for e in entries))
    users = list(set(e["user"] for e in entries))
    print("\nAvailable IPs:")
    for ip in ips:
        print(" ", ip)
    print("\nAvailable Users:")
    for u in users:
        print(" ", u)
    ip_choice = input("\nEnter an IP to filter (or leave blank): ").strip()
    user_choice = input("Enter a username to filter (or leave blank): ").strip()
    filtered = entries
    if ip_choice:
        filtered = [e for e in filtered if e["ip"] == ip_choice]
    if user_choice:
        filtered = [e for e in filtered if e["user"] == user_choice]
    if not filtered:
        print("No matching records found.")
    else:
        print(f"\nFiltered {len(filtered)} entries:")
        for e in filtered:
            print(e)
    return filtered

def main():
    print("=" * 60)
    print("CYBERSECURITY LOG ANALYZER - DAY 1 PRACTICE")
    print("=" * 60)
    file_path = "sample_auth.log"
    if not os.path.exists(file_path):
        print("Creating sample log file...")
        create_sample_log(file_path)
    lines = read_log_file(file_path)
    parsed = parse_all_logs(lines)
    failed = filter_failed_attempts(parsed)
    start = datetime.strptime("2025-01-01 00:00:00", "%Y-%m-%d %H:%M:%S")
    end = datetime.strptime("2025-12-31 23:59:59", "%Y-%m-%d %H:%M:%S")
    time_filtered = filter_by_date(failed, start, end)
    ip_counts = count_failed_by_ip(time_filtered)
    user_counts = count_failed_by_user(time_filtered)
    summary = generate_log_summary(parsed)
    print_log_summary(summary)
    display_summary(ip_counts, user_counts)
    export_to_json(parsed, "parsed_logs.json")
    export_to_csv(parsed, "parsed_logs.csv")
    export_summary_to_csv(ip_counts, user_counts, "failed_summary.csv")
    save_summary_json(summary, "summary.json")
    interactive_filter(parsed)
    print("\nAnalysis completed. Files saved in current directory.")

if __name__ == "__main__":
    main()
