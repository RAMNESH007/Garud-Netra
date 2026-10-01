import csv
import random
import string
from datetime import datetime, timedelta
from pathlib import Path


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

random.seed(42)

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_NORMAL = 300
NUM_BURST = 50
NUM_FANOUT = 50
NUM_PEELING = 50


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def random_txid():
    return "".join(random.choices(string.hexdigits.lower(), k=16))


def random_ip():
    return ".".join(str(random.randint(1, 254)) for _ in range(4))


def random_address(prefix="addr"):
    suffix = "".join(
        random.choices(string.ascii_letters + string.digits, k=10)
    )
    return f"{prefix}_{suffix}"


def random_country():
    countries = ["IN", "US", "DE", "GB", "SG", "CA"]
    return random.choice(countries)


def random_asn():
    return random.randint(1000, 65000)


def random_timestamp(start_time):
    seconds = random.randint(0, 60 * 60 * 24 * 30)
    return start_time + timedelta(seconds=seconds)


# --------------------------------------------------
# DATA GENERATION
# --------------------------------------------------

start_time = datetime(2026, 1, 1, 0, 0, 0)

transactions = []
network_events = []


# --------------------------------------------------
# 1. NORMAL TRANSACTIONS
# --------------------------------------------------

for _ in range(NUM_NORMAL):

    txid = random_txid()

    src_ip = random_ip()
    dst_ip = random_ip()

    timestamp = random_timestamp(start_time)

    input_address = random_address("input")
    output_address = random_address("output")

    amount = round(random.uniform(0.01, 5.0), 6)
    fee = round(random.uniform(0.00001, 0.01), 6)

    transactions.append({
        "timestamp": timestamp.isoformat(),
        "txid": txid,
        "input_addresses": input_address,
        "output_addresses": output_address,
        "input_amounts": amount,
        "output_amounts": round(amount - fee, 6),
        "fee": fee,
        "script_type": random.choice(["P2PKH", "P2SH", "P2WPKH"])
    })

    network_events.append({
        "timestamp": timestamp.isoformat(),
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(30000, 50000),
        "dst_port": 8333,
        "txid": txid,
        "geo_country": random_country(),
        "asn": random_asn()
    })


# --------------------------------------------------
# 2. HIGH-FREQUENCY / BURST ACTIVITY
# --------------------------------------------------

burst_ip = "185.100.50.10"
burst_address = random_address("burst")

for i in range(NUM_BURST):

    timestamp = start_time + timedelta(
        minutes=10,
        seconds=i * 5
    )

    txid = random_txid()

    output_address = random_address("burst_out")

    amount = round(random.uniform(2.0, 8.0), 6)
    fee = round(random.uniform(0.001, 0.01), 6)

    transactions.append({
        "timestamp": timestamp.isoformat(),
        "txid": txid,
        "input_addresses": burst_address,
        "output_addresses": output_address,
        "input_amounts": amount,
        "output_amounts": round(amount - fee, 6),
        "fee": fee,
        "script_type": "P2WPKH"
    })

    network_events.append({
        "timestamp": timestamp.isoformat(),
        "src_ip": burst_ip,
        "dst_ip": random_ip(),
        "src_port": random.randint(30000, 50000),
        "dst_port": 8333,
        "txid": txid,
        "geo_country": "XX",
        "asn": 64550
    })


# --------------------------------------------------
# 3. HIGH FAN-OUT ACTIVITY
# --------------------------------------------------

fanout_address = random_address("fanout")
fanout_ip = "172.20.10.20"

for i in range(NUM_FANOUT):

    timestamp = start_time + timedelta(
        days=5,
        minutes=i
    )

    txid = random_txid()

    output_address = random_address("fanout_out")

    amount = round(random.uniform(1.0, 4.0), 6)
    fee = round(random.uniform(0.0005, 0.005), 6)

    transactions.append({
        "timestamp": timestamp.isoformat(),
        "txid": txid,
        "input_addresses": fanout_address,
        "output_addresses": output_address,
        "input_amounts": amount,
        "output_amounts": round(amount - fee, 6),
        "fee": fee,
        "script_type": "P2SH"
    })

    network_events.append({
        "timestamp": timestamp.isoformat(),
        "src_ip": fanout_ip,
        "dst_ip": random_ip(),
        "src_port": random.randint(30000, 50000),
        "dst_port": 8333,
        "txid": txid,
        "geo_country": "RU",
        "asn": 64510
    })


# --------------------------------------------------
# 4. PEELING-CHAIN-LIKE ACTIVITY
# --------------------------------------------------

current_address = random_address("peel")
peeling_ip = "45.90.10.10"

remaining_amount = 50.0

for i in range(NUM_PEELING):

    timestamp = start_time + timedelta(
        days=10,
        minutes=i * 10
    )

    txid = random_txid()

    next_address = random_address("peel_next")

    transfer_amount = round(
        remaining_amount * random.uniform(0.80, 0.95),
        6
    )

    fee = round(random.uniform(0.001, 0.01), 6)

    output_amount = max(
        round(transfer_amount - fee, 6),
        0.000001
    )

    transactions.append({
        "timestamp": timestamp.isoformat(),
        "txid": txid,
        "input_addresses": current_address,
        "output_addresses": next_address,
        "input_amounts": transfer_amount,
        "output_amounts": output_amount,
        "fee": fee,
        "script_type": "P2WSH"
    })

    network_events.append({
        "timestamp": timestamp.isoformat(),
        "src_ip": peeling_ip,
        "dst_ip": random_ip(),
        "src_port": random.randint(30000, 50000),
        "dst_port": 8333,
        "txid": txid,
        "geo_country": "XX",
        "asn": 64520
    })

    remaining_amount = output_amount
    current_address = next_address


# --------------------------------------------------
# SAVE TRANSACTION DATA
# --------------------------------------------------

transaction_file = OUTPUT_DIR / "synthetic_transactions.csv"

with open(transaction_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "timestamp",
            "txid",
            "input_addresses",
            "output_addresses",
            "input_amounts",
            "output_amounts",
            "fee",
            "script_type"
        ]
    )

    writer.writeheader()
    writer.writerows(transactions)


# --------------------------------------------------
# SAVE NETWORK DATA
# --------------------------------------------------

network_file = OUTPUT_DIR / "synthetic_network_events.csv"

with open(network_file, "w", newline="") as file:
    writer = csv.DictWriter(
        file,
        fieldnames=[
            "timestamp",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
            "txid",
            "geo_country",
            "asn"
        ]
    )

    writer.writeheader()
    writer.writerows(network_events)


print("Synthetic dataset created successfully!")
print()
print(f"Transactions : {len(transactions)}")
print(f"Network events: {len(network_events)}")
print()
print(f"Transaction file: {transaction_file}")
print(f"Network file    : {network_file}")
