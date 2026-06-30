#!/usr/bin/env python3
"""
Convert XAU_1d_data.jsonl to xauusd_data.csv
Source: HuggingFace ZombitX64/xauusd-gold-price-historical-data-2004-2025
"""

import json
import csv
import sys
import urllib.request
from pathlib import Path

JSONL_URL = "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1d_data.jsonl"
JSONL_FILE = Path("XAU_1d_data.jsonl")
OUTPUT_CSV = Path("xauusd_data.csv")


def download_jsonl():
    print(f"Downloading {JSONL_URL} ...")
    urllib.request.urlretrieve(JSONL_URL, JSONL_FILE)
    print(f"Saved to {JSONL_FILE}")


def parse_date(raw: str) -> str:
    # Input:  "2004.06.11 00:00"
    # Output: "2004-06-11"
    return raw[:10].replace(".", "-")


def convert(jsonl_path: Path, csv_path: Path):
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "date":   parse_date(rec["Date"]),
                "Open":   rec["Open"],
                "High":   rec["High"],
                "Low":    rec["Low"],
                "Close":  rec["Close"],
                "Volume": rec["Volume"],
            })

    rows.sort(key=lambda r: r["date"])

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "Open", "High", "Low", "Close", "Volume"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Written {len(rows)} rows → {csv_path}")
    print(f"Date range: {rows[0]['date']} → {rows[-1]['date']}")


if __name__ == "__main__":
    local_jsonl = Path(sys.argv[1]) if len(sys.argv) > 1 else None

    if local_jsonl and local_jsonl.exists():
        print(f"Using local file: {local_jsonl}")
        convert(local_jsonl, OUTPUT_CSV)
    else:
        if not JSONL_FILE.exists():
            download_jsonl()
        convert(JSONL_FILE, OUTPUT_CSV)
