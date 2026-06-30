#!/usr/bin/env python3
"""
Convert XAUUSD JSONL data to CSV.
Handles both D1 and M5 timeframes from HuggingFace ZombitX64 datasets.

Usage:
  python3 convert_data.py                         # download & convert D1
  python3 convert_data.py --tf m5                 # download & convert M5
  python3 convert_data.py XAU_1d_data.jsonl       # convert local D1 file
  python3 convert_data.py XAU_5m_data.jsonl --tf m5  # convert local M5 file
"""

import argparse
import csv
import json
import urllib.request
from pathlib import Path

URLS = {
    "d1": "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1d_data.jsonl",
    "m5": "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_5m_data.jsonl",
}

OUTPUTS = {
    "d1": Path("xauusd_data.csv"),
    "m5": Path("xauusd_m5_data.csv"),
}

JSONL_FILES = {
    "d1": Path("XAU_1d_data.jsonl"),
    "m5": Path("XAU_5m_data.jsonl"),
}


def parse_date(raw: str, tf: str) -> str:
    # Input:  "2004.06.11 07:15"
    # D1 output: "2004-06-11"
    # M5 output: "2004-06-11 07:15:00"
    date_part = raw[:10].replace(".", "-")
    if tf == "d1":
        return date_part
    time_part = raw[11:16]  # "07:15"
    return f"{date_part} {time_part}:00"


def download(tf: str) -> Path:
    url = URLS[tf]
    dest = JSONL_FILES[tf]
    print(f"Downloading {url} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def convert(jsonl_path: Path, csv_path: Path, tf: str):
    print(f"Converting {jsonl_path} → {csv_path} ...")
    rows = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            rows.append({
                "date":   parse_date(rec["Date"], tf),
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

    print(f"Written {len(rows):,} rows → {csv_path}")
    print(f"Date range: {rows[0]['date']} → {rows[-1]['date']}")


def main():
    parser = argparse.ArgumentParser(description="Convert XAUUSD JSONL to CSV")
    parser.add_argument("jsonl", nargs="?", help="Local JSONL file (optional)")
    parser.add_argument("--tf", choices=["d1", "m5"], default="d1",
                        help="Timeframe: d1 (daily) or m5 (5-minute). Default: d1")
    args = parser.parse_args()

    tf = args.tf
    csv_path = OUTPUTS[tf]

    if args.jsonl:
        jsonl_path = Path(args.jsonl)
    else:
        jsonl_path = JSONL_FILES[tf]
        if not jsonl_path.exists():
            jsonl_path = download(tf)

    convert(jsonl_path, csv_path, tf)


if __name__ == "__main__":
    main()
