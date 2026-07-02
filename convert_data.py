#!/usr/bin/env python3
"""
Convert XAUUSD JSONL data to CSV.
Handles D1, M5, M1 timeframes from HuggingFace ZombitX64 datasets.

Usage:
  python3 convert_data.py                         # download & convert D1
  python3 convert_data.py --tf m5                 # download & convert M5
  python3 convert_data.py --tf m1                 # download & convert M1 (~648 MB, ~7M bars)
  python3 convert_data.py XAU_1d_data.jsonl       # convert local D1 file
  python3 convert_data.py XAU_1m_data.jsonl --tf m1  # convert local M1 file
"""

import argparse
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

URLS = {
    "d1": "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1d_data.jsonl",
    "m5": "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_5m_data.jsonl",
    "m1": "https://huggingface.co/datasets/ZombitX64/xauusd-gold-price-historical-data-2004-2025/resolve/main/XAU_1m_data.jsonl",
}

OUTPUTS = {
    "d1": Path("xauusd_data.csv"),
    "m5": Path("xauusd_m5_data.csv"),
    "m1": Path("xauusd_m1_data.csv"),
}

JSONL_FILES = {
    "d1": Path("XAU_1d_data.jsonl"),
    "m5": Path("XAU_5m_data.jsonl"),
    "m1": Path("XAU_1m_data.jsonl"),
}


def parse_date(raw: str, tf: str) -> str:
    # Input:  "2004.06.11 07:15"
    # D1 output: "2004-06-11"
    # M5/M1 output: "2004-06-11 07:15:00"
    date_part = raw[:10].replace(".", "-")
    if tf == "d1":
        return date_part
    time_part = raw[11:16]  # "07:15"
    return f"{date_part} {time_part}:00"


class _ProgressHook:
    def __init__(self, label: str):
        self.label   = label
        self.start   = time.time()
        self.last_mb = 0

    def __call__(self, count, block_size, total_size):
        downloaded = count * block_size
        mb = downloaded / 1_048_576
        total_mb = total_size / 1_048_576 if total_size > 0 else 0
        if mb - self.last_mb >= 10 or downloaded >= total_size:
            elapsed = time.time() - self.start
            speed   = mb / elapsed if elapsed > 0 else 0
            pct     = downloaded / total_size * 100 if total_size > 0 else 0
            print(
                f"\r  {self.label}: {mb:.0f} MB / {total_mb:.0f} MB "
                f"({pct:.0f}%)  {speed:.1f} MB/s",
                end="", flush=True,
            )
            self.last_mb = mb
        if total_size > 0 and downloaded >= total_size:
            print()


def download(tf: str) -> Path:
    url  = URLS[tf]
    dest = JSONL_FILES[tf]
    size_mb = {"d1": "~0.1", "m5": "~200", "m1": "~620"}.get(tf, "?")
    print(f"Downloading {tf.upper()} data ({size_mb} MB) ...")
    print(f"  Source: {url}")
    urllib.request.urlretrieve(url, dest, reporthook=_ProgressHook(tf.upper()))
    print(f"  Saved → {dest}")
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
    parser.add_argument("--tf", choices=["d1", "m5", "m1"], default="d1",
                        help="Timeframe: d1 / m5 / m1. Default: d1")
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
