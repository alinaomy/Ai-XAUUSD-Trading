#!/usr/bin/env python3
"""
Download pre-trained AI models for XAUUSD Trading System

This script downloads the trained ensemble models and market regime detector
from Hugging Face Hub for immediate use.
"""

import os
import requests
from pathlib import Path
import zipfile
import tarfile
import shutil
from tqdm import tqdm
import argparse

# Hugging Face model repository
HF_REPO = "JonusNattapong/AI-XAUUSD-Trading"
HF_BASE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main"

# Model files to download
MODEL_FILES = [
    "models/ensemble_ppo.zip",
    "models/ensemble_td3.zip",
    "models/ensemble_sac.zip",
    "models/market_regime_detector.pkl",
    "models/feature_scaler.pkl",
    "config/model_config.json",
    "config/regime_config.json",
]

class ModelDownloader:
    """Handles downloading and extracting pre-trained models."""

    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.models_dir = self.base_dir / "models"
        self.config_dir = self.base_dir / "config"

        # Create directories
        self.models_dir.mkdir(exist_ok=True)
        self.config_dir.mkdir(exist_ok=True)

    def download_file(self, filename: str, output_path: Path) -> bool:
        """Download a single file from Hugging Face."""
        url = f"{HF_BASE_URL}/{filename}"

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as file, tqdm(
                desc=f"Downloading {filename}",
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as progress_bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    progress_bar.update(size)

            return True

        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to download {filename}: {e}")
            return False
        except Exception as e:
            print(f"❌ Error downloading {filename}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip or tar archives."""
        try:
            if archive_path.suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffix in ['.tar', '.gz', '.bz2']:
                with tarfile.open(archive_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"❌ Unsupported archive format: {archive_path}")
                return False

            # Remove archive after extraction
            archive_path.unlink()
            return True

        except Exception as e:
            print(f"❌ Error extracting {archive_path}: {e}")
            return False

    def download_models(self, force: bool = False) -> bool:
        """Download all required models and configurations."""
        print("🚀 Downloading AI-XAUUSD Trading Models...")
        print(f"📁 Target directory: {self.base_dir.absolute()}")

        success_count = 0
        total_count = len(MODEL_FILES)

        for filename in MODEL_FILES:
            file_path = self.base_dir / filename
            temp_path = file_path.with_suffix(file_path.suffix + '.tmp')

            # Skip if file exists and not forcing download
            if file_path.exists() and not force:
                print(f"✅ {filename} already exists, skipping...")
                success_count += 1
                continue

            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download file
            if self.download_file(filename, temp_path):
                # Move temp file to final location
                temp_path.rename(file_path)

                # Extract archives
                if file_path.suffix in ['.zip', '.tar', '.gz', '.bz2']:
                    extract_to = file_path.parent
                    if filename.startswith('models/'):
                        # Extract model archives to models directory
                        extract_to = self.models_dir

                    print(f"📦 Extracting {filename}...")
                    if self.extract_archive(file_path, extract_to):
                        print(f"✅ Successfully extracted {filename}")
                    else:
                        print(f"❌ Failed to extract {filename}")
                        continue

                success_count += 1
                print(f"✅ Downloaded {filename}")
            else:
                print(f"❌ Failed to download {filename}")

        print(f"\n📊 Download Summary: {success_count}/{total_count} files successful")

        if success_count == total_count:
            print("🎉 All models downloaded successfully!")
            print("\n🚀 You can now run:")
            print("   python advanced_trading_demo.py")
            print("   python live_ensemble_trading.py")
            return True
        else:
            print("⚠️  Some files failed to download. Please check your internet connection.")
            return False

    def verify_models(self) -> bool:
        """Verify that all required model files are present."""
        print("🔍 Verifying model files...")

        required_files = [
            "models/ppo_model.zip",  # Extracted from ensemble_ppo.zip
            "models/td3_model.zip",  # Extracted from ensemble_td3.zip
            "models/sac_model.zip",  # Extracted from ensemble_sac.zip
            "models/market_regime_detector.pkl",
            "models/feature_scaler.pkl",
            "config/model_config.json",
            "config/regime_config.json",
        ]

        missing_files = []
        for filename in required_files:
            if not (self.base_dir / filename).exists():
                missing_files.append(filename)

        if missing_files:
            print("❌ Missing model files:")
            for filename in missing_files:
                print(f"   - {filename}")
            return False
        else:
            print("✅ All model files verified!")
            return True

def main():
    parser = argparse.ArgumentParser(description="Download AI-XAUUSD Trading Models")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if files already exist"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing models without downloading"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Target directory for model downloads"
    )

    args = parser.parse_args()

    downloader = ModelDownloader(args.dir)

    if args.verify_only:
        success = downloader.verify_models()
    else:
        success = downloader.download_models(force=args.force)
        if success:
            downloader.verify_models()

    return 0 if success else 1

if __name__ == "__main__":
    exit(main())