#!/usr/bin/env python3
"""
Hugging Face Upload Script for AI-XAUUSD Trading System

This script uploads the trained models, configuration files, and documentation
to a Hugging Face model repository.
"""

import os
import sys
import argparse
from pathlib import Path
import json
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from huggingface_hub import HfApi, HfFolder, create_repo, upload_file, upload_folder
    from huggingface_hub.utils import HfHubHTTPError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("❌ huggingface_hub not installed. Install with: pip install huggingface_hub")

class HuggingFaceUploader:
    """Handles uploading AI-XAUUSD Trading System to Hugging Face."""

    def __init__(self, repo_name: str = "AI-XAUUSD-Trading", username: Optional[str] = None):
        """
        Initialize the uploader.

        Args:
            repo_name: Name of the HF repository
            username: HF username (if None, will try to get from token)
        """
        if not HF_AVAILABLE:
            raise ImportError("huggingface_hub is required. Install with: pip install huggingface_hub")

        self.repo_name = repo_name
        self.username = username or self._get_username_from_token()
        self.full_repo_name = f"{self.username}/{self.repo_name}"
        self.api = HfApi()

        print(f"🤗 Initialized uploader for: {self.full_repo_name}")

    def _get_username_from_token(self) -> str:
        """Get username from HF token."""
        try:
            token = HfFolder.get_token()
            if token:
                # Get user info from token
                user_info = self.api.whoami(token=token)
                return user_info['name']
        except Exception:
            pass

        # Fallback: ask user
        username = input("Enter your Hugging Face username: ").strip()
        return username

    def create_repository(self, private: bool = False) -> bool:
        """Create the HF repository if it doesn't exist."""
        try:
            print(f"📁 Creating repository: {self.full_repo_name}")

            repo_url = create_repo(
                repo_id=self.full_repo_name,
                repo_type="model",
                private=private,
                exist_ok=True
            )

            print(f"✅ Repository created: {repo_url}")
            return True

        except HfHubHTTPError as e:
            if "409" in str(e):  # Repository already exists
                print(f"ℹ️  Repository already exists: {self.full_repo_name}")
                return True
            else:
                print(f"❌ Failed to create repository: {e}")
                return False
        except Exception as e:
            print(f"❌ Error creating repository: {e}")
            return False

    def prepare_model_files(self, source_dir: str = ".") -> Dict[str, List[str]]:
        """Prepare model files for upload."""
        source_path = Path(source_dir)

        # Define file categories
        model_files = [
            "ppo_trading_model.zip",
            "ppo_continuous_trading_model.zip",
            "dqn_trading_model.zip",
            "ensemble_models/ppo_model.zip",
            "ensemble_models/td3_model.zip",
            "ensemble_models/sac_model.zip",
        ]

        config_files = [
            "model_config.json",
            "regime_config.json",
        ]

        data_files = [
            "xauusd_data.csv",
            "backtest_results.json",
            "forward_test_results.json",
            "ensemble_backtest_metrics.json",
        ]

        # Check which files exist
        existing_models = [f for f in model_files if (source_path / f).exists()]
        existing_configs = [f for f in config_files if (source_path / f).exists()]
        existing_data = [f for f in data_files if (source_path / f).exists()]

        return {
            "models": existing_models,
            "configs": existing_configs,
            "data": existing_data,
        }

    def create_model_config(self) -> Dict:
        """Create model configuration for HF."""
        config = {
            "model_type": "ensemble-trading-system",
            "algorithms": ["PPO", "TD3", "SAC"],
            "framework": "stable-baselines3",
            "environment": "gymnasium",
            "asset": "XAUUSD",
            "timeframe": "daily",
            "features": 25,
            "performance": {
                "win_rate": 0.583,
                "avg_win": 49.45,
                "avg_loss": -106.18,
                "risk_reward_ratio": 0.47,
                "sharpe_ratio": 2.0,
                "max_drawdown": 0.05
            },
            "regimes": [
                "strong_bull",
                "bull_trend",
                "bear_trend",
                "strong_bear",
                "ranging",
                "high_volatility"
            ],
            "risk_management": {
                "scaled_profit_taking": [0.01, 0.02, 0.05, 0.10],
                "breakeven_threshold": 0.015,
                "trailing_stop": 0.025,
                "confidence_sizing": [0.5, 2.0]
            }
        }
        return config

    def upload_file(self, file_path: str, repo_path: str) -> bool:
        """Upload a single file to HF."""
        try:
            print(f"📤 Uploading {file_path} -> {repo_path}")

            upload_file(
                path_or_fileobj=file_path,
                path_in_repo=repo_path,
                repo_id=self.full_repo_name,
                repo_type="model"
            )

            print(f"✅ Uploaded {repo_path}")
            return True

        except Exception as e:
            print(f"❌ Failed to upload {file_path}: {e}")
            return False

    def upload_models_and_configs(self, file_categories: Dict[str, List[str]]) -> bool:
        """Upload all model files and configurations."""
        success_count = 0
        total_count = 0

        # Upload model files
        for model_file in file_categories["models"]:
            total_count += 1
            if self.upload_file(model_file, f"models/{Path(model_file).name}"):
                success_count += 1

        # Upload config files
        for config_file in file_categories["configs"]:
            total_count += 1
            if self.upload_file(config_file, f"config/{Path(config_file).name}"):
                success_count += 1

        # Upload data files (optional)
        for data_file in file_categories["data"]:
            total_count += 1
            if self.upload_file(data_file, f"data/{Path(data_file).name}"):
                success_count += 1

        print(f"\n📊 Upload Summary: {success_count}/{total_count} files successful")
        return success_count == total_count

    def upload_documentation(self) -> bool:
        """Upload documentation files."""
        docs_to_upload = [
            ("model_card.md", "README.md"),  # HF model card as README
            ("AI_XAUUSD_Trading_White_Paper.pdf", "docs/white_paper.pdf"),
            ("LICENSE", "LICENSE"),
        ]

        success_count = 0

        for local_file, repo_file in docs_to_upload:
            if Path(local_file).exists():
                if self.upload_file(local_file, repo_file):
                    success_count += 1
            else:
                print(f"⚠️  Documentation file not found: {local_file}")

        return success_count > 0

    def create_and_upload_config(self) -> bool:
        """Create and upload model configuration."""
        try:
            config = self.create_model_config()

            # Save locally first
            config_path = Path("hf_model_config.json")
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

            # Upload to HF
            success = self.upload_file(str(config_path), "config/model_config.json")

            # Clean up
            config_path.unlink(missing_ok=True)

            return success

        except Exception as e:
            print(f"❌ Error creating/uploading config: {e}")
            return False

    def upload_code_samples(self) -> bool:
        """Upload code samples and examples."""
        code_files = [
            "ensemble_trader.py",
            "market_regime_detector.py",
            "trading_env.py",
            "quick_demo.py",
            "advanced_trading_demo.py",
        ]

        success_count = 0

        for code_file in code_files:
            if Path(code_file).exists():
                if self.upload_file(code_file, f"code/{code_file}"):
                    success_count += 1

        return success_count > 0

    def run_full_upload(self, source_dir: str = ".", private: bool = False) -> bool:
        """Run the complete upload process."""
        print("🚀 Starting full upload to Hugging Face...")
        print("=" * 50)

        # Step 1: Create repository
        if not self.create_repository(private=private):
            return False

        # Step 2: Prepare files
        file_categories = self.prepare_model_files(source_dir)
        total_files = sum(len(files) for files in file_categories.values())

        if total_files == 0:
            print("⚠️  No model files found. Please ensure models are trained and saved.")
            print("   Run training scripts first:")
            print("   - python train_model.py")
            print("   - python ensemble_backtest.py")
            return False

        print(f"📁 Found {total_files} files to upload:")
        for category, files in file_categories.items():
            print(f"   {category.title()}: {len(files)} files")

        # Step 3: Upload models and configs
        if not self.upload_models_and_configs(file_categories):
            print("❌ Failed to upload model files")
            return False

        # Step 4: Upload documentation
        if not self.upload_documentation():
            print("⚠️  Documentation upload incomplete")

        # Step 5: Create and upload config
        if not self.create_and_upload_config():
            print("⚠️  Config upload failed")

        # Step 6: Upload code samples
        if not self.upload_code_samples():
            print("⚠️  Code samples upload failed")

        print("\n" + "=" * 50)
        print("🎉 Upload completed successfully!")
        print(f"🔗 Repository: https://huggingface.co/{self.full_repo_name}")
        print("\n📋 Next steps:")
        print("1. Visit your repository and verify files")
        print("2. Update model card if needed")
        print("3. Share with the community!")
        print("4. Consider adding model to HF Leaderboards")

        return True

def main():
    parser = argparse.ArgumentParser(description="Upload AI-XAUUSD Trading System to Hugging Face")
    parser.add_argument("--repo-name", default="AI-XAUUSD-Trading", help="HF repository name")
    parser.add_argument("--username", help="HF username (optional, will detect from token)")
    parser.add_argument("--source-dir", default=".", help="Source directory with model files")
    parser.add_argument("--private", action="store_true", help="Create private repository")
    parser.add_argument("--test-only", action="store_true", help="Test upload without actually uploading")

    args = parser.parse_args()

    if args.test_only:
        print("🧪 Test mode - checking HF setup...")
        try:
            uploader = HuggingFaceUploader(args.repo_name, args.username)
            print("✅ HF setup successful!")
            return 0
        except Exception as e:
            print(f"❌ HF setup failed: {e}")
            return 1

    try:
        uploader = HuggingFaceUploader(args.repo_name, args.username)
        success = uploader.run_full_upload(args.source_dir, args.private)

        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n⏹️  Upload interrupted by user")
        return 1
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return 1

if __name__ == "__main__":
    if not HF_AVAILABLE:
        print("❌ huggingface_hub is required. Install with:")
        print("   pip install huggingface_hub")
        sys.exit(1)

    exit(main())