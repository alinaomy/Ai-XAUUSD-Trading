#!/usr/bin/env python3
"""
arXiv Submission Preparation Script

This script prepares the AI-XAUUSD Trading white paper for submission to arXiv.
arXiv requires specific formatting and metadata for submissions.
"""

import os
import tarfile
import shutil
from pathlib import Path
from datetime import datetime
import json

class ArxivSubmissionPreparer:
    """Prepares documents for arXiv submission."""

    def __init__(self, source_dir: str = ".", output_dir: str = "arxiv_submission"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def create_submission_metadata(self) -> dict:
        """Create metadata for arXiv submission."""
        metadata = {
            "title": "AI-Driven XAUUSD Trading System: Maximum Profitability Framework",
            "authors": [
                {
                    "name": "JonusNattapong",
                    "affiliation": "Independent Researcher",
                    "email": "jonusnattapong@zombitx64.com"
                },
                {
                    "name": "Zombitx64",
                    "affiliation": "Independent Researcher",
                    "email": "jonusnattapong@zombitx64.com"
                }
            ],
            "abstract": """This paper presents a comprehensive AI-driven trading system for XAUUSD (Gold vs US Dollar) that achieves the ambitious target of 45 USD daily profit through advanced machine learning techniques, market regime detection, and adaptive risk management. The system combines ensemble reinforcement learning algorithms with sophisticated exit strategies and real-time market condition analysis.

Key achievements include: 58.3% win rate (26% improvement over baseline), 11x better average wins ($4.16 → $49.45), risk-reward ratio of 1:0.47 (2.8x improvement), and market regime-adaptive parameters across 6 distinct market conditions.

The methodology integrates PPO, TD3, and SAC reinforcement learning algorithms in a confidence-weighted ensemble framework, with real-time market regime detection using ADX and volatility metrics. Advanced risk management features include scaled profit-taking, breakeven stops, confidence-based position sizing, and trailing stops.""",
            "categories": [
                "q-fin.TR",  # Trading
                "q-fin.CP",  # Computational Finance
                "cs.AI",     # Artificial Intelligence
                "cs.LG"      # Machine Learning
            ],
            "keywords": [
                "reinforcement learning",
                "algorithmic trading",
                "ensemble methods",
                "market regime detection",
                "risk management",
                "XAUUSD",
                "gold trading",
                "machine learning finance"
            ],
            "comments": "26 pages, 12 figures, 8 tables. Source code and models available at https://github.com/JonusNattapong/AI-XAUUSD-Trading and https://huggingface.co/JonusNattapong/AI-XAUUSD-Trading",
            "journal_ref": None,
            "doi": None,
            "license": "arXiv.org perpetual, non-exclusive license to distribute this article",
            "submission_date": datetime.now().strftime("%Y-%m-%d"),
            "version": "1.0"
        }
        return metadata

    def create_arxiv_metadata_file(self) -> str:
        """Create the 00README.XXX file required by arXiv."""
        metadata = self.create_submission_metadata()

        readme_content = f"""Title: {metadata['title']}

Authors: {', '.join([author['name'] for author in metadata['authors']])}

Abstract: {metadata['abstract']}

Categories: {', '.join(metadata['categories'])}

Comments: {metadata['comments']}

Submission Date: {metadata['submission_date']}

License: {metadata['license']}

---

This submission contains the white paper for the AI-Driven XAUUSD Trading System.

Files included:
- AI_XAUUSD_Trading_White_Paper.pdf: Main white paper (PDF format)
- metadata.json: Submission metadata

For source code and trained models, see:
- GitHub: https://github.com/JonusNattapong/AI-XAUUSD-Trading
- Hugging Face: https://huggingface.co/JonusNattapong/AI-XAUUSD-Trading

Contact: jonusnattapong@zombitx64.com
"""

        readme_path = self.output_dir / "00README.txt"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        return str(readme_path)

    def copy_pdf_file(self) -> str:
        """Copy the PDF file to submission directory."""
        source_pdf = self.source_dir / "AI_XAUUSD_Trading_White_Paper.pdf"

        if not source_pdf.exists():
            raise FileNotFoundError(f"PDF file not found: {source_pdf}")

        dest_pdf = self.output_dir / "AI_XAUUSD_Trading_White_Paper.pdf"
        shutil.copy2(source_pdf, dest_pdf)

        return str(dest_pdf)

    def create_metadata_json(self) -> str:
        """Create a JSON file with submission metadata."""
        metadata = self.create_submission_metadata()

        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        return str(metadata_path)

    def create_tarball(self) -> str:
        """Create the submission tarball."""
        tarball_name = f"arxiv_submission_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tar.gz"
        tarball_path = self.source_dir / tarball_name

        with tarfile.open(tarball_path, "w:gz") as tar:
            # Add files from output directory
            for file_path in self.output_dir.iterdir():
                if file_path.is_file():
                    tar.add(file_path, arcname=file_path.name)

        return str(tarball_path)

    def prepare_submission(self) -> dict:
        """Prepare the complete arXiv submission package."""
        print("📄 Preparing arXiv submission package...")

        # Create metadata file
        readme_file = self.create_arxiv_metadata_file()
        print(f"✅ Created README: {readme_file}")

        # Copy PDF
        pdf_file = self.copy_pdf_file()
        print(f"✅ Copied PDF: {pdf_file}")

        # Create metadata JSON
        metadata_file = self.create_metadata_json()
        print(f"✅ Created metadata: {metadata_file}")

        # Create tarball
        tarball_file = self.create_tarball()
        print(f"✅ Created submission tarball: {tarball_file}")

        submission_info = {
            "tarball": tarball_file,
            "readme": readme_file,
            "pdf": pdf_file,
            "metadata": metadata_file,
            "size_mb": round(os.path.getsize(tarball_file) / (1024 * 1024), 2)
        }

        return submission_info

def create_arxiv_submission_script():
    """Create a script for arXiv SWORD submission."""
    script_content = '''#!/usr/bin/env python3
"""
arXiv SWORD Submission Script

This script submits the prepared tarball to arXiv using the SWORD protocol.
Note: You need valid arXiv credentials and the submission must be approved.

Requirements:
- Valid arXiv account with submission permissions
- Prepared submission tarball
- Python requests library
"""

import requests
import sys
from pathlib import Path

def submit_to_arxiv(tarball_path: str, username: str, password: str):
    """
    Submit to arXiv using SWORD protocol.

    Note: This is a simplified example. arXiv SWORD submission requires:
    1. Valid arXiv account with submission permissions
    2. Proper authentication
    3. Correct submission format
    4. Manual approval process

    For actual submission, use arXiv's web interface or contact them for API access.
    """

    if not Path(tarball_path).exists():
        print(f"❌ Tarball not found: {tarball_path}")
        return False

    # arXiv SWORD endpoint (this is a placeholder - actual endpoint may vary)
    sword_url = "https://arxiv.org/sword/deposit"

    print("⚠️  arXiv SWORD Submission Notes:")
    print("1. arXiv SWORD submission requires special permissions")
    print("2. Most users should use the web interface: https://arxiv.org/submit")
    print("3. Contact arXiv administrators for API access if needed")
    print("4. Submissions go through moderation before publication")
    print()

    print("📤 For manual submission:")
    print(f"   File to upload: {tarball_path}")
    print("   Go to: https://arxiv.org/submit")
    print("   Select category: Quantitative Finance (q-fin)")
    print("   Upload the tarball and follow the submission process")
    print()

    # This is just a demonstration - actual SWORD submission is more complex
    try:
        with open(tarball_path, 'rb') as f:
            files = {'file': ('submission.tar.gz', f, 'application/x-tar-gz')}

            # Note: This won't work without proper arXiv credentials and setup
            response = requests.post(
                sword_url,
                auth=(username, password),
                files=files,
                headers={'Content-Type': 'application/zip'}
            )

            print(f"Response Status: {response.status_code}")
            print(f"Response: {response.text}")

            return response.status_code in [200, 201, 202]

    except Exception as e:
        print(f"❌ Submission failed: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python arxiv_submit.py <tarball_path> <username> <password>")
        print("Example: python arxiv_submit.py arxiv_submission_20251010.tar.gz myuser mypass")
        print()
        print("⚠️  Note: arXiv SWORD requires special permissions.")
        print("   Most users should submit via: https://arxiv.org/submit")
        sys.exit(1)

    tarball_path = sys.argv[1]
    username = sys.argv[2]
    password = sys.argv[3]

    success = submit_to_arxiv(tarball_path, username, password)

    if success:
        print("✅ Submission successful!")
    else:
        print("❌ Submission failed. Check the notes above.")
'''

    script_path = Path("arxiv_submit.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    # Make executable on Unix-like systems
    try:
        script_path.chmod(0o755)
    except:
        pass  # Windows doesn't support chmod

    return str(script_path)

def main():
    """Main preparation function."""
    print("🚀 arXiv Submission Preparation")
    print("=" * 40)

    # Check if PDF exists
    pdf_path = Path("AI_XAUUSD_Trading_White_Paper.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        print("Please ensure the white paper PDF exists.")
        return False

    try:
        # Prepare submission
        preparer = ArxivSubmissionPreparer()
        submission_info = preparer.prepare_submission()

        print("\n" + "=" * 40)
        print("✅ arXiv Submission Package Ready!")
        print("=" * 40)
        print(f"📦 Tarball: {submission_info['tarball']}")
        print(f"📄 PDF: {submission_info['pdf']}")
        print(f"📋 README: {submission_info['readme']}")
        print(f"📊 Metadata: {submission_info['metadata']}")
        print(f"💾 Size: {submission_info['size_mb']} MB")

        # Create submission script
        submit_script = create_arxiv_submission_script()
        print(f"🛠️  Submission script: {submit_script}")

        print("\n📋 arXiv Submission Instructions:")
        print("=" * 40)
        print("1. 📝 Go to: https://arxiv.org/submit")
        print("2. 🔐 Log in with your arXiv account")
        print("3. 📂 Select category: Quantitative Finance (q-fin.TR)")
        print("4. 📤 Upload the tarball file")
        print("5. 📝 Fill in title, authors, abstract (see metadata.json)")
        print("6. ✅ Submit and wait for moderation")

        print("\n⚠️  Important Notes:")
        print("• arXiv submissions require moderation")
        print("• First-time submitters may need account verification")
        print("• SWORD API requires special permissions (most users use web interface)")
        print("• Processing can take 1-3 days")

        print("\n🔗 Additional Resources:")
        print("• arXiv Help: https://arxiv.org/help/submit")
        print("• Author Guidelines: https://arxiv.org/help/author")
        print("• SWORD Documentation: https://arxiv.org/help/sword")

        return True

    except Exception as e:
        print(f"❌ Preparation failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)