#!/usr/bin/env python3
"""
Build Documentation for AI-XAUUSD Trading System

This script generates PDF documentation from LaTeX sources and builds
HTML documentation using Sphinx.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import argparse

def run_command(cmd, cwd=None, check=True):
    """Run a shell command and return the result."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=cwd,
            capture_output=True,
            text=True,
            check=check
        )
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ Command failed: {cmd}")
        print(f"Error: {e}")
        return False

def check_latex_installation():
    """Check if LaTeX is installed and available."""
    print("🔍 Checking LaTeX installation...")

    # Check for common LaTeX distributions
    latex_commands = ['pdflatex', 'xelatex', 'lualatex']

    for cmd in latex_commands:
        if run_command(f"{cmd} --version", check=False):
            print(f"✅ Found {cmd}")
            return cmd

    print("❌ LaTeX not found. Please install a LaTeX distribution:")
    print("   - TeX Live: https://tug.org/texlive/")
    print("   - MiKTeX: https://miktex.org/")
    print("   - MacTeX (macOS): https://tug.org/mactex/")
    return None

def build_pdf_whitepaper(latex_cmd='pdflatex'):
    """Build PDF from LaTeX white paper."""
    print("📄 Building PDF white paper...")

    tex_file = "AI_XAUUSD_Trading_White_Paper.tex"

    if not Path(tex_file).exists():
        print(f"❌ LaTeX file not found: {tex_file}")
        return False

    # Create output directory
    output_dir = Path("docs/pdfs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build PDF (run multiple times for references)
    for i in range(2):
        print(f"🔄 LaTeX compilation pass {i+1}/2")
        cmd = f"{latex_cmd} -output-directory={output_dir} {tex_file}"
        if not run_command(cmd):
            return False

    # Move PDF to root directory
    pdf_file = output_dir / tex_file.replace('.tex', '.pdf')
    if pdf_file.exists():
        shutil.copy2(pdf_file, '.')
        print(f"✅ PDF generated: AI_XAUUSD_Trading_White_Paper.pdf")
        return True
    else:
        print("❌ PDF generation failed")
        return False

def check_sphinx_installation():
    """Check if Sphinx is installed."""
    print("🔍 Checking Sphinx installation...")

    try:
        import sphinx
        print(f"✅ Sphinx found (version {sphinx.__version__})")
        return True
    except ImportError:
        print("❌ Sphinx not found. Install with: pip install sphinx sphinx-rtd-theme")
        return False

def build_html_docs():
    """Build HTML documentation using Sphinx."""
    print("🌐 Building HTML documentation...")

    docs_dir = Path("docs")
    source_dir = docs_dir / "source"
    build_dir = docs_dir / "build"

    if not source_dir.exists():
        print("📝 Creating Sphinx documentation structure...")

        # Create source directory
        source_dir.mkdir(parents=True, exist_ok=True)

        # Create basic conf.py
        conf_py = source_dir / "conf.py"
        conf_py.write_text("""
# Configuration file for the Sphinx documentation builder.

project = 'AI-XAUUSD Trading System'
copyright = '2024, JonusNattapong / Zombitx64'
author = 'JonusNattapong / Zombitx64'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
""")

        # Create index.rst
        index_rst = source_dir / "index.rst"
        index_rst.write_text("""
AI-XAUUSD Trading System Documentation
=======================================

Welcome to the AI-XAUUSD Trading System documentation.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
""")

        # Create API documentation file
        api_rst = source_dir / "api.rst"
        api_rst.write_text("""
API Reference
=============

.. automodule:: ensemble_trader
   :members:

.. automodule:: market_regime_detector
   :members:

.. automodule:: trading_env
   :members:

.. automodule:: live_ensemble_trading
   :members:
""")

    # Build HTML docs
    cmd = f"sphinx-build -b html {source_dir} {build_dir / 'html'}"
    if run_command(cmd):
        print(f"✅ HTML docs built in {build_dir / 'html'}")
        return True
    else:
        print("❌ HTML documentation build failed")
        return False

def main():
    parser = argparse.ArgumentParser(description="Build documentation for AI-XAUUSD Trading System")
    parser.add_argument("--pdf-only", action="store_true", help="Build only PDF documentation")
    parser.add_argument("--html-only", action="store_true", help="Build only HTML documentation")
    parser.add_argument("--latex-cmd", default="pdflatex", help="LaTeX command to use (pdflatex, xelatex, lualatex)")

    args = parser.parse_args()

    print("🏗️  Building AI-XAUUSD Trading System Documentation")
    print("=" * 55)

    success = True

    # Build PDF documentation
    if not args.html_only:
        latex_cmd = check_latex_installation()
        if latex_cmd:
            if not build_pdf_whitepaper(latex_cmd):
                success = False
        else:
            success = False

    # Build HTML documentation
    if not args.pdf_only:
        if check_sphinx_installation():
            if not build_html_docs():
                success = False
        else:
            success = False

    if success:
        print("\n🎉 Documentation build completed successfully!")
        print("\nGenerated files:")
        if not args.html_only and Path("AI_XAUUSD_Trading_White_Paper.pdf").exists():
            print("  📄 AI_XAUUSD_Trading_White_Paper.pdf")
        if not args.pdf_only and Path("docs/build/html/index.html").exists():
            print("  🌐 docs/build/html/index.html")
    else:
        print("\n❌ Documentation build failed. Check errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()