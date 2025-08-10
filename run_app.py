#!/usr/bin/env python3
"""
FinDocGPT - AI-Powered Financial Document Analysis & Investment Strategy
Run script for the Streamlit application
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import pandas
        import plotly
        from openai import OpenAI
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install dependencies with: pip install -r requirements.txt")
        return False

def check_env_file():
    """Check if .env file exists and has OpenRouter API key"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️  .env file not found")
        print("Creating .env file with placeholder...")
        with open(".env", "w") as f:
            f.write("# FinDocGPT Environment Variables\n")
            f.write("# Add your OpenRouter API key here\n")
            f.write("OPEN_ROUTER_API_KEY=your_openrouter_api_key_here\n")
        print("📝 Please edit .env file and add your OpenRouter API key")
        return False
    
    with open(".env", "r") as f:
        content = f.read()
        if "your_openrouter_api_key_here" in content:
            print("⚠️  Please add your OpenRouter API key to .env file")
            return False
    
    print("✅ Environment variables configured")
    return True

def check_data_files():
    """Check if FinanceBench data files exist"""
    data_files = [
        "data/financebench_open_source.jsonl",
        "data/financebench_document_information.jsonl"
    ]
    
    missing_files = []
    for file_path in data_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing data files: {missing_files}")
        print("Please ensure FinanceBench data is available")
        return False
    
    print("✅ FinanceBench data files found")
    return True

def main():
    """Main function to run the FinDocGPT application"""
    print("🚀 FinDocGPT - AI-Powered Financial Analysis")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check environment
    if not check_env_file():
        print("⚠️  Application will run with limited functionality")
        print("   Add OpenRouter API key to .env for full AI features")
    
    # Check data files
    if not check_data_files():
        print("⚠️  Application will run with limited data")
        print("   Ensure FinanceBench data is available for full features")
    
    print("\n🎯 Starting FinDocGPT application...")
    print("📊 Open your browser to http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the application")
    print("=" * 50)
    
    try:
        # Run Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "main.py",
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 FinDocGPT application stopped")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    main() 