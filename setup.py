#!/usr/bin/env python3
"""
Setup script for AI Therapist Agent
This script helps you set up the environment and API keys.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version: {sys.version}")
    return True

def check_env_file():
    """Check if .env file exists and has API key."""
    env_file = Path('.env')
    if not env_file.exists():
        print("❌ .env file not found.")
        print("📝 Creating .env file from .env.example...")
        
        example_file = Path('.env.example')
        if example_file.exists():
            with open(example_file, 'r') as src, open(env_file, 'w') as dst:
                dst.write(src.read())
            print("✅ .env file created. Please edit it and add your TOGETHER_API_KEY.")
        else:
            print("❌ .env.example file not found. Please create a .env file manually.")
        return False
    
    # Check if API key is set
    with open(env_file, 'r') as f:
        content = f.read()
        if 'TOGETHER_API_KEY=your_actual_together_ai_api_key_here' in content:
            print("⚠️  Please update your TOGETHER_API_KEY in the .env file.")
            return False
        elif 'TOGETHER_API_KEY=' in content and len(content.split('TOGETHER_API_KEY=')[1].split('\n')[0].strip()) > 10:
            print("✅ API key appears to be set in .env file.")
            return True
        else:
            print("⚠️  TOGETHER_API_KEY not found or appears empty in .env file.")
            return False

def check_requirements():
    """Check if required packages are installed."""
    try:
        import together
        print("✅ together package is installed.")
    except ImportError:
        print("❌ together package not found.")
        print("📦 Install with: pip install -r requirements.txt")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv package is installed.")
    except ImportError:
        print("❌ python-dotenv package not found.")
        print("📦 Install with: pip install python-dotenv")
        return False
    
    return True

def main():
    """Main setup function."""
    print("🤖 AI Therapist Agent Setup")
    print("=" * 40)
    
    checks_passed = 0
    total_checks = 3
    
    if check_python_version():
        checks_passed += 1
    
    if check_requirements():
        checks_passed += 1
    
    if check_env_file():
        checks_passed += 1
    
    print("\n" + "=" * 40)
    print(f"Setup Status: {checks_passed}/{total_checks} checks passed")
    
    if checks_passed == total_checks:
        print("🎉 Setup complete! You can now run:")
        print("   python agent_v1.py")
    else:
        print("⚠️  Please fix the issues above before running the agent.")
        print("\n📝 Quick start guide:")
        print("1. Get API key from https://together.ai")
        print("2. Update TOGETHER_API_KEY in .env file")
        print("3. Install requirements: pip install -r requirements.txt")
        print("4. Run: python agent_v1.py")

if __name__ == "__main__":
    main()
