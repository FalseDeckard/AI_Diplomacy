#!/usr/bin/env python3
"""
Test script to verify .env file is being loaded correctly
"""

from config import config
import os
from pathlib import Path

def test_env_loading():
    print("🔍 Testing .env file loading...")
    print("=" * 50)
    
    # Check if .env file exists
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file found at: {env_file.resolve()}")
        
        # Read and show (first few chars of) each API key
        print("\n📋 API Keys from config:")
        api_keys = [
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY", 
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "OPENROUTER_API_KEY",
            "TOGETHER_API_KEY"
        ]
        
        for key in api_keys:
            try:
                value = getattr(config, key)
                if value and len(value) > 0:
                    print(f"✅ {key}: {value[:8]}..." if len(value) > 8 else f"✅ {key}: [short key]")
                else:
                    print(f"❌ {key}: Not set or empty")
            except Exception as e:
                print(f"❌ {key}: Error - {e}")
        
        print("\n📋 API Keys from environment variables:")
        for key in api_keys:
            env_value = os.environ.get(key)
            if env_value:
                print(f"✅ {key}: {env_value[:8]}..." if len(env_value) > 8 else f"✅ {key}: [short key]")
            else:
                print(f"❌ {key}: Not in environment")
                
    else:
        print(f"❌ .env file not found at: {env_file.resolve()}")
        print("💡 Create a .env file in the project root with your API keys")
        
    print("\n📝 Expected .env file format:")
    print("""
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here  
GEMINI_API_KEY=your-gemini-key-here
DEEPSEEK_API_KEY=your-deepseek-key-here
OPENROUTER_API_KEY=sk-or-your-openrouter-key-here
TOGETHER_API_KEY=your-together-api-key-here
""")

if __name__ == "__main__":
    test_env_loading()

