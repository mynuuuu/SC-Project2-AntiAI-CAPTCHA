"""
Test Script: Verify Your Attacker Setup
"""

import os
import sys

def test_imports():
    """Test if all required libraries are installed"""
    print("=" * 60)
    print("Testing Imports...")
    print("=" * 60)
    
    try:
        from selenium import webdriver
        print("✅ selenium installed")
    except ImportError:
        print("❌ selenium NOT installed - run: pip install selenium")
        return False
    
    try:
        from openai import OpenAI
        print("✅ openai installed")
    except ImportError:
        print("❌ openai NOT installed - run: pip install openai")
        return False
    
    try:
        from anthropic import Anthropic
        print("✅ anthropic installed")
    except ImportError:
        print("❌ anthropic NOT installed - run: pip install anthropic")
        return False
    
    try:
        from PIL import Image
        print("✅ Pillow installed")
    except ImportError:
        print("❌ Pillow NOT installed - run: pip install Pillow")
        return False
    
    return True

def test_api_key():
    """Test if API key is set"""
    print("\n" + "=" * 60)
    print("Testing API Key...")
    print("=" * 60)
    
    api_key = os.getenv('OPEN_API_KEY')
    
    if not api_key:
        print("❌ OPENAI_API_KEY not set!")
        print("\nTo fix:")
        print("  export OPENAI_API_KEY='sk-proj-your-key-here'")
        return False
    
    if not api_key.startswith('sk-'):
        print("⚠️  API key doesn't start with 'sk-' - might be invalid")
        return False
    
    print(f"✅ API key set: {api_key[:20]}...")
    return True

def test_openai_connection():
    """Test actual OpenAI API connection"""
    print("\n" + "=" * 60)
    print("Testing OpenAI Connection...")
    print("=" * 60)
    
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv('OPEN_API_KEY'))
        
        print("Sending test request to OpenAI...")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": "Say 'API test successful!'"}],
            max_tokens=20
        )
        
        result = response.choices[0].message.content
        print(f"✅ OpenAI response: {result}")
        return True
        
    except Exception as e:
        print(f"❌ OpenAI connection failed: {e}")
        return False

def test_chromedriver():
    """Test if ChromeDriver is available"""
    print("\n" + "=" * 60)
    print("Testing ChromeDriver...")
    print("=" * 60)
    
    try:
        from selenium import webdriver
        
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        
        driver = webdriver.Chrome(options=options)
        driver.get("https://www.google.com")
        print(f"✅ ChromeDriver working - Page title: {driver.title}")
        driver.quit()
        return True
        
    except Exception as e:
        print(f"❌ ChromeDriver failed: {e}")
        print("\nTo fix:")
        print("  brew install chromedriver")
        print("  or download from: https://chromedriver.chromium.org/")
        return False

def main():
    """Run all tests"""
    print("\n🧪 CAPTCHA ATTACKER SETUP TEST")
    print("=" * 60)
    
    results = {
        "Imports": test_imports(),
        "API Key": test_api_key(),
        "OpenAI Connection": test_openai_connection(),
        "ChromeDriver": test_chromedriver()
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:.<30} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n🎉 ALL TESTS PASSED! You're ready to attack CAPTCHAs!")
        print("\nNext steps:")
        print("1. Start your CAPTCHA: npm start")
        print("2. Run attacker: python adaptive_llm_attacker.py http://localhost:3000")
    else:
        print("\n⚠️  Some tests failed. Fix the issues above and try again.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
