"""
Helper script to check browser version and ChromeDriver compatibility

This script helps you find the correct ChromeDriver version for your browser.
Works with Chrome, Arc, Edge, and other Chromium-based browsers.
"""

import subprocess
import sys
import platform
import os

def get_chrome_version_mac():
    """Get Chrome/Chromium version on macOS"""
    try:
        # Try Chrome first
        result = subprocess.run(
            ['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return version, 'Chrome'
    except:
        pass
    
    try:
        # Try Arc
        result = subprocess.run(
            ['/Applications/Arc.app/Contents/MacOS/Arc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return version, 'Arc'
    except:
        pass
    
    try:
        # Try Edge
        result = subprocess.run(
            ['/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return version, 'Edge'
    except:
        pass
    
    return None, None


def get_chrome_version_linux():
    """Get Chrome/Chromium version on Linux"""
    try:
        result = subprocess.run(
            ['google-chrome', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return version, 'Chrome'
    except:
        pass
    
    try:
        result = subprocess.run(
            ['chromium-browser', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[-1]
            return version, 'Chromium'
    except:
        pass
    
    return None, None


def get_chrome_version_windows():
    """Get Chrome version on Windows"""
    try:
        import winreg
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Google\Chrome\BLBeacon"
        )
        version, _ = winreg.QueryValueEx(key, "version")
        winreg.CloseKey(key)
        return version, 'Chrome'
    except:
        pass
    
    return None, None


def get_major_version(full_version):
    """Extract major version number from full version string"""
    try:
        return full_version.split('.')[0]
    except:
        return None


def check_chromedriver_installed():
    """Check if ChromeDriver is installed and get its version"""
    try:
        result = subprocess.run(
            ['chromedriver', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            version = result.stdout.strip().split()[1]
            return version
    except:
        pass
    
    return None


def main():
    print("=" * 60)
    print("Browser & ChromeDriver Version Checker")
    print("=" * 60)
    print()
    
    # Get browser version
    system = platform.system()
    print(f"Operating System: {system}")
    print()
    
    if system == "Darwin":  # macOS
        version, browser_name = get_chrome_version_mac()
    elif system == "Linux":
        version, browser_name = get_chrome_version_linux()
    elif system == "Windows":
        version, browser_name = get_chrome_version_windows()
    else:
        print("❌ Unsupported operating system")
        sys.exit(1)
    
    if version:
        major_version = get_major_version(version)
        print(f"✓ Found {browser_name} browser")
        print(f"  Full version: {version}")
        print(f"  Major version: {major_version}")
        print()
        
        # Check ChromeDriver
        chromedriver_version = check_chromedriver_installed()
        if chromedriver_version:
            chromedriver_major = get_major_version(chromedriver_version)
            print(f"✓ ChromeDriver is installed")
            print(f"  Version: {chromedriver_version}")
            print(f"  Major version: {chromedriver_major}")
            print()
            
            if major_version == chromedriver_major:
                print("✅ Versions match! You're good to go.")
            else:
                print("⚠️  Version mismatch detected!")
                print(f"   Browser major version: {major_version}")
                print(f"   ChromeDriver major version: {chromedriver_major}")
                print()
                print("   You need to download ChromeDriver version", major_version)
        else:
            print("❌ ChromeDriver is not installed or not in PATH")
            print()
            print("📥 Download Instructions:")
            print(f"   1. Go to: https://googlechromelabs.github.io/chrome-for-testing/")
            print(f"   2. Download ChromeDriver for Chrome version {major_version}")
            print(f"   3. Extract and add to PATH, or place in project directory")
            print()
            print("   Or use Homebrew (macOS):")
            print(f"   brew install chromedriver")
            print()
            print("   Or use npm:")
            print(f"   npm install -g chromedriver")
    else:
        print("❌ Could not detect browser version")
        print()
        print("Manual check:")
        if system == "Darwin":
            print("  Arc: /Applications/Arc.app/Contents/MacOS/Arc --version")
            print("  Chrome: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome --version")
        elif system == "Linux":
            print("  google-chrome --version")
            print("  chromium-browser --version")
        elif system == "Windows":
            print("  Check Chrome version in: chrome://version/")
    
    print()
    print("=" * 60)
    print("ChromeDriver Download Links:")
    print("=" * 60)
    print("• Chrome for Testing (Recommended):")
    print("  https://googlechromelabs.github.io/chrome-for-testing/")
    print()
    print("• Legacy ChromeDriver (if needed):")
    print("  https://chromedriver.chromium.org/downloads")
    print()
    print("• Homebrew (macOS):")
    print("  brew install chromedriver")
    print()
    print("• npm (cross-platform):")
    print("  npm install -g chromedriver")
    print("=" * 60)


if __name__ == "__main__":
    main()

