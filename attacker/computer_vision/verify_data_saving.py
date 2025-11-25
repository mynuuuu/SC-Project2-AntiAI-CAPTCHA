#!/usr/bin/env python3
"""
Verification script to check bot data saving setup
"""

import sys
from pathlib import Path

# Add parent directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"

def check_data_directory():
    """Check if data directory exists and is writable"""
    print("=" * 60)
    print("CHECKING DATA DIRECTORY")
    print("=" * 60)
    
    if DATA_DIR.exists():
        print(f"✓ Data directory exists: {DATA_DIR}")
    else:
        print(f"✗ Data directory does NOT exist: {DATA_DIR}")
        print(f"  Creating it...")
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {DATA_DIR}")
    
    # Check if writable
    test_file = DATA_DIR / ".test_write"
    try:
        test_file.write_text("test")
        test_file.unlink()
        print(f"✓ Data directory is writable")
    except Exception as e:
        print(f"✗ Data directory is NOT writable: {e}")
        return False
    
    return True

def check_existing_files():
    """Check for existing bot data files"""
    print("\n" + "=" * 60)
    print("CHECKING EXISTING BOT DATA FILES")
    print("=" * 60)
    
    bot_files = ['bot_captcha1.csv', 'bot_captcha2.csv', 'bot_captcha3.csv']
    
    for filename in bot_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            # Get file size and line count
            size = filepath.stat().st_size
            try:
                lines = len(filepath.read_text().splitlines())
            except:
                lines = "?"
            print(f"✓ {filename}: exists ({size} bytes, ~{lines} lines)")
        else:
            print(f"○ {filename}: does not exist (will be created on first run)")

def check_human_data_files():
    """Check for existing human data files"""
    print("\n" + "=" * 60)
    print("CHECKING HUMAN DATA FILES (for comparison)")
    print("=" * 60)
    
    human_files = ['captcha1.csv', 'captcha2.csv', 'captcha3.csv', 
                   'rotation_layer.csv', 'layer3_question.csv']
    
    found_any = False
    for filename in human_files:
        filepath = DATA_DIR / filename
        if filepath.exists():
            size = filepath.stat().st_size
            try:
                lines = len(filepath.read_text().splitlines())
            except:
                lines = "?"
            print(f"✓ {filename}: exists ({size} bytes, ~{lines} lines)")
            found_any = True
    
    if not found_any:
        print("○ No human data files found")
        print("  Run 'npm start' and manually solve CAPTCHAs to collect human data")

def print_next_steps():
    """Print next steps for user"""
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    
    print("\n1. Collect Bot Data:")
    print("   cd attacker/computer_vision")
    print("   python3 example_attack.py http://localhost:3000 --batch 10")
    
    print("\n2. Verify Bot Data:")
    print("   ls -lh ../../data/bot_*.csv")
    print("   head -n 3 ../../data/bot_captcha1.csv")
    
    print("\n3. Train Supervised Models:")
    print("   cd ../../scripts")
    print("   python3 train_supervised_model.py \\")
    print("       --layer slider_layer1 \\")
    print("       --human-file captcha1.csv \\")
    print("       --bot-file bot_captcha1.csv")
    
    print("\n4. Test Accuracy:")
    print("   python3 server.py")
    print("   # Test with real humans and bot attacks")
    
    print("\n" + "=" * 60)

def main():
    print("\n" + "=" * 60)
    print("BOT DATA SAVING VERIFICATION")
    print("=" * 60)
    print()
    
    # Run checks
    dir_ok = check_data_directory()
    if not dir_ok:
        print("\n✗ Data directory check failed!")
        sys.exit(1)
    
    check_existing_files()
    check_human_data_files()
    print_next_steps()
    
    print("\n✓ All checks passed! Ready to collect bot data.")
    print()

if __name__ == "__main__":
    main()

