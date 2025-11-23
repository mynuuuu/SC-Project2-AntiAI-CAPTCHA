"""
Example script demonstrating how to use the CV attacker

This script shows how to attack a CAPTCHA system using the computer vision attacker.
"""

import sys
import time
from cv_attacker import CVAttacker, PuzzleType

def attack_single_captcha(url: str, headless: bool = False):
    """
    Attack a single CAPTCHA on a webpage
    
    Args:
        url: URL of the page containing the CAPTCHA
        headless: Run browser in headless mode
    """
    attacker = CVAttacker(headless=headless, use_model_classification=True)
    
    try:
        print(f"\n{'='*60}")
        print(f"Attacking CAPTCHA at: {url}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        result = attacker.attack_captcha(url)
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("ATTACK RESULTS")
        print("="*60)
        print(f"Overall Success: {'✓ YES' if result['success'] else '✗ NO'}")
        print(f"Puzzle Type: {result['puzzle_type']}")
        print(f"Attempts: {result['attempts']}")
        print(f"Time Elapsed: {elapsed_time:.2f} seconds")
        
        if result.get('slider_result'):
            print("\n" + "-"*60)
            print("SLIDER PUZZLE RESULTS")
            print("-"*60)
            print(f"Solved: {'✓ YES' if result['slider_result']['success'] else '✗ NO'}")
            if result['slider_result'].get('model_classification'):
                sc = result['slider_result']['model_classification']
                print(f"ML Classification: {sc['decision'].upper()}")
                print(f"Human Probability: {sc['prob_human']:.3f}")
                print(f"Events Captured: {sc['num_events']}")
                print(f"Would be accepted: {'✓ YES' if sc['is_human'] else '✗ NO (BOT DETECTED)'}")
            if result['slider_result'].get('error'):
                print(f"Error: {result['slider_result']['error']}")
        
        if result.get('rotation_result'):
            print("\n" + "-"*60)
            print("ROTATION PUZZLE RESULTS")
            print("-"*60)
            print(f"Solved: {'✓ YES' if result['rotation_result']['success'] else '✗ NO'}")
            if result['rotation_result'].get('model_classification'):
                rc = result['rotation_result']['model_classification']
                print(f"ML Classification: {rc['decision'].upper()}")
                print(f"Human Probability: {rc['prob_human']:.3f}")
                print(f"Events Captured: {rc['num_events']}")
                print(f"Would be accepted: {'✓ YES' if rc['is_human'] else '✗ NO (BOT DETECTED)'}")
            if result['rotation_result'].get('error'):
                print(f"Error: {result['rotation_result']['error']}")
        
        if result.get('model_classification'):
            classification = result['model_classification']
            print("\n" + "-"*60)
            print("OVERALL ML MODEL CLASSIFICATION")
            print("-"*60)
            print(f"Decision: {classification['decision'].upper()}")
            print(f"Human Probability: {classification['prob_human']:.3f}")
            print(f"Events Captured: {classification['num_events']}")
            print(f"Would be accepted: {'✓ YES' if classification['is_human'] else '✗ NO (BOT DETECTED)'}")
        
        if result.get('error'):
            print(f"\nError: {result['error']}")
        print("="*60 + "\n")
        
        return result['success']
        
    except Exception as e:
        print(f"\nError during attack: {e}")
        return False
    finally:
        attacker.close()


def attack_multiple_captchas(url: str, num_attempts: int = 5, headless: bool = False):
    """
    Attack multiple CAPTCHAs to test success rate
    
    Args:
        url: URL of the page containing the CAPTCHA
        num_attempts: Number of attack attempts
        headless: Run browser in headless mode
    """
    print(f"\n{'='*60}")
    print(f"Running {num_attempts} attack attempts")
    print(f"{'='*60}\n")
    
    successes = 0
    total_time = 0
    
    for i in range(num_attempts):
        print(f"\nAttempt {i+1}/{num_attempts}")
        print("-" * 60)
        
        attacker = CVAttacker(headless=headless, use_model_classification=True)
        
        try:
            start_time = time.time()
            result = attacker.attack_captcha(url)
            elapsed_time = time.time() - start_time
            total_time += elapsed_time
            
            if result['success']:
                successes += 1
                print(f"✓ Success (Time: {elapsed_time:.2f}s)")
            else:
                print(f"✗ Failed (Time: {elapsed_time:.2f}s)")
                if result['error']:
                    print(f"  Error: {result['error']}")
            
            # Wait between attempts
            if i < num_attempts - 1:
                time.sleep(2)
                
        except Exception as e:
            print(f"✗ Error: {e}")
        finally:
            attacker.close()
    
    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total Attempts: {num_attempts}")
    print(f"Successful: {successes}")
    print(f"Failed: {num_attempts - successes}")
    print(f"Success Rate: {(successes/num_attempts)*100:.1f}%")
    print(f"Average Time: {total_time/num_attempts:.2f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    # Default URL - adjust to your CAPTCHA system
    default_url = "http://localhost:3000"
    
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = default_url
    
    # Check if running in batch mode
    if len(sys.argv) > 2 and sys.argv[2] == "--batch":
        num_attempts = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        attack_multiple_captchas(url, num_attempts, headless=False)
    else:
        attack_single_captcha(url, headless=False)

