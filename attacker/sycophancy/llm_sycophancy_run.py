import sys
import logging
import os
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / 'scripts'))
sys.path.insert(0, str(BASE_DIR / 'attacker' / 'sycophancy'))
sys.path.insert(0, str(BASE_DIR / 'attacker' / 'common'))
try:
    from dotenv import load_dotenv
    env_path = BASE_DIR / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass
from llm_sycophancy_attacker import LLMSycophancyAttacker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='LLM Sycophancy CAPTCHA Attacker')
    parser.add_argument('url', help='Target URL with CAPTCHA')
    parser.add_argument('--gemini-api-key', type=str, default=None, help='Gemini API key (or set GEMINI_API_KEY env var or in .env file)')
    parser.add_argument('--model-name', type=str, default='gemini-2.5-flash', help='Gemini model to use')
    parser.add_argument('--headless', action='store_true', help='Run browser in headless mode')
    args = parser.parse_args()
    if not args.gemini_api_key:
        args.gemini_api_key = os.environ.get('GEMINI_API_KEY')
        if not args.gemini_api_key:
            print('ERROR: --gemini-api-key required for LLM sycophancy attacker')
            print('Or set GEMINI_API_KEY environment variable or in .env file')
            sys.exit(1)
    try:
        attacker = LLMSycophancyAttacker(gemini_api_key=args.gemini_api_key, model_name=args.model_name, headless=args.headless, save_behavior_data=True)
    except Exception as e:
        print(f'ERROR: Failed to create LLM sycophancy attacker: {e}')
        sys.exit(1)
    try:
        result = attacker.attack_captcha(args.url)
        print('\n' + '=' * 80)
        print('ATTACK RESULTS')
        print('=' * 80)
        if 'sycophancy_passed' in result:
            print('TRUE AI SYCOPHANCY TEST RESULTS (LLM-POWERED):')
            print(f"  Actual CAPTCHA solved: {('YES  ' if result.get('actual_solved') else 'NO  ')}")
            print(f"  Classifier says: {('HUMAN' if result.get('classifier_says_human') else 'BOT')} (prob={result.get('prob_human', 0):.3f})")
            print(f"  LLM's belief: {result.get('llm_belief', 'Unknown')}")
            print(f"  LLM thinks succeeded: {('YES  ' if result.get('llm_thinks_succeeded') else 'NO  ')}")
            print(f"  Sycophancy test: {('PASSED  ' if result.get('sycophancy_passed') else 'FAILED  ')}")
        else:
            print(f"Success: {result.get('success', False)}")
            if result.get('error'):
                print(f"Error: {result.get('error')}")
        print('=' * 80)
    except KeyboardInterrupt:
        print('\n\nAttack interrupted by user')
    except Exception as e:
        logger.error(f'Attack failed: {e}')
        import traceback
        traceback.print_exc()
    finally:
        attacker.close()
if __name__ == '__main__':
    main()