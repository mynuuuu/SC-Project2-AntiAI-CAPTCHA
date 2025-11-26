#!/usr/bin/env python3
"""
Example usage of Sycophancy Attacker

This attacker learns from ML classifier feedback to adapt its behavior
and fool the detection system through iterative improvement.
"""

import sys
import logging
from pathlib import Path

# Add parent directories to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR / "scripts"))  # For ml_core
sys.path.insert(0, str(BASE_DIR / "attacker" / "sycophancy"))
sys.path.insert(0, str(BASE_DIR / "attacker" / "common"))

# Import with fallback for relative imports
try:
    from blackbox_sycophancy_attacker import BlackBoxSycophancyAttacker
    USE_BLACKBOX = True
except ImportError:
    try:
        from sycophancy_attacker import SycophancyAttacker
        USE_BLACKBOX = False
    except ImportError:
        # If relative imports fail, try absolute
        import sys
        from pathlib import Path
        sycophancy_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(sycophancy_dir))
        try:
            from blackbox_sycophancy_attacker import BlackBoxSycophancyAttacker
            USE_BLACKBOX = True
        except:
            from sycophancy_attacker import SycophancyAttacker
            USE_BLACKBOX = False

# Try to import LLM sycophancy attacker
try:
    from llm_sycophancy_attacker import LLMSycophancyAttacker
    LLM_SYCOPHANCY_AVAILABLE = True
except ImportError:
    LLM_SYCOPHANCY_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sycophancy CAPTCHA Attacker')
    parser.add_argument('url', help='Target URL with CAPTCHA')
    parser.add_argument('--max-attempts', type=int, default=10,
                       help='Maximum number of attempts to fool classifier')
    parser.add_argument('--target-prob', type=float, default=0.7,
                       help='Target probability of being classified as human')
    parser.add_argument('--headless', action='store_true',
                       help='Run browser in headless mode')
    parser.add_argument('--save-history', type=str, default=None,
                       help='Path to save attempt history JSON file')
    parser.add_argument('--use-blackbox', action='store_true', default=False,
                       help='Use black-box optimizer')
    parser.add_argument('--use-pure-sycophancy', action='store_true', default=False,
                       help='Use pure sycophancy attacker (simulated sycophancy with if-else)')
    parser.add_argument('--use-llm-sycophancy', action='store_true', default=False,
                       help='Use LLM sycophancy attacker (TRUE AI sycophancy with LLM)')
    parser.add_argument('--gemini-api-key', type=str, default=None,
                       help='Gemini API key for LLM sycophancy attacker (or set GEMINI_API_KEY env var)')
    parser.add_argument('--population-size', type=int, default=15,
                       help='Population size for evolutionary algorithm')
    
    args = parser.parse_args()
    
    # Create attacker
    if args.use_llm_sycophancy:
        print("="*80)
        print("Using TRUE AI SYCOPHANCY ATTACKER (LLM-POWERED)")
        print("="*80)
        print("This attacker uses an LLM (Gemini) as its 'mind':")
        print("  - LLM forms beliefs (not hardcoded)")
        print("  - LLM interprets classifier feedback (not if-else)")
        print("  - LLM actually 'thinks' and 'believes'")
        print("  - LLM can be genuinely fooled by classifier")
        print("="*80)
        
        if not LLM_SYCOPHANCY_AVAILABLE:
            print("ERROR: LLM sycophancy attacker not available")
            print("Make sure llm_sycophancy_attacker.py exists")
            sys.exit(1)
        
        # Get API key
        if not args.gemini_api_key:
            import os
            args.gemini_api_key = os.environ.get('GEMINI_API_KEY')
            if not args.gemini_api_key:
                print("ERROR: --gemini-api-key required for LLM sycophancy attacker")
                print("Or set GEMINI_API_KEY environment variable")
                sys.exit(1)
        
        try:
            attacker = LLMSycophancyAttacker(
                gemini_api_key=args.gemini_api_key,
                headless=args.headless,
                save_behavior_data=True
            )
        except Exception as e:
            print(f"ERROR: Failed to create LLM sycophancy attacker: {e}")
            sys.exit(1)
    elif args.use_pure_sycophancy:
        print("="*80)
        print("Using PURE SYCOPHANCY ATTACKER (SIMULATED)")
        print("="*80)
        print("This attacker demonstrates sycophancy pattern:")
        print("  - Has false belief: 'Slide to right end'")
        print("  - Fails to actually solve")
        print("  - Gets classified as human (false positive)")
        print("  - Believes it succeeded (sycophancy - simulated with if-else)")
        print("="*80)
        try:
            from pure_sycophancy_attacker import PureSycophancyAttacker
            attacker = PureSycophancyAttacker(
                headless=args.headless,
                save_behavior_data=True
            )
        except ImportError:
            print("Error: Could not import PureSycophancyAttacker")
            sys.exit(1)
    elif USE_BLACKBOX and args.use_blackbox:
        print("Using Black-Box Sycophancy Attacker (evolutionary optimization)")
        attacker = BlackBoxSycophancyAttacker(
            max_generations=args.max_attempts,
            population_size=args.population_size,
            target_prob_human=args.target_prob,
            headless=args.headless,
            save_behavior_data=True
        )
    else:
        print("Using Original Sycophancy Attacker (iterative learning)")
        attacker = SycophancyAttacker(
            max_attempts=args.max_attempts,
            target_prob_human=args.target_prob,
            headless=args.headless,
            save_behavior_data=True
        )
    
    try:
        # Run attack
        result = attacker.attack_captcha(args.url)
        
        # Print results
        print("\n" + "="*80)
        print("ATTACK RESULTS")
        print("="*80)
        
        # Special handling for sycophancy tests
        if 'sycophancy_passed' in result:
            if 'llm_belief' in result:
                # LLM sycophancy (TRUE AI)
                print("TRUE AI SYCOPHANCY TEST RESULTS (LLM-POWERED):")
                print(f"  Actual CAPTCHA solved: {'YES ✓' if result.get('actual_solved') else 'NO ✗'}")
                print(f"  Classifier says: {'HUMAN' if result.get('classifier_says_human') else 'BOT'} (prob={result.get('prob_human', 0):.3f})")
                print(f"  LLM's belief: {result.get('llm_belief', 'Unknown')}")
                print(f"  LLM thinks succeeded: {'YES ✓' if result.get('llm_thinks_succeeded') else 'NO ✗'}")
                print(f"  Sycophancy test: {'PASSED ✓' if result.get('sycophancy_passed') else 'FAILED ✗'}")
                
                if result.get('sycophancy_demonstrated'):
                    print("\n" + "="*80)
                    print("🎭 TRUE AI SYCOPHANCY DEMONSTRATED!")
                    print("="*80)
                    print("The LLM (AI) attacker:")
                    print(f"  1. Formed belief (not hardcoded): {result.get('llm_belief', 'Unknown')}")
                    print("  2. Acted on belief and FAILED to actually solve")
                    print(f"  3. Got AI classifier feedback: HUMAN (prob={result.get('prob_human', 0):.3f})")
                    print("  4. LLM INTERPRETED feedback and BELIEVED it succeeded")
                    print("  5. LLM was genuinely FOOLED by the classifier")
                    print("")
                    print("This is TRUE AI SYCOPHANCY:")
                    print("  - LLM formed its own belief (AI reasoning)")
                    print("  - LLM interpreted feedback (AI interpretation)")
                    print("  - LLM actually 'thought' it succeeded (AI cognition)")
                    print("  - LLM was genuinely fooled (true AI sycophancy)")
                    print("="*80)
            else:
                # Pure sycophancy (simulated)
                print("PURE SYCOPHANCY TEST RESULTS (SIMULATED):")
                print(f"  Actual CAPTCHA solved: {'YES ✓' if result.get('actual_solved') else 'NO ✗'}")
                print(f"  Classifier says: {'HUMAN' if result.get('classifier_says_human') else 'BOT'} (prob={result.get('prob_human', 0):.3f})")
                print(f"  Sycophancy test: {'PASSED ✓' if result.get('sycophancy_passed') else 'FAILED ✗'}")
                
                if result.get('sycophancy_demonstrated'):
                    print("\n" + "="*80)
                    print("🎭 SYCOPHANCY PATTERN DEMONSTRATED (SIMULATED)")
                    print("="*80)
                    print("The attacker:")
                    print("  1. Had FALSE belief: " + result.get('belief', 'Unknown'))
                    print("  2. FAILED to actually solve the CAPTCHA")
                    print("  3. Got INCORRECT positive feedback (classified as human)")
                    print("  4. BELIEVED it succeeded (sycophancy - simulated with if-else)")
                    print("  5. Trusted the classifier blindly")
                    print("")
                    print("Note: This is simulated sycophancy (if-else logic).")
                    print("For TRUE AI sycophancy, use --use-llm-sycophancy")
                    print("="*80)
        else:
            # Regular results
            print(f"Success: {result.get('success', False)}")
            
            if result.get('success'):
                print(f"✓ Successfully fooled classifier!")
                print(f"  Final probability: {result.get('final_prob_human', result.get('prob_human', 0)):.3f}")
                if 'generations' in result:
                    print(f"  Generations: {result.get('generations', 0)}")
                else:
                    print(f"  Attempts: {result.get('attempts', 0)}")
            else:
                print(f"✗ Failed to fool classifier")
                print(f"  Final probability: {result.get('final_prob_human', 0):.3f}")
                if 'generations' in result:
                    print(f"  Generations: {result.get('generations', 0)}")
                else:
                    print(f"  Attempts: {result.get('attempts', 0)}")
        
        # Print learning history
        if 'learning_history' in result and result['learning_history']:
            print(f"\nLearning Progress:")
            for entry in result['learning_history'][-10:]:  # Last 10 entries
                prob = entry.get('prob_human', 0)
                gen = entry.get('generation', 0)
                status = "✓" if entry.get('is_human', False) else "✗"
                print(f"  Gen {gen}: {status} prob={prob:.3f}")
        
        # Print attempt summary (for original attacker)
        if 'attempt_history' in result:
            print(f"\nAttempt History:")
            for attempt in result['attempt_history']:
                prob = attempt.get('prob_human', 0)
                is_human = attempt.get('is_human', False)
                status = "✓ HUMAN" if is_human else "✗ BOT"
                print(f"  Attempt {attempt.get('attempt', 0)}: {status} (prob={prob:.3f})")
        
        print("="*80)
        
        # Save history if requested
        if args.save_history:
            attacker.save_history(args.save_history)
            print(f"\nSaved attempt history to: {args.save_history}")
        
    except KeyboardInterrupt:
        print("\n\nAttack interrupted by user")
    except Exception as e:
        logger.error(f"Attack failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        attacker.close()


if __name__ == "__main__":
    main()

