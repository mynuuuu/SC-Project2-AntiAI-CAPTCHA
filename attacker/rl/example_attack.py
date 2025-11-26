#!/usr/bin/env python3
"""
Example usage of RL CAPTCHA Attacker

Demonstrates how to use the RL attacker to learn and solve CAPTCHAs.
"""

import sys
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from attacker.rl.rl_attacker import RLAttacker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Example RL attack"""
    
    print("="*80)
    print("RL CAPTCHA Attacker - Example")
    print("="*80)
    print()
    
    # Configuration
    url = "http://localhost:3000"
    captcha_type = "slider"  # or "rotation"
    episodes = 20
    
    print(f"Target URL: {url}")
    print(f"CAPTCHA Type: {captcha_type}")
    print(f"Training Episodes: {episodes}")
    print()
    
    # Initialize attacker
    print("Initializing RL Attacker...")
    attacker = RLAttacker(
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=0.3,  # Start with high exploration
        epsilon_decay=0.995,
        min_epsilon=0.01,
        use_dqn=False,  # Use Q-learning (faster)
        save_policy_dir="models/rl_policies",
        use_model_classification=True,
        save_behavior_data=True,
        headless=False  # Set to True to run without browser window
    )
    
    try:
        # Train on CAPTCHA
        print("\nStarting training...")
        results = attacker.attack_captcha(
            url=url,
            captcha_type=captcha_type,
            episodes=episodes
        )
        
        # Print results
        print("\n" + "="*80)
        print("Training Results:")
        print("="*80)
        print(f"  Success: {results['success']}")
        print(f"  Successful Episodes: {results['successful_episodes']}/{results['episodes']}")
        print(f"  Success Rate: {results['success_rate']:.2%}")
        print(f"  Average Reward: {results['average_reward']:.2f}")
        print(f"  Average Steps: {results['average_steps']:.1f}")
        print(f"  Final Epsilon: {results['final_epsilon']:.3f}")
        print("="*80)
        
        # Agent statistics
        agent_stats = attacker.agent.get_stats()
        print("\nAgent Statistics:")
        print(f"  Number of States: {agent_stats.get('num_states', 'N/A')}")
        print(f"  Final Epsilon: {agent_stats.get('epsilon', 'N/A')}")
        print()
        
        # Example: Load and use saved policy
        print("Example: Loading saved policy for future use...")
        policy_path = f"models/rl_policies/final_policy_{captcha_type}.pkl"
        if Path(policy_path).exists():
            print(f"  Policy saved at: {policy_path}")
            print("  You can load it next time with:")
            print(f"    attacker = RLAttacker(load_policy='{policy_path}')")
        print()
        
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Close browser
        print("Closing browser...")
        attacker.close()
        print("Done!")


if __name__ == "__main__":
    main()

