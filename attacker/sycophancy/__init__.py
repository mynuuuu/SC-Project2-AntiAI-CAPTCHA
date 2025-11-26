"""
Sycophancy Attacker Module

An attacker that learns from ML classifier feedback to adapt its behavior
and fool the detection system through iterative improvement.
"""

from .sycophancy_attacker import SycophancyAttacker
from .behavior_optimizer import BehaviorOptimizer
from .feature_analyzer import FeatureAnalyzer
from .blackbox_sycophancy_attacker import BlackBoxSycophancyAttacker
from .blackbox_optimizer import EvolutionaryOptimizer, LocalRefiner, OptimizedParams

# Handle pure sycophancy import
try:
    from .pure_sycophancy_attacker import PureSycophancyAttacker
except ImportError:
    try:
        from pure_sycophancy_attacker import PureSycophancyAttacker
    except ImportError:
        PureSycophancyAttacker = None

# Handle LLM sycophancy import
try:
    from .llm_sycophancy_attacker import LLMSycophancyAttacker
except ImportError:
    try:
        from llm_sycophancy_attacker import LLMSycophancyAttacker
    except ImportError:
        LLMSycophancyAttacker = None

__all__ = [
    'SycophancyAttacker', 
    'BehaviorOptimizer', 
    'FeatureAnalyzer',
    'BlackBoxSycophancyAttacker',
    'EvolutionaryOptimizer',
    'LocalRefiner',
    'OptimizedParams',
    'PureSycophancyAttacker',
    'LLMSycophancyAttacker'
]

