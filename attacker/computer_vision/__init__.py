"""
Computer Vision CAPTCHA Attacker Package

A generic computer vision-based attacker for pictorial CAPTCHAs.
"""

from .cv_attacker import CVAttacker, PuzzleType
from .cv_utils import (
    preprocess_image,
    detect_rectangles,
    template_matching,
    detect_rotation_angle,
    feature_matching,
    calculate_similarity
)

__all__ = [
    'CVAttacker',
    'PuzzleType',
    'preprocess_image',
    'detect_rectangles',
    'template_matching',
    'detect_rotation_angle',
    'feature_matching',
    'calculate_similarity'
]

