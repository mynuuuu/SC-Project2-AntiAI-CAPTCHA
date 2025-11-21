"""
Computer Vision Utility Functions

Helper functions for image processing, feature detection, and pattern matching
used by the CV attacker.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import logging

logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess image for better feature detection
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        Preprocessed grayscale image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Enhance contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    return enhanced


def detect_rectangles(image: np.ndarray, min_area: int = 1000, 
                     max_area: int = 10000, aspect_ratio_range: Tuple[float, float] = (0.7, 1.3)) -> List[Tuple[int, int, int, int]]:
    """
    Detect rectangular regions in an image
    
    Args:
        image: Input grayscale image
        min_area: Minimum area of rectangles to detect
        max_area: Maximum area of rectangles to detect
        aspect_ratio_range: (min, max) aspect ratio for rectangles
        
    Returns:
        List of (x, y, w, h) bounding boxes
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rectangles = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        aspect_ratio = w / h if h > 0 else 0
        
        if min_area <= area <= max_area and aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]:
            rectangles.append((x, y, w, h))
    
    return rectangles


def template_matching(image: np.ndarray, template: np.ndarray, 
                     threshold: float = 0.7) -> List[Tuple[int, int, float]]:
    """
    Find template matches in image
    
    Args:
        image: Input image (grayscale)
        template: Template to search for (grayscale)
        threshold: Minimum match confidence (0-1)
        
    Returns:
        List of (x, y, confidence) matches
    """
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    locations = np.where(result >= threshold)
    
    matches = []
    for pt in zip(*locations[::-1]):
        x, y = pt
        confidence = result[y, x]
        matches.append((x, y, confidence))
    
    # Remove overlapping matches (non-maximum suppression)
    matches = non_max_suppression(matches, template.shape[1], template.shape[0])
    
    return matches


def non_max_suppression(matches: List[Tuple[int, int, float]], 
                       template_w: int, template_h: int, 
                       overlap_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Remove overlapping template matches (non-maximum suppression)
    
    Args:
        matches: List of (x, y, confidence) matches
        template_w: Template width
        template_h: Template height
        overlap_threshold: Overlap threshold for suppression
        
    Returns:
        Filtered list of matches
    """
    if not matches:
        return []
    
    # Sort by confidence (descending)
    matches = sorted(matches, key=lambda x: x[2], reverse=True)
    
    filtered = []
    for match in matches:
        x, y, conf = match
        
        # Check overlap with existing matches
        overlap = False
        for fx, fy, fconf in filtered:
            # Calculate intersection over union
            x1 = max(x, fx)
            y1 = max(y, fy)
            x2 = min(x + template_w, fx + template_w)
            y2 = min(y + template_h, fy + template_h)
            
            if x2 > x1 and y2 > y1:
                intersection = (x2 - x1) * (y2 - y1)
                union = template_w * template_h * 2 - intersection
                iou = intersection / union if union > 0 else 0
                
                if iou > overlap_threshold:
                    overlap = True
                    break
        
        if not overlap:
            filtered.append(match)
    
    return filtered


def detect_rotation_angle(image: np.ndarray) -> float:
    """
    Detect the rotation angle of an image using edge detection
    
    Args:
        image: Input grayscale image
        
    Returns:
        Rotation angle in degrees (0-360)
    """
    # Edge detection
    edges = cv2.Canny(image, 50, 150)
    
    # Hough line transform to detect dominant lines
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    # Calculate average angle
    angles = []
    for rho, theta in lines[:10]:  # Use first 10 lines
        angle = np.degrees(theta) - 90
        angles.append(angle)
    
    # Return median angle
    return np.median(angles)


def feature_matching(image1: np.ndarray, image2: np.ndarray, 
                    method: str = 'ORB') -> List[Tuple[int, int, int, int]]:
    """
    Match features between two images
    
    Args:
        image1: First image (grayscale)
        image2: Second image (grayscale)
        method: Feature detection method ('ORB', 'SIFT', 'AKAZE')
        
    Returns:
        List of matched keypoint pairs as (x1, y1, x2, y2)
    """
    if method == 'ORB':
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    elif method == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    elif method == 'AKAZE':
        detector = cv2.AKAZE_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Detect keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(image1, None)
    kp2, des2 = detector.detectAndCompute(image2, None)
    
    if des1 is None or des2 is None:
        return []
    
    # Match descriptors
    matches = matcher.match(des1, des2)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Extract matched points
    matched_pairs = []
    for match in matches[:20]:  # Use top 20 matches
        pt1 = kp1[match.queryIdx].pt
        pt2 = kp2[match.trainIdx].pt
        matched_pairs.append((int(pt1[0]), int(pt1[1]), int(pt2[0]), int(pt2[1])))
    
    return matched_pairs


def calculate_homography(points1: List[Tuple[int, int]], 
                        points2: List[Tuple[int, int]]) -> Optional[np.ndarray]:
    """
    Calculate homography matrix between two sets of points
    
    Args:
        points1: List of (x, y) points in first image
        points2: List of (x, y) points in second image
        
    Returns:
        Homography matrix (3x3) or None if calculation fails
    """
    if len(points1) < 4 or len(points2) < 4:
        return None
    
    pts1 = np.array(points1, dtype=np.float32)
    pts2 = np.array(points2, dtype=np.float32)
    
    try:
        H, mask = cv2.findHomography(pts1, pts2, cv2.RANSAC, 5.0)
        return H
    except:
        return None


def detect_color_regions(image: np.ndarray, color_range: Tuple[np.ndarray, np.ndarray]) -> List[Tuple[int, int, int, int]]:
    """
    Detect regions of specific color in image
    
    Args:
        image: Input image (BGR format)
        color_range: (lower_bound, upper_bound) in HSV format
        
    Returns:
        List of (x, y, w, h) bounding boxes
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower, upper = color_range
    
    # Create mask
    mask = cv2.inRange(hsv, lower, upper)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area > 500:  # Minimum area threshold
            regions.append((x, y, w, h))
    
    return regions


def extract_puzzle_piece(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extract a puzzle piece from image given bounding box
    
    Args:
        image: Input image
        bbox: (x, y, w, h) bounding box
        
    Returns:
        Extracted puzzle piece image
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def calculate_similarity(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    Calculate similarity between two images using structural similarity
    
    Args:
        image1: First image
        image2: Second image
        
    Returns:
        Similarity score (0-1)
    """
    # Resize images to same size if needed
    if image1.shape != image2.shape:
        h, w = min(image1.shape[0], image2.shape[0]), min(image1.shape[1], image2.shape[1])
        image1 = cv2.resize(image1, (w, h))
        image2 = cv2.resize(image2, (w, h))
    
    # Convert to grayscale if needed
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    # Calculate structural similarity
    # Using simple correlation coefficient as approximation
    image1_norm = image1.astype(np.float32) / 255.0
    image2_norm = image2.astype(np.float32) / 255.0
    
    correlation = cv2.matchTemplate(image1_norm, image2_norm, cv2.TM_CCOEFF_NORMED)[0, 0]
    
    return max(0.0, correlation)  # Ensure non-negative

