import cv2
import numpy as np
import sys
from pathlib import Path
from scipy.stats import pearsonr
from scipy.spatial.distance import cosine

def compute_saliency_similarity(image_a_path: str, image_b_path: str) -> dict:
    """
    Computes multiple similarity metrics between two saliency maps.
    
    Args:
        image_a_path: Path to the first saliency map (reference)
        image_b_path: Path to the second saliency map (comparison)
    
    Returns:
        Dictionary with various similarity metrics
    """
    # Load images in grayscale
    img_a = cv2.imread(image_a_path, cv2.IMREAD_GRAYSCALE)
    img_b = cv2.imread(image_b_path, cv2.IMREAD_GRAYSCALE)
    
    if img_a is None:
        raise FileNotFoundError(f"Could not load image: {image_a_path}")
    if img_b is None:
        raise FileNotFoundError(f"Could not load image: {image_b_path}")
    
    # Ensure images have the same shape
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]))
    
    # Normalize to [0, 1] range
    img_a_norm = img_a.astype(np.float32) / 255.0
    img_b_norm = img_b.astype(np.float32) / 255.0
    
    # Flatten for correlation calculations
    flat_a = img_a_norm.flatten()
    flat_b = img_b_norm.flatten()
    
    # 1. Pearson Correlation Coefficient (measures linear relationship)
    correlation, _ = pearsonr(flat_a, flat_b)
    
    # 2. Cosine Similarity (measures angular similarity)
    cosine_sim = 1 - cosine(flat_a, flat_b)
    
    # 3. Structural Similarity Index (SSIM) - considers structure
    # Using a simple implementation
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    mu_a = np.mean(img_a)
    mu_b = np.mean(img_b)
    sigma_a = np.std(img_a)
    sigma_b = np.std(img_b)
    sigma_ab = np.mean((img_a - mu_a) * (img_b - mu_b))
    
    ssim = ((2 * mu_a * mu_b + c1) * (2 * sigma_ab + c2)) / \
           ((mu_a**2 + mu_b**2 + c1) * (sigma_a**2 + sigma_b**2 + c2))
    
    # 4. Mean Absolute Error (original metric, but as error not similarity)
    mae = np.mean(np.abs(img_a_norm - img_b_norm))
    pixel_similarity = (1 - mae) * 100
    
    # 5. Intersection over Union for salient regions (threshold-based)
    threshold = 0.5
    binary_a = (img_a_norm > threshold).astype(np.float32)
    binary_b = (img_b_norm > threshold).astype(np.float32)
    intersection = np.sum(binary_a * binary_b)
    union = np.sum(np.maximum(binary_a, binary_b))
    iou = intersection / union if union > 0 else 0
    
    return {
        'pixel_similarity': pixel_similarity,
        'pearson_correlation': correlation * 100,
        'cosine_similarity': cosine_sim * 100,
        'ssim': ssim * 100,
        'iou_salient_regions': iou * 100
    }


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python pixel_wise_difference.py <image_a_path> <image_b_path>")
        sys.exit(1)
    
    image_a = sys.argv[1]
    image_b = sys.argv[2]
    
    try:
        metrics = compute_saliency_similarity(image_a, image_b)
        print("Saliency Map Similarity Metrics:")
        print(f"  Pixel-wise similarity: {metrics['pixel_similarity']:.2f}%")
        print(f"  Pearson correlation: {metrics['pearson_correlation']:.2f}%")
        print(f"  Cosine similarity: {metrics['cosine_similarity']:.2f}%")
        print(f"  SSIM: {metrics['ssim']:.2f}%")
        print(f"  IoU (salient regions): {metrics['iou_salient_regions']:.2f}%")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)