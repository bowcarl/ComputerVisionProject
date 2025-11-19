"""
Evaluation metrics for NeRF: PSNR, SSIM, and LPIPS
"""

import torch
import torch.nn.functional as F
import numpy as np
from skimage.metrics import structural_similarity as ssim
import lpips


def compute_psnr_img(pred, target):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        pred: Predicted image tensor [H, W, 3] or [B, H, W, 3], values in [0, 1]
        target: Target image tensor [H, W, 3] or [B, H, W, 3], values in [0, 1]
    
    Returns:
        PSNR value in dB (higher is better)
    """
    mse = F.mse_loss(pred, target)
    
    if mse < 1e-10:  # Essentially perfect match
        return 100.0  # Return a high but finite value
    
    psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(mse))
    return psnr.item()


def compute_ssim_img(pred, target):
    """
    Compute Structural Similarity Index between two images.
    
    Args:
        pred: Predicted image tensor [H, W, 3], values in [0, 1]
        target: Target image tensor [H, W, 3], values in [0, 1]
    
    Returns:
        SSIM value in [0, 1] (higher is better)
    """
    # Convert to numpy and ensure correct shape
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    
    # Ensure values are in [0, 1]
    pred_np = np.clip(pred_np, 0, 1)
    target_np = np.clip(target_np, 0, 1)
    
    # Compute SSIM per channel and average
    ssim_vals = []
    for c in range(3):  # RGB channels
        ssim_val = ssim(
            pred_np[..., c], 
            target_np[..., c], 
            data_range=1.0,
            gaussian_weights=True,  # Use gaussian weighting
            sigma=1.5,              # Standard deviation for Gaussian kernel
            use_sample_covariance=False
        )
        ssim_vals.append(ssim_val)
    
    return float(np.mean(ssim_vals))


def compute_lpips_img(pred, target, lpips_model, device='cuda'):
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity) between two images.
    Lower values indicate more perceptually similar images.
    
    Args:
        pred: Predicted image tensor [H, W, 3], values in [0, 1]
        target: Target image tensor [H, W, 3], values in [0, 1]
        lpips_model: Pre-initialized LPIPS model
        device: Device to run computation on
    
    Returns:
        LPIPS distance (lower is better, typically in [0, 1])
    """
    # LPIPS expects [B, 3, H, W] in range [-1, 1]
    pred_img = pred.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
    target_img = target.permute(2, 0, 1).unsqueeze(0)
    
    # Normalize from [0, 1] to [-1, 1]
    pred_img = pred_img * 2 - 1
    target_img = target_img * 2 - 1
    
    # Move to device and compute
    with torch.no_grad():
        dist = lpips_model(pred_img.to(device), target_img.to(device))
    
    return float(dist.item())


def compute_all_metrics(pred, target, lpips_model=None, device='cuda'):
    """
    Compute all metrics at once for convenience.
    
    Args:
        pred: Predicted image [H, W, 3] in [0, 1]
        target: Target image [H, W, 3] in [0, 1]
        lpips_model: Optional pre-initialized LPIPS model
        device: Device for computation
    
    Returns:
        Dictionary with 'psnr', 'ssim', and optionally 'lpips'
    """
    metrics = {
        'psnr': compute_psnr_img(pred, target),
        'ssim': compute_ssim_img(pred, target)
    }
    
    if lpips_model is not None:
        metrics['lpips'] = compute_lpips_img(pred, target, lpips_model, device)
    
    return metrics


def initialize_lpips_model(net='alex', device='cuda'):
    """
    Initialize LPIPS model. Call this once at the start of your script.
    
    Args:
        net: Network to use ('alex', 'vgg', or 'squeeze')
             'alex' is fastest and most commonly used
        device: Device to load model on
    
    Returns:
        LPIPS model ready for evaluation
    """
    lpips_model = lpips.LPIPS(net=net).to(device)
    lpips_model.eval()  # Set to evaluation mode
    return lpips_model


def evaluate_batch(pred_images, target_images, lpips_model=None, device='cuda'):
    """
    Evaluate metrics across a batch of images.
    
    Args:
        pred_images: Tensor [B, H, W, 3] or list of [H, W, 3] tensors
        target_images: Tensor [B, H, W, 3] or list of [H, W, 3] tensors
        lpips_model: Optional LPIPS model
        device: Device for computation
    
    Returns:
        Dictionary with average metrics and per-image metrics
    """
    if isinstance(pred_images, torch.Tensor) and pred_images.dim() == 4:
        # Convert batch tensor to list
        pred_images = [pred_images[i] for i in range(pred_images.shape[0])]
        target_images = [target_images[i] for i in range(target_images.shape[0])]
    
    all_psnr = []
    all_ssim = []
    all_lpips = []
    
    for pred, target in zip(pred_images, target_images):
        metrics = compute_all_metrics(pred, target, lpips_model, device)
        all_psnr.append(metrics['psnr'])
        all_ssim.append(metrics['ssim'])
        if 'lpips' in metrics:
            all_lpips.append(metrics['lpips'])
    
    results = {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'ssim_mean': np.mean(all_ssim),
        'ssim_std': np.std(all_ssim),
        'psnr_per_image': all_psnr,
        'ssim_per_image': all_ssim,
    }
    
    if all_lpips:
        results['lpips_mean'] = np.mean(all_lpips)
        results['lpips_std'] = np.std(all_lpips)
        results['lpips_per_image'] = all_lpips
    
    return results