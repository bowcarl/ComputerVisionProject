
"""
Code used for removing background from the images, and replacing it with pure black
"""

#pip install git+https://github.com/facebookresearch/segment-anything.git
#pip install opencv-python pillow torch torchvision

import os, cv2, torch
import numpy as np
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Paths
image_dir = r"Complete path to the images that needs their background removed"  # The image directory
output_dir = r"The directiry which the images should be outputted to"           # An empty directory

os.makedirs(output_dir, exist_ok=True)

# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Load checkpoint used for background removal.
# This file needs to be installed from GitHub under the section "Model checkpoints" from this link: https://github.com/facebookresearch/segment-anything.git
sam_checkpoint = "sam_vit_h_4b8939.pth"
sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)
sam.to(device)

# Automatic mask generator (full quality)
mask_generator = SamAutomaticMaskGenerator(
    sam,
    points_per_side=64,        # higher → more accurate masks
    pred_iou_thresh=0.88,      # stricter → fewer false positives
    stability_score_thresh=0.95
)

# Max dimension for resizing
max_dim = 1024

def merge_center_masks(masks, image_shape, radius_fraction=0.05):
    """
    Merge all masks that intersect the central region.
    radius_fraction: fraction of smaller image dimension defining center region
    """
    h, w = image_shape[:2]
    cy, cx = h // 2, w // 2
    radius = int(min(h, w) * radius_fraction)
    Y, X = np.ogrid[:h, :w]
    center_region = ((Y - cy)**2 + (X - cx)**2) <= radius**2

    combined_mask = np.zeros((h, w), dtype=np.uint8)
    for m in masks:
        mask_bool = m["segmentation"].astype(bool)
        if np.any(mask_bool & center_region):
            combined_mask = np.maximum(combined_mask, mask_bool.astype(np.uint8) * 255)
    return combined_mask

# Process images
for fname in os.listdir(image_dir):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img_path = os.path.join(image_dir, fname)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize if too large
    h, w = image_rgb.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1.0:
        new_w, new_h = int(w*scale), int(h*scale)
        image_rgb = cv2.resize(image_rgb, (new_w, new_h))
        print(f"Resized {fname} to {new_w}x{new_h}")

    masks = mask_generator.generate(image_rgb)
    if len(masks) == 0:
        print(f"No masks found for {fname}, skipping.")
        continue

    # Merge all masks near center
    alpha_mask = merge_center_masks(masks, image_rgb.shape, radius_fraction=0.1)

    # Apply mask to image and fill background with black, was white
    # white_bg = np.ones_like(image_rgb, dtype=np.uint8) * 255  # white background
    black_bg = np.zeros_like(image_rgb, dtype=np.uint8)  # black background
    image_noback = np.where(alpha_mask[..., None] == 255, image_rgb, black_bg)

    out_path = os.path.join(output_dir, os.path.splitext(fname)[0] + "_noback.png")
    image_noback_bgr = cv2.cvtColor(image_noback, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, image_noback_bgr)
    print(f"Saved: {out_path}")


print("All images processed successfully!")

