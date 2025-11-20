"""
Neural Radiance Fields (NeRF) training model. Takes a set of images and tries rendering a 3D scene.
"""

#################################
#            Imports            #
#################################
import torch
import matplotlib.pyplot as plt
import time
import os
import json
import cv2
import numpy as np
from pathlib import Path
import imageio
import matplotlib.pyplot as plt

from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    NDCMultinomialRaysampler,
    MonteCarloRaysampler,
    EmissionAbsorptionRaymarcher,
    ImplicitRenderer,
)

from nerf_model_baseline import NeuralRadianceField_OLD
from nerf_model_improved import NeuralRadianceField
from metrics import (initialize_lpips_model, evaluate_batch)

from utils.plot_image_grid import image_grid
from utils.generate_cow_renders import generate_cow_renders
from utils.helper_functions import (generate_rotating_nerf,
                                    huber,
                                    show_full_render,
                                    sample_images_at_mc_locs)




#################################
#      Variables and setup      #
#################################

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

volume_extent_world = 3.0
scale = 0.125               # Downscaling the images to reduce RAM usage


# Getting images for the COLMAP Objects
TRANSFORMS_FILE = r"path to images" 
IMAGE_FOLDER = r"path to transform.json" 


#################################
#           Functions           #
#################################

# Mask generation. Only works for black background images
def generate_mask_universal_black_bg(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().numpy()
    img_uint8 = (img * 255).astype(np.uint8)
    gray = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2GRAY)
   
    THRESHOLD = 15 # Set a low threshold close to black
    _, mask_bg_removed = cv2.threshold(gray, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask_bg_removed, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = mask.astype(np.float32) / 255.0
    return mask

def load_colmap_data(transforms_file, images_dir, device):
    """
    Loads images, centers the scene, rotates it to be flat (auto-orientation),
    and scales it to fit in the unit sphere.
    """ 
    with open(transforms_file, 'r') as f:
        data = json.load(f)
    frames = data.get("frames", [])

    # Intrinsic Parameters
    camera_params = data
    if "w" not in camera_params:
        W = 800 
        H = 800
    else:
        W = camera_params["w"]
        H = camera_params["h"]
    
    fov_x_rad = camera_params["camera_angle_x"]
    fov_half_deg = np.rad2deg(fov_x_rad / 2.0)
    
    # Load all C2W matrices
    c2w_matrices = []
    file_paths = []
    
    for frame in frames:
        matrix = np.array(frame["transform_matrix"])
        c2w_matrices.append(matrix)
        file_paths.append(frame["file_path"])
        
    c2w_matrices = np.array(c2w_matrices) # Shape (N, 4, 4)

    # Try and recenter
    camera_centers = c2w_matrices[:, :3, 3]
    center_of_attention = np.mean(camera_centers, axis=0)
    c2w_matrices[:, :3, 3] -= center_of_attention

    # Try and fix rotation on the images
    # SVD is used to find the plane that best fits the camera centers
    # and rotate that plane to be the XZ plane (flat ground).

    # SVD on the centered camera positions
    current_centers = c2w_matrices[:, :3, 3]
    U, S, Vh = np.linalg.svd(current_centers)
    
    # The vector corresponding to the smallest singular value (last row of Vh)
    # is the normal vector to the plane of the cameras (the "Up" direction of the ring)
    plane_normal = Vh[2, :]
    
    # We want to align this normal with the Target World Up
    target_up = np.array([0, 1, 0])
    
    # Check direction: if normal points down relative to standard coord, flip it
    if plane_normal[1] < 0:
        plane_normal = -plane_normal

    # Calculate rotation matrix to align plane_normal to target_up
    v = np.cross(plane_normal, target_up)
    c = np.dot(plane_normal, target_up)
    s = np.linalg.norm(v)
    
    if s < 1e-6:
        # Already aligned
        R_align = np.eye(3)
    else:
        k = np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])
        R_align = np.eye(3) + k + k @ k * ((1 - c) / (s ** 2))

    # Apply alignment rotation to all cameras
    for i in range(len(c2w_matrices)):
        c2w_matrices[i, :3, 3] = R_align @ c2w_matrices[i, :3, 3]
        c2w_matrices[i, :3, :3] = R_align @ c2w_matrices[i, :3, :3]

    print("Auto-orientation applied (Plane alignment via SVD).")

    # Rescale
    distances = np.linalg.norm(c2w_matrices[:, :3, 3], axis=1)
    max_dist = np.max(distances)
    scale_factor = 1.0 / max_dist 
    
    c2w_matrices[:, :3, 3] *= scale_factor
    
    # Convert to Pytorch3D
    loaded_images = []
    Rs, Ts = [], []
    
    R_flip = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float32)

    for idx, file_path in enumerate(file_paths):
        filename = Path(file_path).name
        image_path = os.path.join(images_dir, filename)
        
        if not os.path.exists(image_path):
            continue

        img = cv2.imread(image_path)
        if img is None: continue

        if scale != 1.0:
            H_orig, W_orig, _ = img.shape
            W = int(W_orig * scale)
            H = int(H_orig * scale)
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0
        loaded_images.append(torch.from_numpy(img).float())

        c2w = c2w_matrices[idx]
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]

        R_c2w = R_c2w @ R_flip 
        
        R_w2c = R_c2w.T
        T_w2c = -R_c2w.T @ T_c2w
        
        Rs.append(torch.from_numpy(R_w2c).float())
        Ts.append(torch.from_numpy(T_w2c).float())
    
    if not loaded_images:
        raise ValueError("No images loaded.")
    
    target_images = torch.stack(loaded_images)
    masks = [torch.from_numpy(generate_mask_universal_black_bg(img_tensor.numpy())).float() 
             for img_tensor in loaded_images]
    target_silhouettes = torch.stack(masks)
    
    Rs = torch.stack(Rs)
    Ts = torch.stack(Ts)
    
    Zn, Zf = 0.1, 3.0 

    target_cameras = FoVPerspectiveCameras(R=Rs, T=Ts, znear=Zn, zfar=Zf, fov=fov_half_deg, aspect_ratio=W/H, device=device)
    return target_cameras, target_images, target_silhouettes

def cow_model(): 
    target_cameras, target_images, target_silhouettes = generate_cow_renders(num_views=40, azimuth_range=180) 
    print(f'Generated {len(target_images)} images/silhouettes/cameras.')
    return target_cameras, target_images, target_silhouettes


def colmap_model():
    print(f"Loading data from {TRANSFORMS_FILE} and {IMAGE_FOLDER}...")
    try:
        target_cameras, target_images, target_silhouettes = load_colmap_data(TRANSFORMS_FILE, IMAGE_FOLDER, device)
    except Exception as e:
        print(f"FATAL: Error loading data: {e}")
        exit(1)

    print(f"Loaded {len(target_images)} images/cameras from real-world data.")

    H, W = target_images.shape[1:3]
    print(f"Image size: {H}x{W}")

    return target_cameras, target_images, target_silhouettes


#################################
#    Image collection setup     #
#################################

# What code to run. 0 means running with custom COLMAP data
use_cow_data = 0

if use_cow_data == 1:
    target_cameras, target_images, target_silhouettes = cow_model()
    H_render = target_images.shape[1] * 2
    W_render = H_render
else:
    print("Using COLMAP...")
    volume_extent_world = 6.0 # Objects are larger, so need a higher value
    target_cameras, target_images, target_silhouettes = colmap_model()
    H_render = target_images.shape[1]
    W_render = target_images.shape[2]



# Plotting the camera angles, used for debugging
def visualize_cameras(cameras, title="Camera Setup"):
    # Get camera centers in World Coordinates
    cam_centers = cameras.get_camera_center() # Shape (N, 3)
    
    # Move to CPU and numpy for plotting
    cam_centers = cam_centers.cpu().numpy()
    
    x = cam_centers[:, 0]
    y = cam_centers[:, 1]
    z = cam_centers[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot Cameras as blue dots
    ax.scatter(x, y, z, c='blue', marker='^', label='Cameras')
    
    # Plot the Object Center (0,0,0) as a red dot
    ax.scatter([0], [0], [0], c='red', marker='o', s=100, label='Object Center (0,0,0)')
    
    # Draw lines from cameras to center to verify orientation
    # Only every the direction of every 5th camera is visualised
    for i in range(0, len(x), 5):
        ax.plot([x[i], 0], [y[i], 0], [z[i], 0], 'k-', alpha=0.2)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()
    
    # Set axis limits to be equal so the scale isn't distorted
    max_range = max(x.max()-x.min(), y.max()-y.min(), z.max()-z.min()) / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.show()

visualize_cameras(target_cameras)

#################################
#     Neural field setup        #
#################################

neural_radiance_field = NeuralRadianceField()
neural_radiance_field = neural_radiance_field.to(device)

raysampler_mc = MonteCarloRaysampler(
    min_x = -1.0,
    max_x = 1.0,
    min_y = -1.0,
    max_y = 1.0,
    n_rays_per_image=750,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world,
)

raymarcher_grid = EmissionAbsorptionRaymarcher()
renderer_mc = ImplicitRenderer(raysampler=raysampler_mc, raymarcher=raymarcher_grid)


raysampler_grid = NDCMultinomialRaysampler(
    image_height=H_render,
    image_width=W_render,
    n_pts_per_ray=128,
    min_depth=0.1,
    max_depth=volume_extent_world)

renderer_grid = ImplicitRenderer(raysampler=raysampler_grid, raymarcher=raymarcher_grid)


torch.manual_seed(1)
renderer_grid = renderer_grid.to(device)
renderer_mc = renderer_mc.to(device)
target_cameras = target_cameras.to(device)
target_images = target_images.to(device)
target_silhouettes = target_silhouettes.to(device)

#################################
#     Training loop setup       #
#################################

lr = 1e-3
optimizer = torch.optim.Adam(neural_radiance_field.parameters(), lr=lr) 
batch_size = 1
n_iter = 1001
start_iter = 0
start_time = time.time()
loss_history_color, loss_history_sil = [], []
lpips_model = initialize_lpips_model(net='alex', device=device)

"""
######### Resume training ############ 
checkpoint = torch.load('nerf_checkpoint_30000.pth', map_location=device)
neural_radiance_field.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_iter = checkpoint['iteration'] + 1
print(f"Resuming training from iteration {start_iter}")
loss_history_color = checkpoint['loss_history']['color']
loss_history_color = checkpoint['loss_history']['silhouette']
"""

t_iter = n_iter + start_iter # used in case training is continued

for iteration in range(start_iter, t_iter):
    if iteration == round(t_iter * 0.75):
        print('Decreasing LR 10-fold ...')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

        optimizer.zero_grad()
    batch_idx = torch.randperm(len(target_cameras))[:batch_size]

    batch_cameras = FoVPerspectiveCameras(
        R = target_cameras.R[batch_idx],
        T = target_cameras.T[batch_idx],
        znear = target_cameras.znear[batch_idx],
        zfar = target_cameras.zfar[batch_idx],
        aspect_ratio = target_cameras.aspect_ratio[batch_idx],
        fov = target_cameras.fov[batch_idx],
        device = device)

    rendered_images_silhouettes, sampled_rays = renderer_mc(
        cameras=batch_cameras,
        volumetric_function=neural_radiance_field
    )

    rendered_images, rendered_silhouettes = (
        rendered_images_silhouettes.split([3, 1], dim=-1)
    )

    silhouettes_at_rays = sample_images_at_mc_locs(
        target_silhouettes[batch_idx, ..., None],
        sampled_rays.xys
    )

    sil_err = huber(
        rendered_silhouettes,
        silhouettes_at_rays,
    ).abs().mean()
    colors_at_rays = sample_images_at_mc_locs(
        target_images[batch_idx],
        sampled_rays.xys
    )

    color_err = huber(
        rendered_images,
        colors_at_rays,
    ).abs().mean()

    loss = color_err + sil_err
    loss_history_color.append(float(color_err))
    loss_history_sil.append(float(sil_err))

    loss.backward()
    optimizer.step()

    if iteration % 1000 == 0:
        elapsed_time = time.time() - start_time
        print(f"Iter {iteration}: loss={loss.item():.5f}, color={color_err.item():.5f}, "
              f"sil={sil_err.item():.5f}, time={elapsed_time:.2f}s")
        if iteration % 5000 == 0:
            show_idx = torch.randperm(len(target_cameras))[:1]
            fig = show_full_render(
                neural_radiance_field,
                FoVPerspectiveCameras(
                    R = target_cameras.R[show_idx], 
                    T = target_cameras.T[show_idx], 
                    znear = target_cameras.znear[show_idx],
                    zfar = target_cameras.zfar[show_idx],
                    aspect_ratio = target_cameras.aspect_ratio[show_idx],
                    fov = target_cameras.fov[show_idx],
                    device = device,
                ), 
                target_images[show_idx][0],
                target_silhouettes[show_idx][0],
                renderer_grid,
                loss_history_color,
                loss_history_sil,
            )
            fig.savefig(f'intermediate_{iteration}')

##### Saving a checkpoint #####
"""
checkpoint = {
    'iteration': n_iter,
    'model_state_dict': neural_radiance_field.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss_history': {
        'color': loss_history_color,
        'silhouette': loss_history_sil
    }
}
torch.save(checkpoint, 'nerf_checkpoint_100000.pth')
print("Checkpoint saved!")
"""

#################################
#      Displaying Metrics       #
#################################

print("FINAL EVALUATION ON ALL IMAGES")
all_rendered = []
all_targets = []

# Get actual image size for consistent evaluation
H_target, W_target = target_images.shape[1:3]
print(f"Target image size: {H_target}x{W_target}")

# Create evaluation renderer with matching dimensions
raysampler_eval = NDCMultinomialRaysampler(
    image_height=H_target,
    image_width=W_target,
    n_pts_per_ray=128,
    min_depth=1.0,
    max_depth=6.0
)
renderer_eval = ImplicitRenderer(
    raysampler=raysampler_eval, 
    raymarcher=raymarcher_grid
).to(device)

with torch.no_grad():
    for i in range(len(target_cameras)):
        cam = FoVPerspectiveCameras(
            R=target_cameras.R[i:i+1],
            T=target_cameras.T[i:i+1],
            znear=target_cameras.znear[i:i+1],
            zfar=target_cameras.zfar[i:i+1],
            aspect_ratio=target_cameras.aspect_ratio[i:i+1],
            fov=target_cameras.fov[i:i+1],
            device=device
        )
        
        # Use evaluation renderer with correct size
        rendered, _ = renderer_eval(
            cameras=cam,
            volumetric_function=neural_radiance_field.batched_forward
        )
        
        all_rendered.append(rendered[0, ..., :3])
        all_targets.append(target_images[i])
        
        if (i + 1) % 10 == 0:
            print(f"Rendered {i+1}/{len(target_cameras)} images...")

# Compute final metrics
final_results = evaluate_batch(all_rendered, all_targets, lpips_model, device)

print("FINAL RESULTS:")
print(f"PSNR:  {final_results['psnr_mean']:.2f} ± {final_results['psnr_std']:.2f} dB")
print(f"SSIM:  {final_results['ssim_mean']:.4f} ± {final_results['ssim_std']:.4f}")
print(f"LPIPS: {final_results['lpips_mean']:.4f} ± {final_results['lpips_std']:.4f}")


#################################
#      Displaying Results       #
#################################

# Rendering a grid of images
with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(neural_radiance_field, target_cameras, renderer_grid, n_frames=4*5, device=device) 
image_grid(rotating_nerf_frames.clamp(0., 1.).cpu().numpy(), rows=4, cols=5, rgb=True, fill=True)
plt.show()


# Rendering a GIF of the 3D scene
print("Loading GIF...")
with torch.no_grad():
    rotating_nerf_frames = generate_rotating_nerf(
        neural_radiance_field,
        target_cameras,
        renderer_grid,
        n_frames=4*5,
        device=device
    )
frames = (rotating_nerf_frames.clamp(0., 1.).cpu().numpy() * 255).astype('uint8') # Convert frames to 0-255 and uint8 for GIF
name = "rotating_nerf.gif"     # Save as a GIF
imageio.mimsave(name, frames, fps=5, loop=0)