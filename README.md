# 🚀 Complete COLMAP to NeRF Training Guide

This guide details the setup, image processing (for both GPU and CPU systems), and the final configuration required to run your PyTorch3D NeRF training script successfully.
I. Initial Project Setup & Prerequisites
1. Project Folder Structure (Critical First Step)
Create the main project folder on your desktop named nerf_test (e.g., C:\Users\YourUser\Desktop\nerf_test).
Inside nerf_test, create an empty folder named images.
Place the conversion script colmap2nerf.py inside nerf_test.
Place all the high-quality input images (50+ recommended) inside the images folder (You can either take photos with your phone or a camera).
Instructions for taking photos:
Avoid shadows that change from different angels.Using a solid background color (light blue, grey, etc), makes the result better. Keep the object centered and stable, use consistent height and spacing. More accurate and fast using smaller objects (40-80 depending on the size) well angled photos. Use (20k iterations for testing) and maybe 50k for the actual report and potential 3D printing.
Path Example
Contents
C:\Users\YourUser\Desktop\nerf_test
Main project directory.
├── colmap2nerf.py
Conversion script.
└── images
Your input images.

2. COLMAP Installation
Download and extract the latest COLMAP release from GitHub Releases.
Extract the zip file to a standard location, such as C:\Program Files\COLMAP.
(Here you might need administrative privileges to create folders in Program Files.)
Open Command Prompt as Administrator and run the following command to add COLMAP to your system's PATH, allowing you to run it from any folder:
set PATH=%PATH%;"C:\Program Files\COLMAP"



II. COLMAP Image Processing (Choose Your Path)
Navigate your Command Prompt to your project folder (e.g., cd C:\Users\YourUser\Desktop\nerf_test).
Option A: NVIDIA (CUDA) GPU via GUI (Fastest)
Use the COLMAP GUI to take advantage of CUDA acceleration.
Launch the COLMAP GUI. (this is the COLMAP.bat file)
Feature Extraction (GPU):
Go to Processing → Feature extraction.
Set the Image path to your images folder.
Set the Database path to be an empty file named colmap.db in your nerf_test root folder.
Under SiftExtraction, ensure use_gpu is set to true.
Click Extract.
Feature Matching (GPU):
Go to Processing → Feature matching.
Ensure use_gpu is set to true.
Select the Exhaustive matching method.
Click Match.
Sparse Reconstruction:
Go to Reconstruction → Start new reconstruction.
Set Output path to a new folder named colmap_sparse.
Click Run.
Option B: CMD/CPU (Universal & AMD)
Run these commands sequentially in the Command Prompt. This option explicitly disables GPU use (use_gpu 0) for maximum compatibility.
Step
Command
Description
1. Feature Extraction
colmap feature_extractor --database_path colmap.db --image_path images --SiftExtraction.use_gpu 0
Finds features using the CPU.
2. Feature Matching
colmap exhaustive_matcher --database_path colmap.db --SiftMatching.use_gpu 0
Matches features between image pairs.
3. Sparse Reconstruction
colmap mapper --database_path colmap.db --image_path images --output_path colmap_sparse
Generates the 3D point cloud and camera poses.


Resource: COLMAP Command Line Guide
The video below explains the COLMAP process and commands, which can be helpful if you prefer to use the command line entirely, even on an NVIDIA machine (by changing use_gpu 0 to use_gpu 1):
https://www.youtube.com/watch?v=s-RP4yiMqP4
Final Conversion (Required for All Options)
After the sparse reconstruction is complete, run these two final commands in the Command Prompt from your nerf_test directory:
Step
Command
Description
4. Convert Binary to Text
colmap model_converter --input_path colmap_sparse/0 --output_path colmap_text --output_type TXT
Converts the reconstruction into a text-based format.
5. Generate NeRF JSON
python colmap2nerf.py --text colmap_text --images images
Output: Creates the final transforms.json file, containing all camera intrinsics and poses.

III. NeRF Training Script Configuration
The NeRF training script must be updated in two crucial areas to ensure robust execution and GPU acceleration.
1. Absolute Path Configuration (Required for Both)
Update the following lines in your training script to use the correct absolute raw string paths for the input files. This is essential for preventing pathing errors. 
As seen from the code below the NeRF training script is based on the intrinsics and poses gathered from the transforms.json file created with Colmap earlier.
# UPDATE THESE PATHS IN YOUR NeRF SCRIPT
# Use r"" (raw string) to correctly handle Windows backslashes

TRANSFORMS_FILE = r"C:\Users\YourUser\Desktop\nerf_test\transforms.json" 
IMAGE_FOLDER = r"C:\Users\YourUser\Desktop\nerf_test\images"


2. Device Setup for GPU Acceleration
The script is configured to automatically detect and use the GPU. Ensure this logic is present at the beginning of your script:

# The script automatically selects CUDA if NVIDIA GPU is present.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


3. Memory Optimization (CRITICAL)
To avoid the out-of-memory error, ensure the intermediate visualization renderer is configured to a fixed, lower resolution (e.g., 512x512) instead of the original full image size:
# FIX: Fixed resolution for plotting/GIF generation (Prevents CPU/RAM exhaustion)
VISUALIZATION_SIZE = 512 

# NDCMultinomialRaysampler for full-frame visualization
raysampler_grid = NDCMultinomialRaysampler(
    image_height=VISUALIZATION_SIZE, 
    image_width=VISUALIZATION_SIZE, 
    # ... rest of parameters
)
