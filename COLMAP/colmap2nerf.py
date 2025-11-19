#!/usr/bin/env python3
"""
colmap2nerf_fixed_autodetect.py

Drop-in replacement for colmap2nerf that:
 - uses COLMAP qvec as-is (no negation)
 - computes camera poses robustly
 - auto-detects whether to apply the common COLMAP->NeRF flip by
   comparing per-frame pointing angles to the scene center (choose the
   option that makes cameras point *toward* the center)
 - uses median of points3D.txt for scene center if available
 - uses percentile-based scaling (90th percentile -> target radius)
 - outputs transforms.json compatible with Instant-NGP / Nerfstudio
 - preserves most original CLI flags
"""

import argparse
import os
import sys
import math
import json
from pathlib import Path
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser(description="Convert COLMAP text model to NeRF transforms.json (auto-detect flip)")
    parser.add_argument("--text", default="colmap_text", help="COLMAP text folder")
    parser.add_argument("--images", default="images", help="images folder path")
    parser.add_argument("--out", default="transforms.json", help="output transforms.json path")
    parser.add_argument("--aabb_scale", default=32, choices=["1","2","4","8","16","32","64","128"])
    parser.add_argument("--skip_early", default=0, type=int)
    parser.add_argument("--keep_colmap_coords", action="store_true",
                        help="If set, force keeping COLMAP coordinates (skip flip check).")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output")
    return parser.parse_args()

# ---------- Math helpers ----------
def qvec2rotmat(qvec):
    """Convert quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix.
       Use qvec exactly as COLMAP writes it (do NOT negate by default)."""
    q0, q1, q2, q3 = qvec
    return np.array([
        [1 - 2*q2*q2 - 2*q3*q3, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1*q1 - 2*q2*q2]
    ])

def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

def sharpness(imagePath):
    img = cv2.imread(imagePath)
    if img is None:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(variance_of_laplacian(gray))

# ---------- I/O: read COLMAP TXT ----------
def read_cameras_txt(cameras_txt):
    cams = {}
    with open(cameras_txt, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            els = line.split()
            camid = int(els[0])
            model = els[1]
            w = float(els[2]); h = float(els[3])
            fl = float(els[4])
            cam = {"model": model, "w": w, "h": h, "fl_x": fl, "fl_y": fl,
                   "cx": w/2.0, "cy": h/2.0, "k1":0.0, "k2":0.0, "p1":0.0, "p2":0.0}
            try:
                if model == "SIMPLE_PINHOLE":
                    cam["cx"] = float(els[5]); cam["cy"] = float(els[6])
                elif model == "PINHOLE":
                    cam["fl_y"] = float(els[5]); cam["cx"] = float(els[6]); cam["cy"] = float(els[7])
                elif model == "OPENCV":
                    cam["fl_y"] = float(els[5]); cam["cx"]=float(els[6]); cam["cy"]=float(els[7])
                    cam["k1"]=float(els[8]); cam["k2"]=float(els[9]); cam["p1"]=float(els[10]); cam["p2"]=float(els[11])
            except Exception:
                pass
            cam["camera_angle_x"] = math.atan(cam["w"]/(cam["fl_x"]*2.0))*2.0
            cam["camera_angle_y"] = math.atan(cam["h"]/(cam["fl_y"]*2.0))*2.0
            cams[camid] = cam
    return cams

def read_points3d_txt(points_txt):
    if not os.path.exists(points_txt):
        return None
    pts=[]
    with open(points_txt, "r") as f:
        for line in f:
            line=line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4: continue
            x,y,z = float(parts[1]), float(parts[2]), float(parts[3])
            pts.append([x,y,z])
    if len(pts)==0:
        return None
    return np.array(pts)

def parse_images_txt(images_txt):
    """Return list of image records. images.txt is two lines per image:
       line A = image metadata (ID, qvec, tvec, camid, name)
       line B = 2D points (we skip)
       This function returns a list of dicts: {id,name,camid,qvec,tvec}
    """
    records=[]
    with open(images_txt, "r") as f:
        raw_lines = [l.rstrip("\n") for l in f.readlines() if l.strip()]
    lines = [l for l in raw_lines if not l.strip().startswith("#")]
    for idx in range(0, len(lines), 2):
        line = lines[idx].strip()
        elems = line.split()
        if len(elems) < 9:
            continue
        img_id = int(elems[0])
        qvec = np.array(list(map(float, elems[1:5])))
        tvec = np.array(list(map(float, elems[5:8])))
        camid = int(elems[8])
        name = " ".join(elems[9:])
        records.append({"id":img_id, "name":name, "camid":camid, "qvec":qvec, "tvec":tvec})
    return records

# ---------- coordinate mode helpers ----------
def apply_colmap_to_nerf_flip_c2w(c2w):
    """Apply the single explicit COLMAP->NeRF flip many scripts use.
       Important: keep this as a single, documented transform — we'll auto-detect."""
    c2w2 = c2w.copy()
    c2w2[0:3,2] *= -1
    c2w2[0:3,1] *= -1
    c2w2 = c2w2[[1,0,2,3],:]
    c2w2[2,:] *= -1
    return c2w2

def apply_same_flip_points(pts):
    """Apply the same flip to a point cloud (pts: N x 3)"""
    p = pts.copy()
    p[:,2] *= -1
    p[:,1] *= -1
    p = p[:, [1,0,2]]
    p[:,2] *= -1
    return p

# ---------- utilities ----------
def build_c2w_from_qt(qvec, tvec):
    R = qvec2rotmat(qvec)
    t = tvec.reshape([3,1])
    w2c = np.concatenate([np.concatenate([R,t],axis=1), np.array([[0,0,0,1]])], axis=0)
    c2w = np.linalg.inv(w2c)
    return c2w

def median_center_from_points_or_cameras(points3d_transformed, cam_positions):
    if points3d_transformed is not None and len(points3d_transformed) >= 10:
        return np.median(points3d_transformed, axis=0), "points3D (median)"
    if cam_positions is not None and len(cam_positions) >= 4:
        return np.median(cam_positions, axis=0), "camera median (fallback)"
    return np.mean(cam_positions, axis=0), "camera mean (last resort)"

def evaluate_mode(records, points3d, apply_flip):
    """Given parsed image records and original 3D points, compute:
       - transformed camera positions and forward axes under apply_flip mode
       - compute scene center (median of points or camera median after same transform)
       - return mean camera->center pointing angle (lower is better) and metadata"""
    frames = []
    cam_positions = []
    forwards = []
    for rec in records:
        c2w = build_c2w_from_qt(rec["qvec"], rec["tvec"])
        if apply_flip:
            c2w = apply_colmap_to_nerf_flip_c2w(c2w)
        pos = c2w[0:3,3].copy()
        fwd = c2w[0:3,2].copy()
        frames.append({"pos":pos, "fwd":fwd})
        cam_positions.append(pos)
    cam_positions = np.array(cam_positions)
    # transform points if available
    if points3d is not None and apply_flip:
        pts_t = apply_same_flip_points(points3d)
    else:
        pts_t = points3d.copy() if points3d is not None else None
    center, source = median_center_from_points_or_cameras(pts_t, cam_positions)
    # compute pointing angles
    angles = []
    for fr in frames:
        pos = fr["pos"] - center
        vec_to_center = -pos / (np.linalg.norm(pos)+1e-12)
        fwd = fr["fwd"] / (np.linalg.norm(fr["fwd"])+1e-12)
        dot = np.clip(np.dot(fwd, vec_to_center), -1.0, 1.0)
        ang = math.degrees(math.acos(dot))
        angles.append(ang)
    angles = np.array(angles)
    return {
        "mean_angle": float(np.mean(angles)),
        "median_angle": float(np.median(angles)),
        "max_angle": float(np.max(angles)),
        "center": center,
        "center_source": source,
        "cam_positions": cam_positions,
        "frames": frames,
        "points_transformed": pts_t
    }

# ---------- main ----------
def main():
    args = parse_args()
    TEXT_FOLDER = args.text
    IMAGE_FOLDER = args.images
    OUT_PATH = args.out
    AABB_SCALE = int(args.aabb_scale)
    SKIP_EARLY = int(args.skip_early)

    # sanity checks
    images_txt = os.path.join(TEXT_FOLDER, "images.txt")
    cameras_txt = os.path.join(TEXT_FOLDER, "cameras.txt")
    points3d_txt = os.path.join(TEXT_FOLDER, "points3D.txt")
    if not os.path.exists(images_txt):
        print(f"FATAL: {images_txt} missing. Did you run 'colmap model_converter --output_type TXT'?"); sys.exit(1)
    if not os.path.exists(cameras_txt):
        print(f"FATAL: {cameras_txt} missing."); sys.exit(1)

    cams = read_cameras_txt(cameras_txt)
    records = parse_images_txt(images_txt)
    pts = read_points3d_txt(points3d_txt)

    if len(records) == 0:
        print("FATAL: no images parsed. Check images.txt"); sys.exit(1)

    print(f"Read {len(cams)} cameras, {len(records)} image records, {0 if pts is None else len(pts)} 3D points.")

    if args.keep_colmap_coords:
        chosen_mode = False
        print("User requested --keep_colmap_coords: forcing no flip (use raw COLMAP coordinates).")
        mode_eval = evaluate_mode(records, pts, apply_flip=False)
    else:
        # evaluate both modes and choose one with smaller mean camera->center pointing angle
        print("Auto-detecting best coordinate mode (flip vs no-flip) by checking camera->center pointing angles...")
        eval_no_flip = evaluate_mode(records, pts, apply_flip=False)
        eval_flip = evaluate_mode(records, pts, apply_flip=True)

        print(f" no_flip: mean_angle={eval_no_flip['mean_angle']:.3f} deg (center from {eval_no_flip['center_source']})")
        print(f"    flip: mean_angle={eval_flip['mean_angle']:.3f} deg (center from {eval_flip['center_source']})")

        if eval_no_flip["mean_angle"] <= eval_flip["mean_angle"]:
            chosen_mode = False
            mode_eval = eval_no_flip
            print("Choosing NO-FLIP mode (cameras point closer towards scene center).")
        else:
            chosen_mode = True
            mode_eval = eval_flip
            print("Choosing FLIP mode (cameras point closer towards scene center).")

    # Now build final frames using chosen_mode and follow final centering + scaling procedure
    final_frames = []
    cam_positions = []
    for rec in records:
        c2w = build_c2w_from_qt(rec["qvec"], rec["tvec"])
        if chosen_mode:
            c2w = apply_colmap_to_nerf_flip_c2w(c2w)
        pos = c2w[0:3,3].copy()
        final_frames.append({"c2w": c2w, "pos": pos, "name": rec["name"], "camid": rec["camid"]})
        cam_positions.append(pos)
    cam_positions = np.array(cam_positions)

    # points under chosen mode
    if pts is not None:
        if chosen_mode:
            pts_chosen = apply_same_flip_points(pts)
        else:
            pts_chosen = pts.copy()
    else:
        pts_chosen = None

    # compute robust center & translate
    center, center_source = median_center_from_points_or_cameras(pts_chosen, cam_positions)
    print(f"Scene center chosen from {center_source}: {center}")

    for f in final_frames:
        f["c2w"][0:3,3] -= center

    # scaling: use 90th percentile of camera distances
    cam_dists = np.linalg.norm(np.array([f["c2w"][0:3,3] for f in final_frames]), axis=1)
    p90 = np.percentile(cam_dists, 90)
    if p90 <= 0:
        p90 = np.mean(cam_dists)
    target_scale = 4.0
    scale = target_scale / (p90 + 1e-12)
    print(f"Scaling by factor {scale:.6f} (p90 distance {p90:.6f}, target radius {target_scale})")
    for f in final_frames:
        f["c2w"][0:3,3] *= scale

    # Build JSON out
    out = {"aabb_scale": AABB_SCALE, "frames": []}
    # add top-level intrinsics from first camera if present
    first_cam = next(iter(cams.values()))
    if first_cam is not None:
        out.update({
            "camera_angle_x": first_cam.get("camera_angle_x"),
            "camera_angle_y": first_cam.get("camera_angle_y"),
            "fl_x": first_cam.get("fl_x"),
            "fl_y": first_cam.get("fl_y"),
            "cx": first_cam.get("cx"),
            "cy": first_cam.get("cy"),
            "w": first_cam.get("w"),
            "h": first_cam.get("h"),
            "k1": first_cam.get("k1", 0.0),
            "k2": first_cam.get("k2", 0.0),
            "p1": first_cam.get("p1", 0.0),
            "p2": first_cam.get("p2", 0.0),
            "is_fisheye": first_cam.get("is_fisheye", False),
        })

    for f in final_frames:
        tm = np.array(f["c2w"], dtype=float)
        frame_obj = {
            "file_path": "./" + os.path.relpath(os.path.join(args.images, f["name"])).replace("\\","/"),
            "transform_matrix": tm.tolist()
        }
        # copy per-frame intrinsics if available
        camid = f["camid"]
        if camid in cams:
            frame_obj.update(cams[camid])
        out["frames"].append(frame_obj)

    # write output
    if os.path.exists(args.out) and not args.overwrite:
        print(f"FATAL: {args.out} exists. Use --overwrite to replace."); sys.exit(1)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"Wrote '{args.out}' with {len(out['frames'])} frames. Chosen flip mode = {'FLIP' if chosen_mode else 'NO-FLIP'}.")

if __name__ == "__main__":
    main()
