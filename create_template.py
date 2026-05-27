#!/usr/bin/env python3
"""
Create SMPLX++ pipeline input from dataset_rex.

- Finds all 8-digit subfolders (camera/view IDs).
- For each: loads first-frame image and mask (mask/pha/...), applies mask, saves to images/<cam_id>.png (RGBA, transparent background).
- Writes calibration_full.json containing only the cameras for those images.

Output folder contains:
  images/              - one masked image per camera as RGBA PNG (transparent background)
  calibration_full.json - NeuS2-style: w, h, aabb_scale, scale, offset, from_na, frames[]
    (each frame: file_path, transform_matrix 4x4 c2w, intrinsic_matrix 4x4 from K).
    w, h match the actual image shape.
  smpl_params.npz       - copied from data_dir (required by the full SMPLX++ pipeline).

Images are always written as RGBA PNG with transparent background (alpha=0 outside mask).
"""

import argparse
import json
import re
import shutil
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    p = argparse.ArgumentParser(description="Create SMPLX++ pipeline input: masked images in images/ + calibration_full.json")
    p.add_argument(
        "--data_dir",
        type=str,
        default="/gpfs/scratch1/shared/wmenu/dataset_rex",
        help="Root directory containing 8-digit subfolders, calibration_full.json",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Output directory (default: <data_dir>/smplxpp_input)",
    )
    p.add_argument(
        "--frame",
        type=str,
        default="00000003",
        help="Frame ID to use (e.g. 00000000, 00000003)",
    )
    p.add_argument(
        "--mask_thresh",
        type=int,
        default=128,
        help="Mask threshold (0-255); pixels with value >= thresh are foreground",
    )
    args = p.parse_args()

    data_root = Path(args.data_dir)
    out_root = Path(args.out_dir) if args.out_dir else data_root / "smplxpp_input"
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = out_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # 1) Find 8-digit subfolders (cameras)
    digit8 = re.compile(r"^\d{8}$")
    subdirs = sorted([d for d in data_root.iterdir() if d.is_dir() and digit8.match(d.name)])
    if not subdirs:
        raise FileNotFoundError(f"No 8-digit subfolders in {data_root}")
    print(f"Found {len(subdirs)} view folders")

    # 2) Load calibration and keep only cameras we have
    calib_path = data_root / "calibration_full.json"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration: {calib_path}")
    with open(calib_path, "r") as f:
        calib_data = json.load(f)

    cam_ids = [d.name for d in subdirs]
    calib_keys = set(calib_data.keys())
    missing = [c for c in cam_ids if c not in calib_keys]
    if missing:
        raise RuntimeError(f"Calibration missing keys for cameras: {missing}")
    # 3) For each camera: load image + mask, apply mask, save to images/<cam_id>.png (RGBA, transparent); build frames
    frame_id = args.frame
    frames = []
    out_w, out_h = None, None
    for subdir in subdirs:
        cam_id = subdir.name
        img_path = subdir / f"{frame_id}.jpg"
        mask_path = subdir / "mask" / "pha" / f"{frame_id}.jpg"
        if not img_path.exists():
            print(f"  Skip {cam_id}: missing image {img_path}")
            continue
        if not mask_path.exists():
            print(f"  Skip {cam_id}: missing mask {mask_path}")
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skip {cam_id}: failed to read image {img_path}")
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"  Skip {cam_id}: failed to read mask {mask_path}")
            continue

        if out_w is None:
            out_h, out_w = img.shape[0], img.shape[1]
        if mask.shape[:2] != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)

        m = (mask >= args.mask_thresh).astype(np.uint8)
        m_3ch = m[:, :, np.newaxis] if img.ndim == 3 else m
        # Transparent background: RGB = image only where mask, 0 elsewhere; alpha = mask (255 fg, 0 bg)
        masked_rgb = (img * m_3ch).astype(np.uint8)
        alpha = (m * 255).astype(np.uint8)
        if alpha.ndim == 2:
            alpha = alpha[:, :, np.newaxis]
        rgba = np.concatenate([masked_rgb, alpha], axis=-1)  # BGR + A for cv2
        ext, file_path_suffix = ".png", ".png"
        out_img_path = images_dir / f"{cam_id}{ext}"
        cv2.imwrite(str(out_img_path), rgba)

        print(f"  {cam_id} -> images/{cam_id}{ext}")

        # Build frame entry: K -> intrinsic_matrix (4x4), R,T -> transform_matrix (c2w 4x4)
        cam = calib_data[cam_id]
        R = np.array(cam["R"], dtype=np.float32).reshape(3, 3)
        T = np.array(cam["T"], dtype=np.float32).reshape(3, 1)
        K = np.array(cam["K"], dtype=np.float32).reshape(3, 3)
        w2c = np.eye(4, dtype=np.float32)
        w2c[:3, :3] = R
        w2c[:3, 3] = T.ravel()
        c2w = np.linalg.inv(w2c)
        K_4 = np.eye(4, dtype=np.float32)
        K_4[:3, :3] = K
        frames.append({
            "file_path": f"images/{cam_id}{file_path_suffix}",
            "transform_matrix": c2w.tolist(),
            "intrinsic_matrix": K_4.tolist(),
        })

    if not frames:
        raise RuntimeError(f"No valid images found for frame {frame_id}. All views were skipped (missing image or mask).")

    # 4) Write calibration_full.json in NeuS2-style format (w, h from actual image shape)
    calib_out = {
        "w": out_w,
        "h": out_h,
        "aabb_scale": 1.0,
        "scale": 0.5,
        "offset": [0.5, 0.5, 0.5],
        "from_na": True,
        "frames": frames,
    }
    calib_out_path = out_root / "calibration_full.json"
    with open(calib_out_path, "w") as f:
        json.dump(calib_out, f, indent=4)
    print(f"Wrote {calib_out_path} ({len(frames)} frames, w={out_w} h={out_h})")

    # 5) Copy SMPL-X params so the output is ready for the full pipeline
    smpl_copied = False
    for name in ("smpl_params.npz", "smpl_params.json"):
        src = data_root / name
        if src.exists():
            dst = out_root / name
            shutil.copy2(src, dst)
            print(f"Copied {name} -> {dst}")
            smpl_copied = True
            break
    if not smpl_copied:
        raise FileNotFoundError(
            f"No smpl_params.npz or smpl_params.json in {data_root}. "
            "The full pipeline requires SMPL-X parameters."
        )

    print(f"\nDone. Output: {out_root}")
    print("  images/  - masked images (one per camera)")
    print("  calibration_full.json - NeuS2-style (w, h, frames with transform_matrix, intrinsic_matrix)")
    print("  smpl_params.npz - SMPL-X parameters (for pipeline)")


if __name__ == "__main__":
    main()
