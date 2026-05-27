#!/usr/bin/env python3
"""
Crop all frames in the AvatarReX-style dataset at /gpfs/scratch1/shared/wmenu/dataset_rex
to a head-centered region, using the same calibration update logic as for the
SMPLX++ template.

Dataset layout (source, --src_root):
    calibration_full.json        # contains 16 camera entries named like "22010708", ...
    22010708/                    # camera folder
        00000000.jpg
        00000001.jpg
        ...
        mask/pha/00000000.jpg
        ...
    22010710/
    ...

Destination layout (--dst_root) mirrors this:
    calibration_full.json        # per-frame calibration: { "00000000": { cam: { K, R, T, ... } }, ... }
    22010708/00000000.jpg        # cropped & resized image (per-frame crop)
    22010708/mask/pha/00000000.jpg  # face mask from 4D-DRESS for that frame (per-frame)

Workflow:
1. Load dataset and init 4D-DRESS parsers (Graphonomy, SAM) once.
2. For each frame (or first --max_frames): run 4D-DRESS on that frame's 16 views
   to get per-view crop params and face masks for that frame; crop images and
   masks with those params, resize to (W0, H0), and save. Crop and mask are
   thus different for every frame.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import numpy as np
import torch

# Add project root to path so we can import repo modules when executed directly
# Repo root (this file is tools/python/crop_dataset_rex_faces.py -> parents[2])
proj_root = Path(__file__).resolve().parents[2]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

from dataset.dataset_mv_rgb import MvRgbDatasetAvatarReX
from tools.python.sam_face_hair_template_4ddress import (
    create_hyper_params,
    process_single_frame_to_crops_and_masks,
)
from tools.python.sam_face_hair_template_4ddress import GraphParser, Sam


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _crop_and_resize_image(img, x0, y0, w, h, W0, H0, is_mask: bool = False):
    """Crop [y0:y0+h, x0:x0+w] from img and resize back to (W0, H0)."""
    crop = img[y0 : y0 + h, x0 : x0 + w]
    if crop.size == 0:
        # Fallback: return resized original image if crop is invalid
        interp = cv.INTER_NEAREST if is_mask else cv.INTER_LINEAR
        return cv.resize(img, (W0, H0), interpolation=interp)

    interp = cv.INTER_NEAREST if is_mask else cv.INTER_LINEAR
    return cv.resize(crop, (W0, H0), interpolation=interp)


def _updated_calibration_from_crop_params(cam_data: dict, crop_params: dict, cam_names: list, W0: int, H0: int) -> dict:
    """Build calibration with updated K (digital zoom) from crop params."""
    updated = {}
    for view_idx, cam_name in enumerate(cam_names):
        if cam_name not in cam_data or view_idx not in crop_params:
            updated[cam_name] = cam_data.get(cam_name, {}).copy()
            continue
        K = np.array(cam_data[cam_name]["K"], dtype=np.float32).reshape(3, 3)
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        cp = crop_params[view_idx]
        cw, ch = cp["width"], cp["height"]
        crop_x0, crop_y0 = cp["x0"], cp["y0"]
        s_x = W0 / cw
        s_y = H0 / ch
        s = (s_x + s_y) / 2.0
        K_new = np.array([
            [s * fx, 0, s * (cx - crop_x0)],
            [0, s * fy, s * (cy - crop_y0)],
            [0, 0, 1],
        ], dtype=np.float32)
        updated[cam_name] = cam_data[cam_name].copy()
        updated[cam_name]["K"] = K_new.tolist()
        updated[cam_name]["imgSize"] = [W0, H0]
    return updated


def run_head_cropping(
    src_root: Path,
    dst_root: Path,
    frame_idx_for_crop: int = 0,
    max_frames: int | None = None,
    device: str = "cuda:0",
    graphonomy_ckpt_dir: str | None = None,
    sam_ckpt_dir: str | None = None,
    use_sam: bool = True,
):
    """
    Per-image pipeline (robust to missing frame indices):
    For each camera folder, take the first N existing `*.jpg` frames and run
    4D-DRESS on that single view to get a crop + face mask for that image.
    This ensures you get N outputs per folder even if some frame numbers are missing
    in some cameras.
    """
    t0 = time.time()
    graphonomy_ckpt_dir = graphonomy_ckpt_dir or "/scratch-shared/wmenu/4ddress_assets/graphonomy"
    # SAM: use arg if provided, else repo-relative, else scratch (symlink target may be missing)
    if sam_ckpt_dir is None:
        _repo = Path(__file__).resolve().parents[2]
        for _d in [
            _repo / "othercode" / "4d-dress" / "4dhumanparsing" / "checkpoints" / "sam",
            Path("/scratch-shared/wmenu/4ddress_assets/sam"),
        ]:
            _f = _d / "sam_vit_h_4b8939.pth"
            try:
                if _f.exists() and _f.resolve().stat().st_size > 0:
                    sam_ckpt_dir = str(_d)
                    break
            except OSError:
                pass
        else:
            sam_ckpt_dir = str(_repo / "othercode" / "4d-dress" / "4dhumanparsing" / "checkpoints" / "sam")
    sam_ckpt_dir = str(Path(sam_ckpt_dir).resolve())
    if use_sam and not (Path(sam_ckpt_dir) / "sam_vit_h_4b8939.pth").resolve().exists():
        raise FileNotFoundError(
            f"SAM checkpoint not found. Tried: {sam_ckpt_dir}\n"
            "Put sam_vit_h_4b8939.pth in that dir (or in othercode/4d-dress/4dhumanparsing/checkpoints/sam), or run with --no_sam."
        )

    print(f"Source root: {src_root}")
    print(f"Destination root: {dst_root}")
    _ensure_dir(dst_root)

    # Discover camera folders from calibration file (preferred)
    # Use only flat camera entries (keys with top-level "K"), not per-frame block keys (8-digit frame IDs).
    print("\n[Step 1] Loading calibration + camera list...")
    cam_data = None
    calib_path = src_root / "calibration_full.json"
    if calib_path.exists():
        with open(calib_path, "r") as f:
            full_calib = json.load(f)
        # Flat cameras: entries that have "K" at top level (exclude 8-digit frame blocks)
        cam_names = sorted([
            k for k in full_calib.keys()
            if isinstance(full_calib.get(k), dict) and "K" in full_calib.get(k, {})
            and not (k.isdigit() and len(k) == 8)
        ])
        if not cam_names:
            cam_names = list(full_calib.keys())
        cam_data = {k: full_calib[k] for k in cam_names}
        W0 = int(cam_data[cam_names[0]]["imgSize"][0])
        H0 = int(cam_data[cam_names[0]]["imgSize"][1])
    else:
        # Fallback: list directories that look like 8-digit camera ids
        cam_names = sorted([p.name for p in src_root.iterdir() if p.is_dir() and p.name.isdigit() and len(p.name) == 8])
        if not cam_names:
            raise RuntimeError(f"No camera folders found under {src_root}")
        # Infer size from first image
        first_img = cv.imread(str(src_root / cam_names[0] / "00000000.jpg"), cv.IMREAD_UNCHANGED)
        if first_img is None:
            raise RuntimeError("Could not infer image resolution (missing calibration_full.json and unreadable images).")
        H0, W0 = first_img.shape[:2]
    view_num = len(cam_names)
    print(f"Found {view_num} cameras: {cam_names}, resolution {W0}x{H0}")

    # Init 4D-DRESS parsers: Graphonomy always; SAM only when use_sam=True
    print("\n[Step 2] Initializing Graphonomy" + (" + SAM..." if use_sam else " (no SAM, parser-only crop)..."))
    parser = GraphParser(model_path=graphonomy_ckpt_dir, init_model=True, device=torch.device(device))
    sam = Sam(model_path=sam_ckpt_dir, init_model=True, device=torch.device(device)) if use_sam else None
    hyper_params = create_hyper_params(outfit="Inner", device=device)

    cap_msg = f"first {max_frames} images" if max_frames is not None else "all images"
    print(f"\n[Step 3] Processing {cap_msg} per camera folder (per-image crops/masks)...")

    # Copy SMPL params once
    if (src_root / "smpl_params.npz").exists() and not (dst_root / "smpl_params.npz").exists():
        (dst_root / "smpl_params.npz").write_bytes((src_root / "smpl_params.npz").read_bytes())

    # Create camera dirs
    for cam_name in cam_names:
        _ensure_dir(dst_root / cam_name)
        _ensure_dir(dst_root / cam_name / "mask" / "pha")

    num_images_processed = 0
    num_images_skipped = 0
    # Per-frame calibration: frame_name -> { cam_name -> { K, R, T, ... } }. Load existing if present.
    calibration_per_frame = {}
    dst_calib_path = dst_root / "calibration_full.json"
    if dst_calib_path.exists():
        try:
            with open(dst_calib_path, "r") as f:
                existing = json.load(f)
            # Load only per-frame blocks (8-digit frame keys); preserve for incremental run
            if existing:
                calibration_per_frame = {
                    k: dict(v) for k, v in existing.items()
                    if isinstance(k, str) and k.isdigit() and len(k) == 8 and isinstance(v, dict)
                }
        except (json.JSONDecodeError, TypeError):
            pass

    for cam_name in cam_names:
        cam_dir = src_root / cam_name
        if not cam_dir.exists():
            continue
        frame_paths = sorted([p for p in cam_dir.glob("*.jpg") if len(p.name) >= 8 and p.name[:8].isdigit()])
        if max_frames is not None:
            frame_paths = frame_paths[:max_frames]
        print(f"  Camera {cam_name}: {len(frame_paths)} frames")

        for i, img_path in enumerate(frame_paths):
            frame_name = img_path.name  # e.g. 00000000.jpg
            frame_key = img_path.stem  # e.g. 00000000 (for calibration)
            dst_img_path = dst_root / cam_name / frame_name
            dst_mask_path = dst_root / cam_name / "mask" / "pha" / frame_name
            # If outputs already exist, skip (incremental fill)
            if dst_img_path.exists() and dst_mask_path.exists():
                num_images_skipped += 1
                continue
            color_img = cv.imread(str(img_path), cv.IMREAD_UNCHANGED)
            if color_img is None:
                continue
            rgb = cv.cvtColor(color_img, cv.COLOR_BGR2RGB)
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

            mask_path = cam_dir / "mask" / "pha" / frame_name
            mask_img = cv.imread(str(mask_path), cv.IMREAD_UNCHANGED) if mask_path.exists() else None
            if mask_img is None:
                mask_img = (np.any(rgb < 250, axis=2)).astype(np.uint8) * 255

            images_original = rgb[None, ...]  # (1, H0, W0, 3)
            render_masks_original = mask_img[None, ...].astype(np.uint8)  # (1, H0, W0)

            crop_params, face_masks_fullres = process_single_frame_to_crops_and_masks(
                images_original,
                render_masks_original,
                parser,
                sam,
                hyper_params,
                device=device,
                verbose=False,
                use_sam=use_sam,
            )
            cp = crop_params.get(0, {"x0": 0, "y0": 0, "width": W0, "height": H0})
            fm = face_masks_fullres[0] if face_masks_fullres and face_masks_fullres[0] is not None else None

            img_cropped = _crop_and_resize_image(rgb, cp["x0"], cp["y0"], cp["width"], cp["height"], W0, H0, is_mask=False)
            cv.imwrite(str(dst_img_path), cv.cvtColor(img_cropped, cv.COLOR_RGB2BGR))
            if fm is not None:
                mask_cropped = _crop_and_resize_image(fm, cp["x0"], cp["y0"], cp["width"], cp["height"], W0, H0, is_mask=True)
                cv.imwrite(str(dst_mask_path), mask_cropped)
            num_images_processed += 1
            # Per-frame calibration: updated K for this (frame, camera)
            if cam_data is not None:
                calib_one = _updated_calibration_from_crop_params(
                    cam_data, {0: cp}, [cam_name], W0, H0
                )
                calibration_per_frame.setdefault(frame_key, {}).update(calib_one)

    # Ensure every frame has all cameras: fill missing with flat (full-image) calibration
    if cam_data is not None and calibration_per_frame and cam_names:
        for frame_key in calibration_per_frame:
            for cam_name in cam_names:
                if cam_name not in calibration_per_frame[frame_key]:
                    calibration_per_frame[frame_key][cam_name] = dict(cam_data[cam_name])

    # Write calibration: flat cameras + per-frame blocks (so all 16 cams per frame are present)
    if cam_data is not None:
        # Preserve flat camera entries and add per-frame blocks with all 16 cameras each
        output_calib = {k: cam_data[k] for k in cam_names}
        for frame_key in calibration_per_frame:
            output_calib[frame_key] = calibration_per_frame[frame_key]
        with open(dst_root / "calibration_full.json", "w") as f:
            json.dump(output_calib, f, indent=2)

    elapsed = time.time() - t0
    print(
        f"\nFinished: {num_images_processed} processed, {num_images_skipped} skipped in {elapsed:.2f} s "
        f"({elapsed / max(1, num_images_processed):.4f} s/image)."
    )
    return elapsed, num_images_processed


def main():
    parser = argparse.ArgumentParser(
        description="Crop /gpfs/scratch1/shared/wmenu/dataset_rex to head region and update calibration."
    )
    parser.add_argument(
        "--src_root",
        type=str,
        required=True,
        help="Source dataset root (e.g. /gpfs/scratch1/shared/wmenu/dataset_rex)",
    )
    parser.add_argument(
        "--dst_root",
        type=str,
        required=True,
        help="Destination dataset root (e.g. /gpfs/scratch1/shared/wmenu/dataset_rex_cropped)",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Optional limit on number of frames per camera to process (for testing).",
    )
    parser.add_argument(
        "--frame_idx_for_crop",
        type=int,
        default=0,
        help="Frame index used to determine crop parameters via 4D-DRESS (default: 0).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device for 4D-DRESS pipeline (e.g., cuda:0).",
    )
    parser.add_argument(
        "--no_sam",
        action="store_true",
        help="Do not load SAM; use Graphonomy (parser) only for skin/hair crop. Faster, no big SAM model.",
    )
    parser.add_argument(
        "--sam_ckpt_dir",
        type=str,
        default=None,
        help="Directory containing sam_vit_h_4b8939.pth (default: repo othercode/.../checkpoints/sam or /scratch-shared/.../sam).",
    )

    args = parser.parse_args()

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    if not src_root.exists():
        raise FileNotFoundError(f"Source root does not exist: {src_root}")

    elapsed, num_images = run_head_cropping(
        src_root=src_root,
        dst_root=dst_root,
        frame_idx_for_crop=args.frame_idx_for_crop,
        max_frames=args.max_frames,
        device=args.device,
        use_sam=not args.no_sam,
        sam_ckpt_dir=args.sam_ckpt_dir,
    )

    print(
        f"\n=== Overall summary ===\n"
        f"Processed {num_images} images in {elapsed:.2f} seconds "
        f"({elapsed / max(1, num_images):.4f} s/image)."
    )


if __name__ == "__main__":
    main()

