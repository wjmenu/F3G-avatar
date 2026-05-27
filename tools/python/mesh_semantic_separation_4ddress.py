#!/usr/bin/env python3
"""
Single-frame mesh semantic separation (skin / hair / shoe / upper / lower / outer) using 4D-DRESS components.

This script is meant to work on an AnimatableGaussians / AvatarReX-style single mesh + multiview images:
- Mesh: OBJ (from NeuS2 extraction)
- Views: RGB(A) images (e.g. 16 views)
- Cameras: NeRF/instant-ngp style `transform.json` (the one we already generate for NeuS2)

Pipeline (single frame):
1) Run Graphonomy (GraphParser) per-view to get per-pixel semantic labels.
2) Rasterize the mesh per-view using PyTorch3D to get face index + barycentric coordinates per pixel.
3) Back-project per-pixel labels to per-vertex votes, then pick argmax => per-vertex labels.
4) Export separated submeshes for hair/shoes/clothes using 4D-DRESS `extract_label_meshes`.

Notes:
- This does NOT run the full 4D-DRESS temporal pipeline (no RAFT / no SAM / no graph-cut by default).
- For best quality you can enable graph-cut if pygco is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np

if TYPE_CHECKING:
    import torch


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _load_images(images_dir: Path, num_views: int | None = None, frames: list | None = None) -> np.ndarray:
    """Load images as uint8 [V,H,W,3]. Accepts RGB or RGBA (drops A).

    If *frames* is provided (list of dicts with "file_path"), images are loaded
    in exactly that order, which guarantees alignment with transforms.json.
    Otherwise falls back to sorted directory listing.
    """
    import cv2

    if frames is not None:
        sel = frames[:num_views] if num_views is not None else frames
        paths = []
        for fr in sel:
            fp = fr["file_path"]
            # file_path may be relative to the parent of images_dir (e.g. "images/xxx.jpg")
            candidate = images_dir / Path(fp).name
            if not candidate.exists():
                candidate = images_dir.parent / fp
            _require(candidate.exists(), f"Image referenced in transforms not found: {fp} (tried {candidate})")
            paths.append(candidate)
    else:
        exts = (".png", ".jpg", ".jpeg")
        paths = sorted([p for p in images_dir.iterdir() if p.suffix.lower() in exts])
        _require(len(paths) > 0, f"No images found in {images_dir}")
        if num_views is not None:
            paths = paths[:num_views]

    imgs = []
    for p in paths:
        im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        _require(im is not None, f"Failed to read image: {p}")
        if im.ndim == 2:
            im = np.repeat(im[..., None], 3, axis=-1)
        if im.shape[2] == 4:
            im = im[:, :, :3]
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        imgs.append(im)
    # ensure consistent shape
    H, W = imgs[0].shape[:2]
    for i, im in enumerate(imgs):
        _require(im.shape[:2] == (H, W), f"Image {paths[i]} has shape {im.shape[:2]} != {(H, W)}")
    return np.stack(imgs, axis=0)


def _load_ngp_transforms(transforms_json: Path) -> Dict:
    j = json.loads(transforms_json.read_text())
    _require("frames" in j and isinstance(j["frames"], list) and len(j["frames"]) > 0, "Invalid transforms.json (no frames)")
    return j


def _get_intrinsics_from_frame(frame: Dict, global_j: Dict) -> Tuple[float, float, float, float, int, int]:
    w = int(frame.get("w", global_j.get("w", 0)))
    h = int(frame.get("h", global_j.get("h", 0)))
    _require(w > 0 and h > 0, "Missing w/h in transforms.json")
    # Prefer fl_x/fl_y/cx/cy if present (we generate these).
    if "fl_x" in frame and "fl_y" in frame and "cx" in frame and "cy" in frame:
        return float(frame["fl_x"]), float(frame["fl_y"]), float(frame["cx"]), float(frame["cy"]), w, h
    # Fallback to intrinsic_matrix (4x4 or 3x3): K[0][0]=fx, K[1][1]=fy, K[0][2]=cx, K[1][2]=cy
    if "intrinsic_matrix" in frame:
        K = frame["intrinsic_matrix"]
        return float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2]), w, h
    # Fallback to camera_angle_x/y (radians) if present.
    import math

    if "camera_angle_x" in frame:
        fx = 0.5 * w / math.tan(0.5 * float(frame["camera_angle_x"]))
        fy = fx
        if "camera_angle_y" in frame:
            fy = 0.5 * h / math.tan(0.5 * float(frame["camera_angle_y"]))
        cx = 0.5 * w
        cy = 0.5 * h
        return fx, fy, cx, cy, w, h
    if "camera_angle_x" in global_j:
        fx = 0.5 * w / math.tan(0.5 * float(global_j["camera_angle_x"]))
        fy = fx
        if "camera_angle_y" in global_j:
            fy = 0.5 * h / math.tan(0.5 * float(global_j["camera_angle_y"]))
        cx = float(global_j.get("cx", 0.5 * w))
        cy = float(global_j.get("cy", 0.5 * h))
        return fx, fy, cx, cy, w, h
    raise RuntimeError("Could not infer intrinsics. Provide fl_x/fl_y/cx/cy or intrinsic_matrix in each frame.")


def _ngp_c2w_to_pytorch3d_RT(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert NeRF/NGP camera-to-world (c2w) to PyTorch3D world-to-view R,T.

    PyTorch3D uses row-vector convention (X_cam = X_world @ R + T) with a
    camera frame where X points left and Y points up (opposite to OpenCV's
    X-right, Y-down). Verified empirically: flip XY then transpose gives
    IoU > 0.98 across all cameras.
    """
    _require(c2w.shape == (4, 4), f"Expected 4x4 c2w, got {c2w.shape}")
    w2c = np.linalg.inv(c2w)
    R_cv = w2c[:3, :3].astype(np.float32)
    t_cv = w2c[:3, 3].astype(np.float32)
    flip_xy = np.diag([-1, -1, 1]).astype(np.float32)
    R = (flip_xy @ R_cv).T.copy()
    T = (flip_xy @ t_cv).copy()
    return R, T


def _foreground_mask_from_rgb(rgb: np.ndarray, thr: int = 250) -> np.ndarray:
    """
    Build a foreground mask from a masked RGB image.
    Handles both white-background and black/transparent-background images by
    detecting which background type is dominant.
    """
    near_white = (rgb[..., 0] >= thr) & (rgb[..., 1] >= thr) & (rgb[..., 2] >= thr)
    near_black = (rgb[..., 0] <= (255 - thr)) & (rgb[..., 1] <= (255 - thr)) & (rgb[..., 2] <= (255 - thr))
    # If more pixels are near-black than near-white, assume black/transparent background
    if near_black.sum() > near_white.sum():
        return ~near_black
    return ~near_white


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _candidate_RT_from_c2w(c2w: np.ndarray, flip: np.ndarray, transpose_r: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a candidate world-to-view R,T for PyTorch3D by applying axis flips and optional transpose.
    flip: 3x3 diagonal matrix with entries +/-1
    transpose_r: whether to transpose rotation matrix (helps when conventions differ)
    """
    w2c = np.linalg.inv(c2w)
    R = w2c[:3, :3].astype(np.float32)
    T = w2c[:3, 3].astype(np.float32)
    # apply flip in camera space
    R = flip @ R
    T = flip @ T
    if transpose_r:
        R = R.T.copy()
    return R, T


def _auto_choose_camera_convention(
    frames: list[Dict],
    transforms: Dict,
    images: np.ndarray,
    V: np.ndarray,
    F: np.ndarray,
    scale_x: float,
    scale_y: float,
) -> Tuple[np.ndarray, bool, float]:
    """
    Try a small set of camera convention fixes and choose the one that best aligns
    the rasterized mesh silhouette with the foreground mask extracted from images.

    Returns: (flip_diag (3,), transpose_r, best_score)
    """
    import torch
    from pytorch3d.structures import Meshes
    from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    th_verts = torch.tensor(V, dtype=torch.float32, device=device).unsqueeze(0)
    th_faces = torch.tensor(F, dtype=torch.int64, device=device).unsqueeze(0)
    mesh = Meshes(verts=th_verts, faces=th_faces)

    H, W = images.shape[1], images.shape[2]
    # Use naive rasterization for camera convention testing to avoid overflow warnings
    raster_settings = RasterizationSettings(
        image_size=(H, W), 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=0,  # Disable binning for reliability during testing
    )

    # Use a few views for scoring to keep it fast.
    n_eval = min(4, images.shape[0])
    eval_ids = list(range(n_eval))

    flips = [
        np.diag([1, 1, 1]).astype(np.float32),
        np.diag([1, -1, -1]).astype(np.float32),
        np.diag([-1, 1, -1]).astype(np.float32),
        np.diag([-1, -1, 1]).astype(np.float32),
        np.diag([1, -1, 1]).astype(np.float32),
        np.diag([1, 1, -1]).astype(np.float32),
        np.diag([-1, 1, 1]).astype(np.float32),
        np.diag([-1, -1, -1]).astype(np.float32),
    ]
    transpose_opts = [False, True]

    best = (-1.0, None, None)  # (score, flip, transpose)
    for flip in flips:
        for tr in transpose_opts:
            scores = []
            for i in eval_ids:
                frame = frames[i]
                fx, fy, cx, cy, _, _ = _get_intrinsics_from_frame(frame, transforms)
                fx *= scale_x
                fy *= scale_y
                cx *= scale_x
                cy *= scale_y
                c2w = np.array(frame["transform_matrix"], dtype=np.float32)
                R, T = _candidate_RT_from_c2w(c2w, flip=flip, transpose_r=tr)

                cams = PerspectiveCameras(
                    focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                    principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                    R=torch.tensor(R[None, ...], dtype=torch.float32, device=device),
                    T=torch.tensor(T[None, ...], dtype=torch.float32, device=device),
                    in_ndc=False,
                    image_size=torch.tensor([[H, W]], dtype=torch.float32, device=device),
                    device=device,
                )
                r = MeshRasterizer(cameras=cams, raster_settings=raster_settings)(mesh)
                sil = (r.pix_to_face[0, :, :, 0] >= 0).detach().cpu().numpy()
                fg = _foreground_mask_from_rgb(images[i])
                scores.append(_iou(sil, fg))
            score = float(np.mean(scores)) if scores else -1.0
            if score > best[0]:
                best = (score, flip, tr)

    best_score, best_flip, best_tr = best
    _require(best_flip is not None and best_tr is not None, "Failed to choose camera convention")
    # Return diag for compactness
    return np.diag(best_flip).astype(np.float32), bool(best_tr), float(best_score)


def _render_labels_from_vertex_labels(
    faces: "torch.Tensor",
    pix_to_face: "torch.Tensor",
    bary: "torch.Tensor",
    v_labels: "torch.Tensor",
    nl: int,
) -> "torch.Tensor":
    """
    Render per-pixel labels from per-vertex labels using rasterization outputs.

    - faces: (F,3) long
    - pix_to_face: (H,W) long, -1 for background
    - bary: (H,W,3) float barycentric coords for faces_per_pixel=1
    - v_labels: (V,) long with values in [-1..nl-1]

    Returns:
    - labels_img: (H,W) long, -1 background
    """
    import torch

    H, W = pix_to_face.shape
    out = torch.full((H, W), -1, dtype=torch.long, device=pix_to_face.device)
    vis = pix_to_face >= 0
    if torch.count_nonzero(vis) == 0:
        return out

    face_idx = pix_to_face[vis]  # (P,)
    verts_idx = faces[face_idx]  # (P,3)
    bary_vis = bary[vis]         # (P,3)
    vlab = v_labels[verts_idx]   # (P,3)

    # Drop any pixels whose face has unlabeled vertices.
    valid = torch.all(vlab >= 0, dim=-1)
    if torch.count_nonzero(valid) == 0:
        return out

    verts_idx = verts_idx[valid]
    bary_vis = bary_vis[valid]
    vlab = vlab[valid]

    # Weighted vote per label: sum_j bary_j * 1[vlab_j == l]
    P = vlab.shape[0]
    scores = torch.zeros((P, nl), dtype=torch.float32, device=pix_to_face.device)
    for j in range(3):
        wj = bary_vis[:, j:j + 1]  # (P,1)
        lj = vlab[:, j]            # (P,)
        scores.scatter_add_(1, lj[:, None], wj)
    pred = torch.argmax(scores, dim=-1)  # (P,)

    out_vis = out[vis]
    out_vis[valid] = pred
    out[vis] = out_vis
    return out


def _label_colors(nl: int) -> np.ndarray:
    # 4D-DRESS surface label colors
    base = np.array(
        [[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]],
        dtype=np.uint8,
    )
    if nl <= base.shape[0]:
        return base[:nl]
    # fallback for >6 labels
    extra = np.random.RandomState(0).randint(0, 255, size=(nl - base.shape[0], 3), dtype=np.uint8)
    return np.concatenate([base, extra], axis=0)


def _save_label_and_overlay(
    out_label_png: Path,
    out_overlay_png: Path,
    rgb_image: np.ndarray,
    labels_hw: "np.ndarray",
    colors: np.ndarray,
) -> None:
    import cv2

    H, W = labels_hw.shape
    label_rgb = np.zeros((H, W, 3), dtype=np.uint8) + 255  # white bg
    for li in range(colors.shape[0]):
        label_rgb[labels_hw == li] = colors[li]

    # save label image
    out_label_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_label_png), cv2.cvtColor(label_rgb, cv2.COLOR_RGB2BGR))

    # overlay with input rgb
    base = rgb_image
    if base.shape[:2] != (H, W):
        base = cv2.resize(base, (W, H), interpolation=cv2.INTER_AREA)
    overlay = cv2.addWeighted(base.astype(np.uint8), 0.5, label_rgb, 0.5, 0.0)
    cv2.imwrite(str(out_overlay_png), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _import_4ddress(project_root: Path):
    """Add 4d-dress to sys.path and import GraphParser + extract_label_meshes + SURFACE_LABEL."""
    dress_root = project_root / "othercode" / "4d-dress"
    _require(dress_root.exists(), f"Missing 4d-dress folder at {dress_root}")

    sys.path.insert(0, str(dress_root))
    sys.path.insert(0, str(dress_root / "4dhumanparsing"))
    sys.path.insert(0, str(dress_root / "dataset"))

    # 4dhumanparsing/lib/utility/parser.py
    from lib.utility.parser import GraphParser
    from utility import SURFACE_LABEL  # dataset/utility.py
    from extract_garment import extract_label_meshes  # dataset/extract_garment.py

    return GraphParser, SURFACE_LABEL, extract_label_meshes


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_obj", required=True, help="Input mesh OBJ")
    p.add_argument("--transforms_json", required=True, help="NeRF/NGP transforms.json used for NeuS2")
    p.add_argument("--images_dir", required=True, help="Directory containing view images (ordered)")
    p.add_argument("--out_dir", required=True, help="Output directory")
    p.add_argument("--outfit", default="Outer", choices=["Inner", "Outer"], help="Controls label grouping (Outer has separate 'outer' label).")
    p.add_argument("--num_views", type=int, default=None, help="Optional: only use first N views")
    p.add_argument("--max_side", type=int, default=512, help="Downscale images so max(H,W)=max_side for parsing+rasterization. "
                                                            "This dramatically reduces GPU memory/time. Set 0 to disable.")
    p.add_argument("--raster_bin_size", type=int, default=None, help="PyTorch3D rasterization bin_size. 0=naive (no binning, slower but reliable). None=auto. Default: auto (0 for large meshes).")
    p.add_argument("--graphonomy_ckpt_dir", default=None, help="Path containing inference.pth (defaults to 4d-dress/4dhumanparsing/checkpoints/graphonomy)")
    p.add_argument("--raft_models_dir", default=None, help="Optional: path containing RAFT weights (e.g. raft-things.pth). Used only for verification in this script.")
    p.add_argument("--sam_ckpt_path", default=None, help="Optional: path to SAM checkpoint. Used only for verification in this script.")
    p.add_argument("--smooth_gco", action="store_true", default=True, help="Run pygco graph-cut smoothing if available (recommended for better quality, especially for small parts like shoes).")
    p.add_argument("--no_smooth_gco", dest="smooth_gco", action="store_false", help="Disable graph-cut smoothing.")
    p.add_argument("--gco_smooth_weight", type=float, default=5.0,
                   help="Graph-cut pairwise smooth weight. Controls how much spatial smoothing is applied. "
                        "Lower values preserve small regions (hair, shoes) better. "
                        "The original 4D-Dress uses 200 with multi-source evidence; "
                        "our single-source (Graphonomy) pipeline needs ~5.0. Default: 5.0")
    p.add_argument("--export_view_labels", action="store_true", help="Export per-view semantic label PNGs + overlay PNGs.")
    p.add_argument("--export_view_parser", action="store_true", help="Export per-view Graphonomy (mapped) label PNGs + overlay PNGs (pre-projection).")
    p.add_argument("--export_original_overlays", action="store_true", help="Export segmentation overlays on original full-resolution images (in addition to downscaled overlays).")
    p.add_argument("--views_subdir", default="views", help="Subdirectory under out_dir to store per-view outputs.")
    p.add_argument("--auto_cam_fix", action="store_true", help="Try to automatically fix camera convention by maximizing silhouette-vs-foreground overlap.")
    p.add_argument(
        "--upperbody_only",
        action="store_true",
        help="Subject has no bottom clothes (only upper body). Vertices classified as 'lower' are reassigned to 'upper' so lower body is merged into upper clothing.",
    )
    p.add_argument(
        "--lower_body_as_skin",
        action="store_true",
        help="Subject has bare legs (no pants/skirt). Vertices classified as 'lower' (dress/pants) are reassigned to 'skin' so the lower body is correctly labeled as skin.",
    )
    args = p.parse_args()

    # Repo root (this file lives in tools/python/...)
    project_root = Path(__file__).resolve().parents[2]
    mesh_obj = Path(args.mesh_obj)
    transforms_json = Path(args.transforms_json)
    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _require(mesh_obj.exists(), f"Missing mesh: {mesh_obj}")
    _require(transforms_json.exists(), f"Missing transforms.json: {transforms_json}")
    _require(images_dir.exists(), f"Missing images_dir: {images_dir}")

    # Import 4D-DRESS components
    GraphParser, SURFACE_LABEL, extract_label_meshes = _import_4ddress(project_root)

    # Resolve Graphonomy checkpoint path
    if args.graphonomy_ckpt_dir:
        ckpt_dir = Path(args.graphonomy_ckpt_dir)
    else:
        ckpt_dir = project_root / "othercode" / "4d-dress" / "4dhumanparsing" / "checkpoints" / "graphonomy"
    ckpt_path = ckpt_dir / "inference.pth"
    _require(ckpt_path.exists(), f"Missing Graphonomy checkpoint: {ckpt_path}\n"
                                 f"See othercode/4d-dress/README.md for download instructions.")

    # Optional: verify RAFT/SAM checkpoints exist (scratch-backed symlinks are recommended)
    if args.raft_models_dir:
        raft_dir = Path(args.raft_models_dir)
    else:
        raft_dir = project_root / "othercode" / "4d-dress" / "4dhumanparsing" / "checkpoints" / "raft" / "models"
    if raft_dir.exists():
        _require((raft_dir / "raft-things.pth").exists(), f"RAFT models dir exists but raft-things.pth missing: {raft_dir}")
    else:
        print(f"[WARNING] RAFT models dir not found (not used by this script): {raft_dir}")

    if args.sam_ckpt_path:
        sam_ckpt = Path(args.sam_ckpt_path)
    else:
        sam_ckpt = project_root / "othercode" / "4d-dress" / "4dhumanparsing" / "checkpoints" / "sam" / "sam_vit_h_4b8939.pth"
    if sam_ckpt.exists():
        pass
    else:
        print(f"[WARNING] SAM checkpoint not found (not used by this script): {sam_ckpt}")

    # Load mesh
    import trimesh
    tri = trimesh.load(mesh_obj, force="mesh", process=False)
    V = np.asarray(tri.vertices, dtype=np.float32)
    F = np.asarray(tri.faces, dtype=np.int64)
    edges = tri.edges_unique

    # Load transforms first so we can load images in the exact frame order
    transforms = _load_ngp_transforms(transforms_json)
    frames = transforms["frames"]

    images = _load_images(images_dir, args.num_views, frames=frames)  # [V,H,W,3] uint8
    _require(len(frames) >= images.shape[0], f"transforms.json has {len(frames)} frames but loaded {images.shape[0]} images")

    # Optional downscale (must adjust intrinsics accordingly)
    scale_x = 1.0
    scale_y = 1.0
    if args.max_side and args.max_side > 0:
        H0, W0 = images.shape[1], images.shape[2]
        s = float(args.max_side) / float(max(H0, W0))
        if s < 1.0:
            import cv2

            new_w = int(round(W0 * s))
            new_h = int(round(H0 * s))
            # ensure at least 2x2
            new_w = max(2, new_w)
            new_h = max(2, new_h)
            resized = []
            for i in range(images.shape[0]):
                resized.append(cv2.resize(images[i], (new_w, new_h), interpolation=cv2.INTER_AREA))
            images = np.stack(resized, axis=0)
            scale_x = new_w / float(W0)
            scale_y = new_h / float(H0)

    # Init Graphonomy parser (runs on CUDA by default)
    parser = GraphParser(model_path=str(ckpt_dir), init_model=True)
    parser_images, parser_vals = parser.parser_images(images)  # parser_vals: torch [V,H,W] of 20-way labels

    # Map CIHP labels -> grouped surface labels (skin/hair/shoe/upper/lower/outer)
    import torch

    if args.outfit == "Outer":
        label_groups = parser.LABEL_GROUP_OUTER.to(parser_vals.device)  # (20,)
        surface_labels = SURFACE_LABEL  # skin hair shoe upper lower outer
    else:
        label_groups = parser.LABEL_GROUP_INNER.to(parser_vals.device)
        surface_labels = SURFACE_LABEL[:-1]  # drop outer

    # parser_vals: (V,H,W) with values 0..19 => map to grouped labels (-1..5)
    grp = label_groups[parser_vals]  # (V,H,W)

    # One-hot votes per pixel for each surface label (nl)
    nl = len(surface_labels)
    votes = torch.stack([(grp == i).float() for i in range(nl)], dim=-1)  # (V,H,W,nl)

    # Optional: export per-view Graphonomy labels (mapped to surface labels) for inspection
    if args.export_view_parser:
        import torch

        colors_nl = _label_colors(nl)
        view_root = out_dir / args.views_subdir
        for i in range(images.shape[0]):
            labels_i = grp[i].detach().cpu().numpy().astype(np.int32)  # (H,W) with -1..nl-1
            _save_label_and_overlay(
                out_label_png=view_root / "parser" / f"view{i:03d}_labels.png",
                out_overlay_png=view_root / "parser" / f"view{i:03d}_overlay.png",
                rgb_image=images[i],
                labels_hw=labels_i,
                colors=colors_nl,
            )

    # Rasterize + backproject to vertices
    try:
        from pytorch3d.structures import Meshes
        from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
    except Exception as e:
        raise RuntimeError(
            "PyTorch3D is required for rasterization-based projection. "
            "Install pytorch3d in your env and retry."
        ) from e

    device = parser_vals.device
    th_verts = torch.tensor(V, dtype=torch.float32, device=device).unsqueeze(0)
    th_faces = torch.tensor(F, dtype=torch.int64, device=device).unsqueeze(0)

    image_h, image_w = images.shape[1], images.shape[2]
    # Fix rasterization overflow by using appropriate bin sizes
    # For large meshes, use bin_size=0 (naive rasterization) to avoid overflow
    num_faces = th_faces.shape[1]
    if args.raster_bin_size is not None:
        bin_size = args.raster_bin_size
        print(f"[INFO] Using user-specified rasterization bin_size={bin_size}")
    elif num_faces > 500000:
        # Very large mesh - use naive rasterization to avoid bin overflow
        bin_size = 0
        print(f"[INFO] Using naive rasterization (bin_size=0) for large mesh ({num_faces} faces)")
    elif num_faces > 200000:
        # Large mesh - use naive for reliability
        bin_size = 0
        print(f"[INFO] Using naive rasterization (bin_size=0) for medium-large mesh ({num_faces} faces)")
    else:
        # Smaller mesh - can use binning
        bin_size = None  # Auto
        print(f"[INFO] Using auto binning for mesh ({num_faces} faces)")
    
    raster_settings = RasterizationSettings(
        image_size=(image_h, image_w), 
        blur_radius=0.0, 
        faces_per_pixel=1,
        bin_size=bin_size,
    )
    mesh = Meshes(verts=th_verts, faces=th_faces)

    # Per-vertex vote accumulator
    v_votes = torch.zeros((V.shape[0], nl), dtype=torch.float32, device=device)

    # Optionally auto-pick a camera convention fix
    flip_diag = np.array([1, 1, 1], dtype=np.float32)
    transpose_r = False
    if args.auto_cam_fix:
        flip_diag, transpose_r, best_score = _auto_choose_camera_convention(
            frames=frames,
            transforms=transforms,
            images=images,
            V=V,
            F=F,
            scale_x=scale_x,
            scale_y=scale_y,
        )
        print(f"[auto_cam_fix] best flip_diag={flip_diag.tolist()} transpose_r={transpose_r} silhouette_iou={best_score:.4f}")

    for i in range(images.shape[0]):
        frame = frames[i]
        fx, fy, cx, cy, w, h = _get_intrinsics_from_frame(frame, transforms)
        fx *= scale_x
        fy *= scale_y
        cx *= scale_x
        cy *= scale_y
        w = image_w
        h = image_h
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        if args.auto_cam_fix:
            flip = np.diag(flip_diag).astype(np.float32)
            R, T = _candidate_RT_from_c2w(c2w, flip=flip, transpose_r=transpose_r)
        else:
            R, T = _ngp_c2w_to_pytorch3d_RT(c2w)

        cams = PerspectiveCameras(
            focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
            principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
            R=torch.tensor(R[None, ...], dtype=torch.float32, device=device),
            T=torch.tensor(T[None, ...], dtype=torch.float32, device=device),
            in_ndc=False,
            image_size=torch.tensor([[h, w]], dtype=torch.float32, device=device),
            device=device,
        )
        r = MeshRasterizer(cameras=cams, raster_settings=raster_settings)(mesh)

        pix_to_face = r.pix_to_face[0, :, :, 0]
        bary = r.bary_coords[0, :, :, 0, :]
        vis = pix_to_face >= 0
        if torch.count_nonzero(vis) == 0:
            continue

        face_idx = pix_to_face[vis]
        verts_idx = th_faces[0, face_idx, :]
        bary_vis = bary[vis]
        votes_vis = votes[i][vis]

        for j in range(3):
            wj = bary_vis[:, j:j+1]
            contrib = votes_vis * wj
            v_votes.index_add_(0, verts_idx[:, j], contrib)

    v_votes = torch.nan_to_num(v_votes)
    vote_sums = v_votes.sum(dim=1, keepdims=True)
    vote_sums = torch.clamp(vote_sums, min=1e-8)
    v_votes_normalized = v_votes / vote_sums
    
    v_labels = torch.argmax(v_votes_normalized, dim=-1)  # (V,)
    v_labels[torch.sum(v_votes, dim=-1) == 0] = -1

    # Optional graph-cut smoothing (requires pygco) - improves quality especially for small parts
    if args.smooth_gco:
        try:
            import warnings
            # Fix: np.int was removed in newer NumPy versions - patch it before importing pygco
            if not hasattr(np, 'int'):
                np.int = np.int32  # type: ignore
            if not hasattr(np, 'float'):
                np.float = np.float32  # type: ignore
            
            # Suppress NumPy deprecation warnings from pygco library
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                sys.path.insert(0, str(project_root / "othercode" / "4d-dress" / "4dhumanparsing" / "lib" / "pygco"))
                from lib.pygco import pygco  # type: ignore

            sw = args.gco_smooth_weight
            print(f"[INFO] Applying graph-cut smoothing (smooth_weight={sw})...")
            
            edges_np = np.asarray(edges, dtype=np.int32)
            edge_weights = np.ones(edges_np.shape[0], dtype=np.float32)

            # Boost shoe/hair votes so graph-cut doesn't erase small regions.
            # This lowers their unary cost, making it harder for smoothing to override them.
            shoe_idx_gc = surface_labels.index("shoe") if "shoe" in surface_labels else 2
            hair_idx_gc = surface_labels.index("hair") if "hair" in surface_labels else 1
            boost = 5.0
            v_votes_boosted = v_votes_normalized.clone()
            v_votes_boosted[:, shoe_idx_gc] *= boost
            v_votes_boosted[:, hair_idx_gc] *= boost
            row_sums = v_votes_boosted.sum(dim=1, keepdim=True).clamp(min=1e-8)
            v_votes_boosted = v_votes_boosted / row_sums

            pre_smooth_labels = np.argmax(v_votes_boosted.detach().cpu().numpy(), axis=1)
            pre_shoe = np.sum(pre_smooth_labels == shoe_idx_gc)
            pre_hair = np.sum(pre_smooth_labels == hair_idx_gc)
            print(f"  Pre-smoothing (after {boost}x shoe/hair boost): shoe={pre_shoe}, hair={pre_hair}")

            unarys = -torch.log(v_votes_boosted + 1e-8).detach().cpu().numpy().astype(np.float32)

            smooth_weights = (1 - np.eye(nl, dtype=np.float32)) * sw
            
            # Suppress warnings during graph-cut computation
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                labels_np = pygco.cut_general_graph(
                    edges_np,
                    edge_weights,
                    unarys,
                    smooth_weights,
                    n_iter=-1,
                    algorithm="expansion",
                )
            v_labels = torch.tensor(labels_np, dtype=torch.long, device=device)
            post_shoe = np.sum(labels_np == shoe_idx_gc)
            post_hair = np.sum(labels_np == hair_idx_gc)
            print(f"  Post-smoothing: shoe={post_shoe}, hair={post_hair}")
            print("[INFO] Graph-cut smoothing completed")
        except Exception as e:
            print(f"[WARNING] Graph-cut smoothing requested but pygco not available or failed: {e}")
            print(f"  Continuing without smoothing. Install pygco for better quality: pip install pygco")

    v_labels_np = v_labels.detach().cpu().numpy().astype(np.int32)

    # Optional: subject has no bottom clothes — remap "lower" (label 4) to "upper" (label 3).
    # This merges lower-body into the upper clothing mesh. Shoes (label 2) are NOT touched.
    if args.upperbody_only:
        lower_idx = surface_labels.index("lower") if "lower" in surface_labels else 4
        upper_idx = surface_labels.index("upper") if "upper" in surface_labels else 3
        n_remap = np.sum(v_labels_np == lower_idx)
        v_labels_np[v_labels_np == lower_idx] = upper_idx  # lower -> upper
        v_labels = torch.tensor(v_labels_np, dtype=torch.long, device=device)
        shoe_idx = surface_labels.index("shoe") if "shoe" in surface_labels else 2
        n_shoe = np.sum(v_labels_np == shoe_idx)
        print(f"[INFO] upperbody_only: remapped {n_remap} 'lower' vertices to 'upper' (shoes untouched: {n_shoe} shoe vertices)")

    # Optional: subject has bare legs (no pants/skirt). Remap "lower" (dress/pants) to "skin".
    if args.lower_body_as_skin:
        lower_idx = surface_labels.index("lower") if "lower" in surface_labels else 4
        skin_idx = surface_labels.index("skin") if "skin" in surface_labels else 0
        n_remap = np.sum(v_labels_np == lower_idx)
        v_labels_np[v_labels_np == lower_idx] = skin_idx  # lower -> skin
        v_labels = torch.tensor(v_labels_np, dtype=torch.long, device=device)
        print(f"[INFO] lower_body_as_skin: remapped {n_remap} 'lower' vertices to 'skin'")

    # Optional: export per-view labels rendered from vertex labels
    if args.export_view_labels:
        colors_nl = _label_colors(nl)
        view_root = out_dir / args.views_subdir
        
        # Load original full-resolution images if needed
        original_images = None
        if args.export_original_overlays:
            print("[INFO] Loading original full-resolution images for overlays...")
            original_images = _load_images(images_dir, args.num_views, frames=frames)

        for i in range(images.shape[0]):
            frame = frames[i]
            fx, fy, cx, cy, w, h = _get_intrinsics_from_frame(frame, transforms)
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            fx_scaled = fx * scale_x
            fy_scaled = fy * scale_y
            cx_scaled = cx * scale_x
            cy_scaled = cy * scale_y
            w_scaled = image_w
            h_scaled = image_h
            if args.auto_cam_fix:
                flip = np.diag(flip_diag).astype(np.float32)
                R, T = _candidate_RT_from_c2w(c2w, flip=flip, transpose_r=transpose_r)
            else:
                R, T = _ngp_c2w_to_pytorch3d_RT(c2w)

            cams_scaled = PerspectiveCameras(
                focal_length=torch.tensor([[fx_scaled, fy_scaled]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx_scaled, cy_scaled]], dtype=torch.float32, device=device),
                R=torch.tensor(R[None, ...], dtype=torch.float32, device=device),
                T=torch.tensor(T[None, ...], dtype=torch.float32, device=device),
                in_ndc=False,
                image_size=torch.tensor([[h_scaled, w_scaled]], dtype=torch.float32, device=device),
                device=device,
            )
            r_scaled = MeshRasterizer(cameras=cams_scaled, raster_settings=raster_settings)(mesh)
            pix_to_face_scaled = r_scaled.pix_to_face[0, :, :, 0]
            bary_scaled = r_scaled.bary_coords[0, :, :, 0, :]
            labels_hw_scaled = _render_labels_from_vertex_labels(th_faces[0], pix_to_face_scaled, bary_scaled, v_labels, nl)
            labels_i_scaled = labels_hw_scaled.detach().cpu().numpy().astype(np.int32)

            _save_label_and_overlay(
                out_label_png=view_root / "mesh" / f"view{i:03d}_labels.png",
                out_overlay_png=view_root / "mesh" / f"view{i:03d}_overlay.png",
                rgb_image=images[i],
                labels_hw=labels_i_scaled,
                colors=colors_nl,
            )
            
            # Also render at original resolution if requested
            if args.export_original_overlays and original_images is not None:
                H0, W0 = original_images[i].shape[:2]
                if args.auto_cam_fix:
                    flip = np.diag(flip_diag).astype(np.float32)
                    R, T = _candidate_RT_from_c2w(c2w, flip=flip, transpose_r=transpose_r)
                else:
                    R, T = _ngp_c2w_to_pytorch3d_RT(c2w)
                cams_orig = PerspectiveCameras(
                    focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                    principal_point=torch.tensor([[cx, cy]], dtype=torch.float32, device=device),
                    R=torch.tensor(R[None, ...], dtype=torch.float32, device=device),
                    T=torch.tensor(T[None, ...], dtype=torch.float32, device=device),
                    in_ndc=False,
                    image_size=torch.tensor([[H0, W0]], dtype=torch.float32, device=device),
                    device=device,
                )
                if args.raster_bin_size is not None:
                    orig_bin_size = args.raster_bin_size
                elif num_faces > 200000:
                    orig_bin_size = 0
                else:
                    orig_bin_size = None
                orig_raster_settings = RasterizationSettings(
                    image_size=(H0, W0),
                    blur_radius=0.0,
                    faces_per_pixel=1,
                    bin_size=orig_bin_size,
                )
                r_orig = MeshRasterizer(cameras=cams_orig, raster_settings=orig_raster_settings)(mesh)
                pix_to_face_orig = r_orig.pix_to_face[0, :, :, 0]
                bary_orig = r_orig.bary_coords[0, :, :, 0, :]
                labels_hw_orig = _render_labels_from_vertex_labels(th_faces[0], pix_to_face_orig, bary_orig, v_labels, nl)
                labels_i_orig = labels_hw_orig.detach().cpu().numpy().astype(np.int32)
                _save_label_and_overlay(
                    out_label_png=view_root / "original" / f"view{i:03d}_labels.png",
                    out_overlay_png=view_root / "original" / f"view{i:03d}_overlay.png",
                    rgb_image=original_images[i],
                    labels_hw=labels_i_orig,
                    colors=colors_nl,
                )

    # Save labels + a colored mesh + separated submeshes
    np.save(out_dir / "vertex_labels.npy", v_labels_np)

    # Create per-vertex colors for quick inspection
    label_colors = _label_colors(nl)
    colors = np.zeros((V.shape[0], 3), dtype=np.uint8)
    for li in range(nl):
        colors[v_labels_np == li] = label_colors[li]
    # background/unlabeled -> white
    colors[v_labels_np < 0] = np.array([255, 255, 255], dtype=np.uint8)

    colored = trimesh.Trimesh(vertices=V, faces=F, vertex_colors=colors, process=False)
    colored.export(out_dir / "mesh_labeled.obj")

    # Split meshes
    label_meshes = extract_label_meshes(V, F, v_labels_np, surface_labels, colors=colors, uvs=None)
    for name, mdata in label_meshes.items():
        sub = trimesh.Trimesh(vertices=mdata["vertices"], faces=mdata["faces"], process=False)
        sub.export(out_dir / f"{name}.obj")

    # Convenience: combine clothes into one mesh
    clothes_parts = [k for k in ["upper", "lower", "outer"] if k in label_meshes]
    if clothes_parts:
        subs = []
        for k in clothes_parts:
            md = label_meshes[k]
            subs.append(trimesh.Trimesh(vertices=md["vertices"], faces=md["faces"], process=False))
        trimesh.util.concatenate(subs).export(out_dir / "clothes.obj")

    print("## done")
    print("out_dir:", out_dir)
    print("wrote:", "vertex_labels.npy, mesh_labeled.obj, skin.obj/hair.obj/shoe.obj/upper.obj/lower.obj/outer.obj, clothes.obj")


if __name__ == "__main__":
    main()

