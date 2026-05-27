#!/usr/bin/env python3
"""
End-to-end pipeline: multiview images + camera transforms + SMPL-X params (+ optional NeuS2) -> SMPLX++ (T-pose).

This script is a thin orchestrator around the existing scripts we added/modified:
  - tools/mesh_semantic_separation_4ddress.py
  - tools/convert_dataset_rex_smpl_params_to_physavatar_pth.py
  - tools/compose_smplxpp_physavatar.py

Optionally, it can run NeuS2 to produce the input mesh (.obj) from your multiview dataset
using othercode/NeuS2/scripts/run.py (requires NeuS2 build + pyngp bindings to be working).

Expected input file structure
-----------------------------
You can create this with create_template.py (from dataset_rex) or provide your own.

  <out_dir>/
    view_images/           # Multiview images, one per camera (e.g. 00000000.jpg .. 00000015.jpg)
    view_masks/            # Optional; used only if you reference them elsewhere
    transforms.json        # NeRF/NGP-style: "frames" with file_path, transform_matrix, fl_x, fl_y, cx, cy, w, h
    smpl_params.npz        # SMPL-X parameter sequence (or smpl_params.json)

  - transforms.json "file_path" must be relative to the directory containing transforms.json.
    Example: if images live in <out_dir>/view_images/, use "view_images/00000000.jpg" etc.,
    so that NeuS2 (which resolves paths relative to the JSON's directory) finds the images.
  - Pipeline args: --images_dir <out_dir>/view_images, --transforms_json <out_dir>/transforms.json,
    --smpl_params_npz <out_dir>/smpl_params.npz, --out_dir <pipeline_output>.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
import json
import trimesh
import numpy as np


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    printable = " ".join([str(c) for c in cmd])
    print("\n[RUN]", printable)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _run_neus2_with_cwd(cmd: list[str], neus2_root: Path) -> None:
    """Run NeuS2 so that working directory is definitely the NeuS2 project root.
    run.py expects cwd to be the NeuS2 directory (output/, configs/, pyngp from build)."""
    neus2_root = neus2_root.resolve()
    # Use shell so we explicitly cd into NeuS2 root (avoids any cwd issues in subprocess)
    cmd_str = " ".join(_shlex_quote(str(c)) for c in cmd)
    shell_cmd = f"cd {_shlex_quote(str(neus2_root))} && {cmd_str}"
    print(f"\n[RUN NeuS2] cwd={neus2_root}")
    print("[RUN]", shell_cmd[:200] + ("..." if len(shell_cmd) > 200 else ""))
    subprocess.run(shell_cmd, shell=True, check=True)


def _shlex_quote(s: str) -> str:
    try:
        import shlex
        return shlex.quote(s)
    except Exception:
        return f"'{s.replace(chr(39), chr(39)+chr(92)+chr(39))}'"


def main() -> None:
    p = argparse.ArgumentParser(
        description="Pipeline: (optional NeuS2) -> 4D-Dress semantic split -> PhysAvatar SMPLX++ composition"
    )

    # Core inputs
    p.add_argument("--images_dir", required=True, help="Directory containing the multiview images (ordered).")
    p.add_argument(
        "--semantic_images_dir",
        default=None,
        help="Optional: directory containing the per-camera view images to use for semantic separation/overlays. "
             "If omitted, uses --images_dir. Useful when --images_dir contains a temporal sequence but transforms.json "
             "frames correspond to camera views.",
    )
    p.add_argument("--transforms_json", required=True, help="NGP/NeRF transforms.json for the multiview cameras.")
    p.add_argument("--smpl_params_npz", required=True, help="SMPL-X params sequence (.npz), e.g. dataset_rex/smpl_params.npz")
    p.add_argument("--out_dir", required=True, help="Output root directory for all pipeline artifacts.")

    # Mesh source (either provided, or generated via NeuS2)
    p.add_argument("--mesh_obj", default=None, help="NeuS2 reconstructed mesh (.obj). If omitted, use --run_neus2.")
    p.add_argument("--run_neus2", action="store_true", help="If set, run NeuS2 to generate the mesh OBJ.")
    p.add_argument(
        "--neus2_root",
        default=str(Path(__file__).resolve().parents[2] / "othercode" / "NeuS2"),
        help="Path to NeuS2 repo root.",
    )
    p.add_argument("--neus2_name", default="neus2_exp", help="Experiment name (NeuS2 output/<name>/...).")
    p.add_argument("--neus2_network", default="base.json", help="NeuS2 network config (under configs/nerf/).")
    p.add_argument("--neus2_steps", type=int, default=100000, help="NeuS2 training steps (also used in output mesh name).")
    p.add_argument("--neus2_marching_res", type=int, default=512, help="NeuS2 marching cubes grid resolution.")
    p.add_argument("--neus2_mesh_thresh", type=float, default=None, help="Optional: NeuS2 marching cubes density threshold.")
    p.add_argument("--neus2_mesh_reduce_ratio", type=float, default=1.0, help="Reduce mesh complexity by lowering marching cubes resolution. 0.5 = ~half mesh (uses ~0.79x resolution). Default: 1.0 (no reduction).")

    # 4D-Dress semantic split options
    p.add_argument("--outfit", default="Outer", choices=["Inner", "Outer"])
    p.add_argument("--num_views", type=int, default=None, help="Optional: only use first N views for parsing/rasterization.")
    p.add_argument("--max_side", type=int, default=512, help="Downscale views for parsing+rasterization (0 disables).")
    p.add_argument("--graphonomy_ckpt_dir", default=None, help="Dir containing Graphonomy inference.pth (scratch-backed).")
    p.add_argument("--raft_models_dir", default=None, help="Optional: RAFT models dir (for verification only).")
    p.add_argument("--sam_ckpt_path", default=None, help="Optional: SAM ckpt path (for verification only).")
    p.add_argument("--smooth_gco", action="store_true", help="Optional: run pygco smoothing if available.")
    p.add_argument("--gco_smooth_weight", type=float, default=5.0, help="Graph-cut pairwise smooth weight. Lower preserves small regions (hair/shoes). Default: 5.0")
    p.add_argument("--auto_cam_fix", action="store_true", default=False, help="Search for camera convention fix (usually not needed; correct convention is now hardcoded).")
    p.add_argument(
        "--upperbody_only",
        action="store_true",
        help="Subject has no bottom clothes (only upper body). Lower-body regions are merged into upper clothing.",
    )
    p.add_argument(
        "--lower_body_as_skin",
        action="store_true",
        help="Subject has bare legs (no pants/skirt). Vertices classified as 'lower' (dress/pants) are reassigned to 'skin'.",
    )

    # SMPL parameter extraction + composition options
    p.add_argument("--ref_frame", type=int, default=0, help="Reference frame index from smpl_params_npz.")
    p.add_argument(
        "--smplx_model_path",
        default=str(Path(__file__).resolve().parents[2] / "smpl_files" / "smplx"),
        help="SMPL-X model folder (contains SMPLX_*.npz).",
    )
    p.add_argument("--vposer_ckpt_path", default=None, help="Optional VPoser checkpoint (TR00_E096.pt).")
    p.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    p.add_argument("--weights_mode", default="inpaint", choices=["nn", "inpaint"], help="Garment LBS weights mode. 'inpaint' is recommended for better garment skinning (uses confidence filtering + inpainting).")
    p.add_argument(
        "--fit_to_skin",
        action="store_true",
        help="If set, run similarity fitting of SMPLX to the segmented skin mesh. (Can be unstable; default off.)",
    )
    p.add_argument("--fit_iters", type=int, default=5)
    p.add_argument("--fit_sample", type=int, default=5000)
    
    # Mesh preprocessing options
    p.add_argument("--skip_hole_fill", action="store_true", help="Skip hole filling step")

    args = p.parse_args()

    # Repo root (this file lives in tools/python/...)
    project_root = Path(__file__).resolve().parents[2]
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    images_dir = Path(args.images_dir)
    transforms_json = Path(args.transforms_json)
    smpl_params_npz = Path(args.smpl_params_npz)
    _require(images_dir.exists(), f"Missing images_dir: {images_dir}")
    _require(transforms_json.exists(), f"Missing transforms_json: {transforms_json}")
    _require(smpl_params_npz.exists(), f"Missing smpl_params_npz: {smpl_params_npz}")

    semantic_images_dir = Path(args.semantic_images_dir) if args.semantic_images_dir else images_dir
    _require(semantic_images_dir.exists(), f"Missing semantic_images_dir: {semantic_images_dir}")

    # (0) Optionally run NeuS2 to generate mesh OBJ
    mesh_obj: Path | None = Path(args.mesh_obj) if args.mesh_obj else None
    if args.run_neus2:
        neus2_root = Path(args.neus2_root).resolve()
        if not neus2_root.exists():
            # Fallback: othercode may live under tools/ (e.g. if moved there)
            alt = project_root / "tools" / "othercode" / "NeuS2"
            if alt.exists():
                neus2_root = alt
                print(f"[INFO] Using NeuS2 at {neus2_root}")
        _require(neus2_root.exists(), f"Missing NeuS2 root: {neus2_root}. Pass --neus2_root or put othercode under repo root or tools/.")
        run_py = neus2_root / "scripts" / "run.py"
        _require(run_py.exists(), f"Missing NeuS2 entrypoint: {run_py}")

        # Calculate marching cubes resolution based on reduction ratio
        # For ~50% mesh reduction: use cube root of ratio (since mesh complexity scales with volume)
        # e.g., 0.5 reduction -> 0.5^(1/3) ≈ 0.79x resolution
        base_res = int(args.neus2_marching_res)
        if args.neus2_mesh_reduce_ratio < 1.0:
            # Scale resolution to achieve target mesh complexity reduction
            # Mesh complexity roughly scales with resolution^3, so to get ratio reduction:
            # new_res = base_res * (ratio)^(1/3)
            # Use a less aggressive reduction to maintain quality - apply reduction more conservatively
            # Instead of cube root, use square root for less aggressive reduction
            reduced_res = int(base_res * np.sqrt(args.neus2_mesh_reduce_ratio))
            # Round to nearest multiple of 16 (NeuS2 requirement)
            reduced_res = ((reduced_res + 8) // 16) * 16
            # Ensure minimum resolution for quality
            if reduced_res < 256:
                print(f"[WARNING] Reduced resolution {reduced_res} is too low, using 256 minimum")
                reduced_res = 256
            print(f"\n[MESH REDUCTION] Reducing mesh complexity by {args.neus2_mesh_reduce_ratio*100:.0f}%")
            print(f"  Base marching cubes resolution: {base_res}")
            print(f"  Reduced marching cubes resolution: {reduced_res}")
            print(f"  (Using sqrt-based reduction for better quality preservation)")
            actual_res = reduced_res
        else:
            actual_res = base_res

        # NeuS2 expects --scene to be the directory containing transforms.json (not the JSON path).
        # It then sets output to <scene>/output/...; if we passed the JSON path we'd get .../transforms.json/output.
        scene_dir = transforms_json.resolve().parent
        # NeuS2 writes mesh under <scene_dir>/output/<name>/mesh/<n_steps>.obj when using testbed; run.py uses output/<name>/mesh
        out_mesh = neus2_root / "output" / args.neus2_name / "mesh" / f"{int(args.neus2_steps)}.obj"
        cmd = [
            sys.executable,
            str(run_py),
            "--scene",
            str(scene_dir),
            "--name",
            str(args.neus2_name),
            "--network",
            str(args.neus2_network),
            "--n_steps",
            str(int(args.neus2_steps)),
            "--save_mesh",
            "--marching_cubes_res",
            str(actual_res),
            "--no_tensorboard",
        ]
        if args.neus2_mesh_thresh is not None:
            cmd += ["--mesh_thresh", str(float(args.neus2_mesh_thresh))]

        # Debug: what paths does NeuS2 expect and do they exist?
        try:
            transforms_data = json.loads(transforms_json.read_text())
            frames = transforms_data.get("frames", [])
            scene_dir_resolved = scene_dir.resolve()
            print("\n[DEBUG NeuS2 image paths]")
            print(f"  scene_dir (transforms.json parent): {scene_dir_resolved}")
            print(f"  NeuS2 resolves each file_path relative to scene_dir.")
            print(f"  Number of frames in transforms.json: {len(frames)}")
            if frames:
                for idx, frame in enumerate(frames[:5]):
                    fp = frame.get("file_path", "")
                    resolved = scene_dir_resolved / fp
                    exists = resolved.exists()
                    print(f"  frame[{idx}] file_path={fp!r} -> {resolved} exists={exists}")
                if len(frames) > 5:
                    print(f"  ... and {len(frames) - 5} more frames")
                missing = [frame.get("file_path") for frame in frames if not (scene_dir_resolved / frame.get("file_path", "")).exists()]
                if missing:
                    print(f"  MISSING ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
                else:
                    print("  All resolved paths exist.")
        except Exception as e:
            print(f"\n[DEBUG NeuS2] Could not check paths: {e}")

        # NeuS2 run.py must be run with cwd = NeuS2 project root (output/, configs/, pyngp, etc.)
        _run_neus2_with_cwd(cmd, neus2_root)
        _require(out_mesh.exists(), f"NeuS2 completed but mesh not found at: {out_mesh}")
        mesh_obj = out_mesh

    _require(mesh_obj is not None, "No mesh provided. Pass --mesh_obj ... or set --run_neus2.")
    _require(mesh_obj.exists(), f"Missing mesh_obj: {mesh_obj}")

    # (0.5) Mesh preprocessing: hole filling and optional decimation
    print("\n[PREPROCESS] Loading and processing mesh...")
    mesh = trimesh.load(str(mesh_obj), force="mesh", process=False)
    print(f"  NeuS2 mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Optional mesh decimation (if using existing mesh and reduction requested)
    if args.neus2_mesh_reduce_ratio < 1.0 and not args.run_neus2:
        target_faces = int(len(mesh.faces) * args.neus2_mesh_reduce_ratio)
        print(f"\n[MESH REDUCTION] Decimating mesh to {args.neus2_mesh_reduce_ratio*100:.0f}% complexity")
        print(f"  Original: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
        print(f"  Target: ~{target_faces} faces")
        
        # Try open3d method first, fallback to sampling if not available
        try:
            mesh_decimated = mesh.simplify_quadratic_decimation(face_count=target_faces)
        except (ModuleNotFoundError, AttributeError, Exception) as e:
            print(f"  Note: open3d not available, using vertex sampling instead ({e})")
            # Simple approach: sample faces uniformly
            import numpy as np
            face_indices = np.linspace(0, len(mesh.faces)-1, min(target_faces, len(mesh.faces)), dtype=np.int32)
            sampled_faces = mesh.faces[face_indices]
            # Get unique vertices
            unique_verts = np.unique(sampled_faces.flatten())
            # Remap face indices
            vert_map = {old: new for new, old in enumerate(unique_verts)}
            remapped_faces = np.array([[vert_map[v] for v in face] for face in sampled_faces], dtype=np.int32)
            mesh_decimated = trimesh.Trimesh(vertices=mesh.vertices[unique_verts], faces=remapped_faces, process=False)
        print(f"  Decimated: {len(mesh_decimated.vertices)} vertices, {len(mesh_decimated.faces)} faces")
        mesh = mesh_decimated
    
    # Fill holes and clean mesh
    if not args.skip_hole_fill:
        print("  Filling holes and cleaning mesh...")
        mesh_filled = mesh.copy()
        
        # First pass: basic cleaning
        mesh_filled.remove_duplicate_faces()
        mesh_filled.remove_unreferenced_vertices()
        mesh_filled.remove_degenerate_faces()
        
        # Fill holes
        mesh_filled.fill_holes()
        
        # Check if holes were filled
        if hasattr(mesh_filled, 'is_watertight'):
            is_watertight = mesh_filled.is_watertight
            print(f"  Mesh is watertight: {is_watertight}")
            if not is_watertight:
                print("  [WARNING] Mesh still has holes after initial filling. Attempting additional processing...")
                # Try more aggressive hole filling
                mesh_filled.fill_holes()
                mesh_filled.remove_duplicate_faces()
                mesh_filled.remove_unreferenced_vertices()
                mesh_filled.remove_degenerate_faces()
                
                # Try to fix non-manifold edges by merging close vertices
                try:
                    # Merge vertices that are very close (helps with small holes)
                    mesh_filled.process(validate=False)
                except:
                    pass
                
                mesh_filled.fill_holes()
                is_watertight_after = mesh_filled.is_watertight if hasattr(mesh_filled, 'is_watertight') else False
                print(f"  Mesh is watertight after additional processing: {is_watertight_after}")
        mesh = mesh_filled
    
    # Save preprocessed mesh (after hole filling)
    preprocessed_mesh_obj = out_root / "00_preprocessed_mesh.obj"
    mesh.export(str(preprocessed_mesh_obj))
    print(f"  Saved preprocessed mesh to: {preprocessed_mesh_obj}")
    print(f"  Final preprocessed mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    mesh_obj = preprocessed_mesh_obj  # Use preprocessed mesh for subsequent steps

    # (1) Semantic separation (4D-Dress)
    seg_dir = out_root / "semantic_split"
    seg_dir.mkdir(parents=True, exist_ok=True)
    semantic_py = project_root / "tools" / "python" / "mesh_semantic_separation_4ddress.py"
    _require(semantic_py.exists(), f"Missing: {semantic_py}")

    cmd = [
        sys.executable,
        str(semantic_py),
        "--mesh_obj",
        str(mesh_obj),
        "--transforms_json",
        str(transforms_json),
        "--images_dir",
        str(semantic_images_dir),
        "--out_dir",
        str(seg_dir),
        "--outfit",
        str(args.outfit),
        "--max_side",
        str(int(args.max_side)),
        "--export_view_labels",
        "--export_view_parser",
        "--export_original_overlays",  # Export overlays on original full-resolution images
        "--smooth_gco",  # Enable graph-cut smoothing by default for better quality
        "--gco_smooth_weight",
        str(args.gco_smooth_weight),
    ]
    if args.upperbody_only:
        cmd += ["--upperbody_only"]
    if args.lower_body_as_skin:
        cmd += ["--lower_body_as_skin"]
    if args.semantic_images_dir:
        print(f"[INFO] Using semantic_images_dir for parsing/overlays: {semantic_images_dir}")
    # Automatically limit to number of cameras in transforms.json if not specified
    if args.num_views is None:
        transforms_data = json.loads(transforms_json.read_text())
        num_cameras = len(transforms_data.get("frames", []))
        if num_cameras > 0:
            cmd += ["--num_views", str(num_cameras)]
            print(f"[INFO] Auto-limiting to {num_cameras} views (number of cameras in transforms.json)")
    else:
        cmd += ["--num_views", str(int(args.num_views))]
    if args.graphonomy_ckpt_dir:
        cmd += ["--graphonomy_ckpt_dir", str(args.graphonomy_ckpt_dir)]
    if args.raft_models_dir:
        cmd += ["--raft_models_dir", str(args.raft_models_dir)]
    if args.sam_ckpt_path:
        cmd += ["--sam_ckpt_path", str(args.sam_ckpt_path)]
    if args.auto_cam_fix:
        cmd += ["--auto_cam_fix"]
    _run(cmd, cwd=project_root)

    # (2) Convert SMPL params to PhysAvatar .pth for the reference frame
    compose_work_dir = out_root / "_compose_work"
    compose_work_dir.mkdir(parents=True, exist_ok=True)
    template_dir = out_root / "template"
    template_dir.mkdir(parents=True, exist_ok=True)
    smpl_ref_pth = compose_work_dir / f"smplx_ref_frame{int(args.ref_frame):08d}.pth"
    convert_py = project_root / "tools" / "python" / "convert_dataset_rex_smpl_params_to_physavatar_pth.py"
    _require(convert_py.exists(), f"Missing: {convert_py}")
    _run(
        [
            sys.executable,
            str(convert_py),
            "--npz",
            str(smpl_params_npz),
            "--frame",
            str(int(args.ref_frame)),
            "--out_pth",
            str(smpl_ref_pth),
        ],
        cwd=project_root,
    )
    _require(smpl_ref_pth.exists(), f"Failed to create SMPLX ref pth: {smpl_ref_pth}")

    # (3) Compose SMPLX++ in canonical A-pose (PhysAvatar utilities)
    compose_py = project_root / "tools" / "python" / "compose_smplxpp_physavatar.py"
    _require(compose_py.exists(), f"Missing: {compose_py}")
    fit_to_body = seg_dir / "skin.obj"
    cmd = [
        sys.executable,
        str(compose_py),
        "--seg_dir",
        str(seg_dir),
        "--smplx_param_pth",
        str(smpl_ref_pth),
        "--out_dir",
        str(template_dir),
        "--work_dir",
        str(compose_work_dir),
        "--smplx_model_path",
        str(args.smplx_model_path),
        "--gender",
        str(args.gender),
        "--weights_mode",
        str(args.weights_mode),
    ]
    # Optional: fit SMPLX to scan (disabled by default - revert to yesterday's method)
    if args.vposer_ckpt_path:
        cmd += ["--vposer_ckpt_path", str(args.vposer_ckpt_path)]
    # Disabled: --fit_to_body_obj (reverted to yesterday's method without fitting)
    # if args.fit_to_skin and fit_to_body.exists():
    #     cmd += [
    #         "--fit_to_body_obj",
    #         str(fit_to_body),
    #         "--fit_iters",
    #         str(int(args.fit_iters)),
    #         "--fit_sample",
    #         str(int(args.fit_sample)),
    #     ]
    _run(cmd, cwd=project_root)

    template_obj = template_dir / "tpose" / "template.obj"
    template_weights = template_dir / "weights" / "template_weights.npy"
    _require(template_obj.exists(), f"Pipeline finished but missing output: {template_obj}")
    _require(template_weights.exists(), f"Pipeline finished but missing output: {template_weights}")

    print("\n[DONE]")
    print("semantic split dir:", seg_dir)
    print("template dir      :", template_dir)
    print("template mesh     :", template_obj)
    print("template weights  :", template_weights)


if __name__ == "__main__":
    main()

