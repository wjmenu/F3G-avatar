#!/usr/bin/env python3
"""
Compose a personalized SMPLX++-style model from:
  - a posed reference-frame SMPLX parameter dict (.pth)
  - segmented non-body meshes (hair / clothes / shoe) in the SAME posed space

Steps (mirrors the paper + PhysAvatar tooling):
  1) Load SMPLX reference parameters and run SMPLX forward (posed body).
  2) For each non-body component:
     - estimate skinning weights via Robust Skin Weights Transfer (weight inpainting),
       using SMPLX as the source and the component mesh as the target
     - canonicalize to T-pose by applying inverse blended rigid transforms,
       then convert T-pose -> A-pose so outputs match the project canonical pose.
  3) Export (all in canonical A-pose):
     - per-component canonical meshes (.obj)
     - per-component LBS weights (.npy)
     - combined canonical mesh (.obj) + combined weights (.npy)

NOTE:
  - You MUST provide SMPLX model files (SMPLX .npz) via --smplx_model_path.
  - VPoser is optional unless your smplx_param contains a 'latent' key.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _load_obj(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    tri = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(tri.vertices, dtype=np.float32)
    F = np.asarray(tri.faces, dtype=np.int64)
    return V, F


def _save_obj(path: Path, V: np.ndarray, F: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tri = trimesh.Trimesh(vertices=V, faces=F, process=False)
    tri.export(path)


def _vertex_normals(V: np.ndarray, F: np.ndarray) -> np.ndarray:
    tri = trimesh.Trimesh(vertices=V, faces=F, process=False)
    vn = np.asarray(tri.vertex_normals, dtype=np.float32)
    # safety for degenerate meshes
    vn[~np.isfinite(vn).all(axis=1)] = 0.0
    return vn


def _unpose_vertices_with_weights(
    vertices_world: torch.Tensor,        # (N,3) or (1,N,3)
    lbs_weights: torch.Tensor,           # (N,J) or (1,N,J)
    transform_mat: torch.Tensor,         # (1,J,4,4) or (J,4,4)
    global_transl: torch.Tensor,         # (1,3) or (3,)
    scale: torch.Tensor,                # (1,) or (1,1) or scalar tensor
) -> torch.Tensor:
    """
    Canonicalize posed vertices into T-pose space by:
      - undo global scale and translation
      - apply per-vertex inverse blended transform (from LBS weights + SMPLX transform_mat)
    """
    if vertices_world.ndim == 2:
        vertices_world = vertices_world.unsqueeze(0)
    if lbs_weights.ndim == 2:
        lbs_weights = lbs_weights.unsqueeze(0)
    if transform_mat.ndim == 3:
        transform_mat = transform_mat.unsqueeze(0)
    if global_transl.ndim == 1:
        global_transl = global_transl.unsqueeze(0)

    # normalize scale shape
    if scale.numel() == 1:
        scale_ = scale.view(1, 1, 1)
    else:
        scale_ = scale.view(1, 1, -1)
        _require(scale_.shape[-1] == 1, f"Expected scalar scale, got shape {tuple(scale.shape)}")

    v = vertices_world / scale_
    v = v - global_transl.view(1, 1, 3)

    # blend transforms
    # lbs_weights: (1,N,J), transform_mat: (1,J,4,4)
    T = torch.einsum("bnj,bjkl->bnkl", lbs_weights, transform_mat)  # (1,N,4,4)
    T_inv = torch.inverse(T)

    ones = torch.ones_like(v[..., :1])
    v_h = torch.cat([v, ones], dim=-1).unsqueeze(-1)  # (1,N,4,1)
    out = (T_inv @ v_h)[..., :3, 0]  # (1,N,3)
    return out


def _tpose_to_apose(
    vertices_t: torch.Tensor,  # (N, 3)
    lbs_weights: torch.Tensor,  # (N, J)
    T_T2A: torch.Tensor,  # (J, 4, 4) joint transforms T-pose -> A-pose
    device: torch.device,
) -> torch.Tensor:
    """Transform vertices from T-pose to canonical A-pose using LBS weights."""
    v_t = vertices_t.to(device) if vertices_t.device != device else vertices_t
    w = lbs_weights.to(device) if lbs_weights.device != device else lbs_weights
    if v_t.ndim == 2:
        v_t = v_t.unsqueeze(0)
    if w.ndim == 2:
        w = w.unsqueeze(0)
    ones = torch.ones_like(v_t[:, :, :1])
    v_t_h = torch.cat([v_t, ones], dim=-1)  # (1, N, 4)
    per_point_T = torch.einsum("bnj,jxy->bnxy", w, T_T2A)  # (1, N, 4, 4)
    v_a_h = torch.einsum("bnxy,bny->bnx", per_point_T, v_t_h)  # (1, N, 4)
    return v_a_h[0, :, :3]


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Compute similarity transform (s, R, t) such that:
      dst ≈ s * (R @ src) + t
    using Umeyama alignment (with scale).
    src, dst: (N,3)
    """
    _require(src.shape == dst.shape and src.shape[1] == 3, "src/dst must be (N,3) and same shape")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    X = src - mu_src
    Y = dst - mu_dst
    cov = (Y.T @ X) / float(src.shape[0])
    U, S, Vt = np.linalg.svd(cov)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    var_src = (X * X).sum() / float(src.shape[0])
    s = float(S.sum() / (var_src + 1e-12))
    t = mu_dst - s * (R @ mu_src)
    return s, R.astype(np.float32), t.astype(np.float32)


def _fit_similarity_icp(
    src: np.ndarray,
    dst: np.ndarray,
    n_iters: int = 5,
    sample: int = 5000,
    seed: int = 0,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    ICP-like similarity fit using nearest neighbors + Umeyama. Returns (s,R,t) mapping src->dst.
    """
    rng = np.random.RandomState(seed)
    src_idx = np.arange(src.shape[0])
    dst_tree = cKDTree(dst)

    # start with identity
    s_tot = 1.0
    R_tot = np.eye(3, dtype=np.float32)
    t_tot = np.zeros(3, dtype=np.float32)

    for _ in range(max(1, n_iters)):
        if sample is not None and sample < src.shape[0]:
            pick = rng.choice(src_idx, size=sample, replace=False)
            src_s = src[pick]
        else:
            src_s = src

        # transform current src
        src_t = (s_tot * (src_s @ R_tot.T)) + t_tot[None, :]
        _, nn = dst_tree.query(src_t, k=1)
        dst_s = dst[nn]

        s, R, t = _umeyama_similarity(src_s, dst_s)

        # compose: new_total(x) = s*(R*x)+t applied to old_total(x)
        # old_total(x) = s_tot*(R_tot*x)+t_tot
        # => new_total(x) = (s*s_tot) * ((R @ R_tot) x) + (s*(R @ t_tot) + t)
        t_tot = (s * (R @ t_tot)) + t
        R_tot = (R @ R_tot).astype(np.float32)
        s_tot = float(s * s_tot)

    return float(s_tot), R_tot.astype(np.float32), t_tot.astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seg_dir", required=True, help="Directory containing segmented meshes (hair.obj, clothes.obj, shoe.obj)")
    p.add_argument("--smplx_param_pth", required=True, help="Reference-frame SMPLX parameter dict (.pth)")
    p.add_argument("--out_dir", required=True, help="Output directory for final template (tpose/template.obj, weights/template_weights.npy)")
    p.add_argument("--work_dir", default=None, help="Directory for intermediate compose outputs (debug, per-component meshes). Defaults to out_dir.")
    p.add_argument("--fit_to_body_obj", default=None, help="Optional: OBJ mesh to fit SMPLX reference pose into (e.g., skin.obj). "
                                                           "If provided, we estimate a similarity transform and bake it into trans/orient/scale.")
    p.add_argument("--fit_iters", type=int, default=5)
    p.add_argument("--fit_sample", type=int, default=5000)

    p.add_argument("--smplx_model_path", required=True, help="Path to SMPLX model folder (contains SMPLX_*.npz)")
    p.add_argument("--vposer_ckpt_path", default=None, help="Optional VPoser checkpoint (TR00_E096.pt). Required only if smplx_param uses 'latent'.")
    p.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    p.add_argument("--num_betas", type=int, default=None, help="Override SMPLX num_betas. Defaults to len(beta) from the param file.")
    p.add_argument("--num_expression_coeffs", type=int, default=None, help="Override SMPLX num_expression_coeffs. Defaults to len(expr) from the param file.")

    p.add_argument("--max_distance_ratio", type=float, default=0.05, help="Robust skinning transfer distance threshold ratio (bbox diag * ratio)")
    p.add_argument("--max_angle_deg", type=float, default=15.0, help="Robust skinning transfer normal angle threshold (degrees)")
    p.add_argument("--weights_mode", default="inpaint", choices=["inpaint", "nn"], help="How to compute garment LBS weights: "
                                                                                       "'inpaint' uses Robust Skin Weights Transfer; "
                                                                                       "'nn' uses nearest-neighbor transfer only (stable baseline).")
    args = p.parse_args()

    # Repo root (this file lives in tools/python/...)
    project_root = Path(__file__).resolve().parents[2]
    phys_root = project_root / "othercode" / "PhysAvatar"
    _require(phys_root.exists(), f"Missing PhysAvatar at {phys_root}")

    # Import PhysAvatar tools
    import sys

    sys.path.insert(0, str(phys_root))
    sys.path.insert(0, str(phys_root / "utils"))
    from utils.smplx_deformer import SmplxDeformer  # type: ignore
    from lbs_weights_inpainting import segregate_vertices_by_confidence, compute_weights_for_remaining_vertices  # type: ignore

    seg_dir = Path(args.seg_dir)
    out_dir = Path(args.out_dir)
    work_dir = Path(args.work_dir) if args.work_dir else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "tpose").mkdir(parents=True, exist_ok=True)
    (out_dir / "weights").mkdir(parents=True, exist_ok=True)
    (work_dir / "debug").mkdir(parents=True, exist_ok=True)
    (work_dir / "tpose").mkdir(parents=True, exist_ok=True)
    (work_dir / "weights").mkdir(parents=True, exist_ok=True)

    # Load SMPLX params
    smplx_param = torch.load(args.smplx_param_pth, map_location="cuda" if torch.cuda.is_available() else "cpu")
    _require("trans" in smplx_param and "orient" in smplx_param and "beta" in smplx_param, "smplx_param missing required keys (trans/orient/beta)")
    if "scale" not in smplx_param:
        smplx_param["scale"] = torch.tensor([1.0], device=smplx_param["trans"].device, dtype=smplx_param["trans"].dtype)

    # Ensure tensors are on device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smplx_param = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in smplx_param.items()}

    # Init SMPLX deformer (PhysAvatar)
    num_betas = args.num_betas
    if num_betas is None:
        num_betas = int(smplx_param["beta"].shape[-1])
    num_expr = args.num_expression_coeffs
    if num_expr is None:
        num_expr = int(smplx_param["expr"].shape[-1])
    lbs_deformer = SmplxDeformer(
        gender=args.gender,
        num_betas=num_betas,
        num_expression_coeffs=num_expr,
        smplx_model_path=args.smplx_model_path,
        vposer_ckpt_path=args.vposer_ckpt_path,
        device=device,
    )

    # Build T-pose -> A-pose transform so we output canonical A-pose (matches rest of project)
    # Use the *project's* smplx (output has .A); PhysAvatar may have loaded a different smplx from site-packages.
    sys.path.insert(0, str(project_root))
    import config as project_config
    # Force project's smplx to load (it has .A on output); clear cache so next import uses project_root.
    for key in list(sys.modules.keys()):
        if key == "smplx" or key.startswith("smplx."):
            del sys.modules[key]
    import smplx as smplx_project
    smplx_model_path_abs = (project_root / args.smplx_model_path).resolve() if not Path(args.smplx_model_path).is_absolute() else Path(args.smplx_model_path)
    smpl_model_apose = smplx_project.SMPLX(
        str(smplx_model_path_abs),
        gender=args.gender,
        use_pca=False,
        num_pca_comps=45,
        flat_hand_mean=True,
        batch_size=1,
    )
    with torch.no_grad():
        tpose_out = smpl_model_apose.forward(
            betas=torch.zeros(10, dtype=torch.float32)[None],
            global_orient=torch.zeros(3, dtype=torch.float32)[None],
            transl=torch.zeros(3, dtype=torch.float32)[None],
            body_pose=torch.zeros(63, dtype=torch.float32)[None],
        )
        apose_out = smpl_model_apose.forward(
            betas=torch.zeros(10, dtype=torch.float32)[None],
            global_orient=project_config.cano_smpl_global_orient[None],
            transl=project_config.cano_smpl_transl[None],
            body_pose=project_config.cano_smpl_body_pose[None],
        )
        A_T = tpose_out.A[0]
        A_A = apose_out.A[0]
        T_T2A = torch.matmul(A_A, torch.linalg.inv(A_T))  # (J, 4, 4) on CPU
        T_T2A = T_T2A.to(device)
    print("[compose] Canonical pose: A-pose (T->A transform applied to all meshes)")

    # Forward SMPLX in the reference pose
    with torch.no_grad():
        smplx_out = lbs_deformer.smplx_forward(smplx_param)

    smplx_V = smplx_out.vertices[0].detach().cpu().numpy().astype(np.float32)
    smplx_F = np.asarray(lbs_deformer.smplx_model.faces, dtype=np.int64)
    smplx_VN = _vertex_normals(smplx_V, smplx_F)

    # Save posed SMPLX (debug)
    _save_obj(work_dir / "debug" / "smplx_ref_pose.obj", smplx_V, smplx_F)

    # Optional: fit SMPLX to target body mesh (similarity) and bake into params
    fit_info = None
    if args.fit_to_body_obj:
        target_V, _ = _load_obj(Path(args.fit_to_body_obj))
        s_fit, R_fit, t_fit = _fit_similarity_icp(
            src=smplx_V,
            dst=target_V,
            n_iters=int(args.fit_iters),
            sample=int(args.fit_sample),
        )
        # Update param: scale1 = s_fit*scale0 ; R1 = R_fit*R0 ; trans1 = R_fit*trans0 + t_fit/scale1
        # Note: smplx_forward expects 'orient' axis-angle and applies 'trans' before we post-scale outputs.
        import pytorch3d.transforms as p3dt

        scale0 = smplx_param["scale"].view(-1)[0].detach().cpu().numpy().astype(np.float32)
        scale1 = float(s_fit * float(scale0))
        
        if abs(scale1) < 1e-6 or abs(scale1) > 1e6:
            fit_info = None
        else:
            R0 = p3dt.axis_angle_to_matrix(smplx_param["orient"]).detach().cpu().numpy()[0].astype(np.float32)
            R1 = (R_fit @ R0).astype(np.float32)
            orient1 = p3dt.matrix_to_axis_angle(torch.tensor(R1[None, ...], dtype=torch.float32, device=device))

            trans0 = smplx_param["trans"].detach().cpu().numpy()[0].astype(np.float32)
            # Fix: t_fit is already in the target space, don't divide by scale1
            # The transformation should be: trans1 = R_fit @ trans0 + t_fit
            trans1 = (R_fit @ trans0) + t_fit

            smplx_param["scale"] = torch.tensor([scale1], dtype=torch.float32, device=device)
            smplx_param["orient"] = orient1.to(device)
            smplx_param["trans"] = torch.tensor(trans1[None, :], dtype=torch.float32, device=device)

            with torch.no_grad():
                smplx_out = lbs_deformer.smplx_forward(smplx_param)
            smplx_V = smplx_out.vertices[0].detach().cpu().numpy().astype(np.float32)
            _save_obj(work_dir / "debug" / "smplx_ref_pose_fitted.obj", smplx_V, smplx_F)
            fit_info = {"s": float(s_fit), "R": R_fit.tolist(), "t": t_fit.tolist(), "scale0": float(scale0), "scale1": float(scale1)}

    # Canonicalize body itself into T-pose space (using its own LBS weights)
    body_w = torch.tensor(lbs_deformer.smplx_model.lbs_weights.detach().cpu().numpy(), dtype=torch.float32, device=device)  # (V,J)
    t_body = _unpose_vertices_with_weights(
        vertices_world=torch.tensor(smplx_V, dtype=torch.float32, device=device),
        lbs_weights=body_w,
        transform_mat=smplx_out.transform_mat.to(device),
        global_transl=smplx_param["trans"],
        scale=smplx_param["scale"],
    )[0]
    # Convert T-pose -> A-pose so output matches project canonical pose
    a_body = _tpose_to_apose(t_body, body_w, T_T2A, device).detach().cpu().numpy().astype(np.float32)
    _save_obj(work_dir / "tpose" / "smplx_body.obj", a_body, smplx_F)

    # Components we care about for garments
    comp_paths = {
        "hair": seg_dir / "hair.obj",
        "clothes": seg_dir / "clothes.obj",
        "shoe": seg_dir / "shoe.obj",
    }

    comp_results: Dict[str, Dict] = {}
    for name, mp in comp_paths.items():
        if not mp.exists():
            if name == "hair":
                # Hair might not be detected - check if we can extract it from the labeled mesh
                print(f"[compose] WARNING: hair.obj not found. Checking if hair exists in labeled mesh...")
                labeled_mesh = seg_dir / "mesh_labeled.obj"
                if labeled_mesh.exists():
                    try:
                        mesh = trimesh.load(labeled_mesh, force="mesh", process=False)
                        # Hair color in 4d-dress is [255, 128, 0] (orange)
                        # Check if any vertices have this color
                        if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
                            hair_color = np.array([255, 128, 0], dtype=np.uint8)
                            hair_mask = np.all(mesh.visual.vertex_colors[:, :3] == hair_color, axis=1)
                            if np.any(hair_mask):
                                print(f"[compose] Found {np.sum(hair_mask)} hair vertices in labeled mesh. Extracting hair...")
                                hair_verts = mesh.vertices[hair_mask]
                                # Find faces where all 3 vertices are hair
                                face_mask = np.all(hair_mask[mesh.faces], axis=1)
                                if np.any(face_mask):
                                    hair_faces = mesh.faces[face_mask]
                                    # Remap vertex indices
                                    old_to_new = np.full(len(mesh.vertices), -1, dtype=np.int32)
                                    old_to_new[hair_mask] = np.arange(np.sum(hair_mask))
                                    hair_faces_remapped = old_to_new[hair_faces]
                                    hair_mesh = trimesh.Trimesh(vertices=hair_verts, faces=hair_faces_remapped, process=False)
                                    hair_mesh.export(mp)
                                    print(f"[compose] Extracted hair mesh: {len(hair_verts)} vertices, {len(hair_faces_remapped)} faces")
                                else:
                                    print(f"[compose] No complete hair faces found. Skipping hair.")
                                    continue
                            else:
                                print(f"[compose] No hair vertices found in labeled mesh. Skipping hair.")
                                continue
                        else:
                            print(f"[compose] Labeled mesh has no vertex colors. Skipping hair.")
                            continue
                    except Exception as e:
                        print(f"[compose] Failed to extract hair from labeled mesh: {e}. Skipping hair.")
                        continue
                else:
                    print(f"[compose] skip missing component: {name} ({mp})")
                    continue
            else:
                print(f"[compose] skip missing component: {name} ({mp})")
                continue

        Vc, Fc = _load_obj(mp)
        VNc = _vertex_normals(Vc, Fc)

        # Robust skin weights transfer (SMPLX -> component) with weight inpainting
        from lbs_weights_inpainting import find_closest_points  # type: ignore

        # For hair, use canonical SMPLX for better correspondence (hair is far from body in posed space)
        # For other components, use posed SMPLX
        if name == "hair":
            # Use canonical (A-pose) SMPLX body for hair correspondence
            a_body_VN = _vertex_normals(a_body, smplx_F)
            src_mesh = {"vertices": a_body, "normal": a_body_VN}
        else:
            # Use posed SMPLX for clothes/shoes
            src_mesh = {"vertices": smplx_V, "normal": smplx_VN}
        
        tgt_mesh = {"vertices": Vc, "faces": Fc, "normal": VNc}

        distances, closest_idx = find_closest_points(tgt_mesh["vertices"], src_mesh["vertices"])
        smplx_lbs = lbs_deformer.smplx_model.lbs_weights.detach().cpu().numpy().astype(np.float32)  # (V,J)

        if args.weights_mode == "nn":
            # Stable baseline: nearest-neighbor weights for all target vertices.
            known_weights = {i: smplx_lbs[int(closest_idx[i])] for i in range(Vc.shape[0])}
        else:
            # For hair, use more lenient thresholds since it's far from body
            if name == "hair":
                # Hair is typically far from body, so use more lenient distance threshold
                hair_distance_ratio = args.max_distance_ratio * 3.0  # 3x more lenient
                hair_angle_deg = args.max_angle_deg * 2.0  # 2x more lenient
                confident_idx, _ = segregate_vertices_by_confidence(
                    src_mesh, tgt_mesh, threshold_distance=hair_distance_ratio, threshold_angle=hair_angle_deg
                )
            else:
                confident_idx, _ = segregate_vertices_by_confidence(
                    src_mesh, tgt_mesh, threshold_distance=args.max_distance_ratio, threshold_angle=args.max_angle_deg
                )
            known_weights = {i: smplx_lbs[int(closest_idx[i])] for i in confident_idx}
            if len(known_weights) == 0:
                known_weights = {i: smplx_lbs[int(closest_idx[i])] for i in range(Vc.shape[0])}

        if len(known_weights) == Vc.shape[0]:
            # No inpainting needed; we already have weights for every vertex.
            Wopt = np.stack([known_weights[i] for i in range(Vc.shape[0])], axis=0).astype(np.float32)
        else:
            Wopt = compute_weights_for_remaining_vertices(
                target_mesh={"vertices": Vc, "faces": Fc},
                known_weights=known_weights,
            )  # (N,J)
        Wopt = np.nan_to_num(Wopt, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Wopt = np.clip(Wopt, 0.0, 1.0)
        rs = Wopt.sum(axis=1, keepdims=True)
        bad = (rs[:, 0] <= 1e-12)
        if np.any(bad):
            Wopt[bad, :] = 0.0
            Wopt[bad, 0] = 1.0
            rs = Wopt.sum(axis=1, keepdims=True)
        Wopt = Wopt / (rs + 1e-12)
        
        # For hair, boost head joint weights and reduce other joint weights
        # The reference version has 96% head joint weight, so we need to prioritize head
        if name == "hair":
            head_joint_idx = 15
            # Find vertices that are closest to head region in canonical space (A-pose body)
            head_region_mask = (a_body[:, 1] > (a_body[:, 1].max() * 0.7))  # Top 30% of body height; numpy
            if np.any(head_region_mask):
                head_region_verts = a_body[head_region_mask]
                # Compute distances from hair vertices (in posed space) to head region
                # Actually, we need to check which hair vertices correspond to head region
                # Use the closest_idx we already computed
                head_correspondence_mask = np.zeros(len(Vc), dtype=bool)
                for i in range(len(Vc)):
                    closest_smplx_idx = int(closest_idx[i])
                    if closest_smplx_idx < len(t_body) and head_region_mask[closest_smplx_idx]:
                        head_correspondence_mask[i] = True
                
                # Boost head joint for vertices near head region
                head_boost_factor = 3.0  # Multiply head weight by this factor
                for i in range(len(Vc)):
                    if head_correspondence_mask[i] or Wopt[i, head_joint_idx] > 0.1:
                        # This vertex is near head or already has some head weight
                        # Boost head weight, reduce others proportionally
                        current_head = Wopt[i, head_joint_idx]
                        other_sum = 1.0 - current_head
                        if other_sum > 1e-6:
                            # Boost head to at least 0.8, or by boost_factor if current is high
                            target_head = min(0.95, max(0.8, current_head * head_boost_factor))
                            scale_others = (1.0 - target_head) / other_sum
                            Wopt[i, :] *= scale_others
                            Wopt[i, head_joint_idx] = target_head
                            # Renormalize
                            Wopt[i, :] /= Wopt[i, :].sum()
            
            # Final pass: ensure most hair vertices have high head weight
            # For vertices with low head weight, transfer weight from other joints to head
            low_head_mask = Wopt[:, head_joint_idx] < 0.5
            if np.any(low_head_mask):
                for i in np.where(low_head_mask)[0]:
                    # Transfer 70% of non-head weight to head
                    other_weight = 1.0 - Wopt[i, head_joint_idx]
                    transfer = other_weight * 0.7
                    Wopt[i, head_joint_idx] += transfer
                    # Reduce other joints proportionally
                    other_mask = np.ones(Wopt.shape[1], dtype=bool)
                    other_mask[head_joint_idx] = False
                    other_sum = Wopt[i, other_mask].sum()
                    if other_sum > 1e-6:
                        Wopt[i, other_mask] *= (1.0 - transfer) / other_sum
                # Renormalize
                rs = Wopt.sum(axis=1, keepdims=True)
                Wopt = Wopt / (rs + 1e-12)
            
            print(f"  After hair weight adjustment: head joint={Wopt[:, head_joint_idx].mean():.4f}, vertices with >0.5 head weight={(Wopt[:, head_joint_idx] > 0.5).sum()}/{len(Wopt)} ({100*(Wopt[:, head_joint_idx] > 0.5).sum()/len(Wopt):.1f}%)")

        Wopt_t = torch.tensor(Wopt, dtype=torch.float32, device=device)
        t_comp = _unpose_vertices_with_weights(
            vertices_world=torch.tensor(Vc, dtype=torch.float32, device=device),
            lbs_weights=Wopt_t,
            transform_mat=smplx_out.transform_mat.to(device),
            global_transl=smplx_param["trans"],
            scale=smplx_param["scale"],
        )[0]
        # Convert T-pose -> A-pose
        a_comp = _tpose_to_apose(t_comp, Wopt_t, T_T2A, device).detach().cpu().numpy().astype(np.float32)

        # Save per-component canonical mesh (A-pose) + weights
        _save_obj(work_dir / "tpose" / f"{name}.obj", a_comp, Fc)
        np.save(work_dir / "weights" / f"{name}_lbs_weights.npy", Wopt.astype(np.float32))

        comp_results[name] = {"V": a_comp, "F": Fc, "W": Wopt.astype(np.float32)}

    # Compose SMPLX++ in A-pose space (canonical pose used by rest of project)
    all_V = [a_body]
    all_F = [smplx_F]
    all_W = [body_w.detach().cpu().numpy().astype(np.float32)]
    offsets = {"smplx_body": 0}
    v_off = t_body.shape[0]
    for name, d in comp_results.items():
        offsets[name] = int(v_off)
        all_V.append(d["V"])
        all_F.append(d["F"] + v_off)
        all_W.append(d["W"])
        v_off += d["V"].shape[0]

    Vc_all = np.concatenate(all_V, axis=0)
    Fc_all = np.concatenate(all_F, axis=0)
    Wc_all = np.concatenate(all_W, axis=0)

    # Check and fill holes in the final composed mesh
    print("\n[FINAL CHECK] Verifying final SMPLX++ mesh integrity...")
    mesh_final = trimesh.Trimesh(vertices=Vc_all, faces=Fc_all, process=False)
    print(f"  Final mesh: {len(mesh_final.vertices)} vertices, {len(mesh_final.faces)} faces")
    
    # Check for holes
    if hasattr(mesh_final, 'is_watertight'):
        is_watertight = mesh_final.is_watertight
        print(f"  Mesh is watertight: {is_watertight}")
        if not is_watertight:
            print("  [FIXING] Filling holes in final mesh...")
            n_verts_before = len(mesh_final.vertices)
            mesh_final.fill_holes()
            mesh_final.remove_duplicate_faces()
            mesh_final.remove_unreferenced_vertices()
            n_verts_after = len(mesh_final.vertices)
            
            # Re-check
            is_watertight_after = mesh_final.is_watertight if hasattr(mesh_final, 'is_watertight') else False
            print(f"  Mesh is watertight after fixing: {is_watertight_after}")
            print(f"  Vertices: {n_verts_before} -> {n_verts_after}")
            
            if n_verts_after != n_verts_before:
                # If vertices were removed, we need to update weights
                # trimesh.remove_unreferenced_vertices() returns a mapping
                # For now, we'll keep the original weights and just truncate if vertices were removed
                if n_verts_after < n_verts_before:
                    print(f"  [NOTE] {n_verts_before - n_verts_after} unreferenced vertices removed. Truncating weights accordingly.")
                    Wc_all = Wc_all[:n_verts_after]
                else:
                    # If vertices were added (unlikely with remove_unreferenced_vertices), we'd need to interpolate weights
                    # For now, just warn
                    print(f"  [WARNING] Vertex count increased. New vertices will have zero weights (may cause issues).")
                    if n_verts_after > len(Wc_all):
                        # Pad with zeros (not ideal, but better than error)
                        pad_size = n_verts_after - len(Wc_all)
                        Wc_all = np.concatenate([Wc_all, np.zeros((pad_size, Wc_all.shape[1]), dtype=Wc_all.dtype)], axis=0)
            
            Vc_all = np.asarray(mesh_final.vertices, dtype=np.float32)
            Fc_all = np.asarray(mesh_final.faces, dtype=np.int64)
            print(f"  Updated mesh: {len(Vc_all)} vertices, {len(Fc_all)} faces")
            
            if not is_watertight_after:
                print("  [WARNING] Mesh still has holes after filling. Proceeding anyway...")
    
    _save_obj(out_dir / "tpose" / "template.obj", Vc_all, Fc_all)
    np.save(out_dir / "weights" / "template_weights.npy", Wc_all.astype(np.float32))

    # Save metadata
    meta = {
        "inputs": {
            "seg_dir": str(seg_dir),
            "smplx_param_pth": str(Path(args.smplx_param_pth)),
            "smplx_model_path": str(Path(args.smplx_model_path)),
            "vposer_ckpt_path": str(args.vposer_ckpt_path) if args.vposer_ckpt_path else None,
        },
        "outputs": {
            "template_obj": str(out_dir / "tpose" / "template.obj"),
            "template_weights": str(out_dir / "weights" / "template_weights.npy"),
            "work_dir": str(work_dir),
            "offsets": offsets,
        },
        "notes": "Garment weights via Robust Skin Weights Transfer (inpainting); canonicalization to T-pose then T->A-pose so canonical meshes match project A-pose.",
    }
    if fit_info is not None:
        meta["fit_to_body_obj"] = {"path": str(args.fit_to_body_obj), "fit": fit_info}
    import json

    (work_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print("## done")
    print("out_dir:", out_dir)


if __name__ == "__main__":
    main()

