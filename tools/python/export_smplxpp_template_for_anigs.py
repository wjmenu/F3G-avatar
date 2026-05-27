#!/usr/bin/env python3
"""
Export a PhysAvatar-style SMPLX++ (ours) as AniGS "template.ply".

AniGS uses `<data_dir>/template.ply` (if present) when generating canonical position maps
in `gen_data/gen_pos_maps.py`. The canonical pose in AniGS is an A-pose
(`config.cano_smpl_body_pose`), while our SMPLX++ composition currently exports a T-pose mesh.

This script:
  - loads `tpose/template.obj` and `weights/template_weights.npy`
  - computes SMPL-X joint transforms for AniGS's canonical A-pose
  - applies LBS to move the SMPLX++ mesh from T-pose -> A-pose
  - writes a `template.ply` you can drop into your AniGS dataset `data_dir/`
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import trimesh
import sys


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def _load_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    tri = trimesh.load(path, force="mesh", process=False)
    V = np.asarray(tri.vertices, dtype=np.float32)
    F = np.asarray(tri.faces, dtype=np.int64)
    return V, F


def _pose_vertices_with_weights(
    vertices_tpose: torch.Tensor,   # (N,3) or (1,N,3)
    lbs_weights: torch.Tensor,      # (N,J) or (1,N,J)
    transform_mat: torch.Tensor,    # (1,J,4,4) or (J,4,4)
    global_transl: torch.Tensor,    # (1,3) or (3,)
    scale: torch.Tensor,            # scalar tensor
) -> torch.Tensor:
    """
    Apply blended rigid transforms (LBS) to pose vertices from rest (T-pose) into the target pose.
    Matches the inverse operation used in tools/compose_smplxpp_physavatar.py.
    """
    if vertices_tpose.ndim == 2:
        vertices_tpose = vertices_tpose.unsqueeze(0)
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

    # blend transforms per vertex
    T = torch.einsum("bnj,bjkl->bnkl", lbs_weights, transform_mat)  # (1,N,4,4)
    ones = torch.ones_like(vertices_tpose[..., :1])
    v_h = torch.cat([vertices_tpose, ones], dim=-1).unsqueeze(-1)  # (1,N,4,1)
    out = (T @ v_h)[..., :3, 0]  # (1,N,3)

    # apply global scale+translation (consistent with PhysAvatar SmplxDeformer.smplx_forward)
    out = out * scale_
    out = out + global_transl.view(1, 1, 3)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--template_dir", required=True, help="Directory produced by compose_smplxpp_physavatar.py (contains tpose/ and weights/)")
    p.add_argument("--smpl_params_npz", required=True, help="AniGS dataset smpl_params.npz (used to read betas and expression dim)")
    p.add_argument("--out_template_ply", required=True, help="Output template.ply path (usually <data_dir>/template.ply)")
    p.add_argument("--smplx_model_path", required=True, help="Path to SMPL-X model folder (contains SMPLX_*.npz)")
    p.add_argument("--gender", default="neutral", choices=["neutral", "male", "female"])
    p.add_argument("--device", default=None, help="torch device, e.g. cuda:0 or cpu (default: auto)")
    p.add_argument("--save_debug_obj", default=None, help="Optional: also export the A-pose mesh as .obj for quick viewing")
    args = p.parse_args()

    proj_root = Path(__file__).resolve().parents[1]
    # Allow running from anywhere (e.g. `~`) by ensuring repo root is importable.
    # This is required for `import config` (repo-local `config.py`).
    if str(proj_root) not in sys.path:
        sys.path.insert(0, str(proj_root))

    template_dir = Path(args.template_dir)
    smpl_params_npz = Path(args.smpl_params_npz)
    out_ply = Path(args.out_template_ply)
    _require(template_dir.exists(), f"Missing template_dir: {template_dir}")
    _require(smpl_params_npz.exists(), f"Missing smpl_params_npz: {smpl_params_npz}")

    template_obj = template_dir / "tpose" / "template.obj"
    template_w = template_dir / "weights" / "template_weights.npy"
    _require(template_obj.exists(), f"Missing: {template_obj}")
    _require(template_w.exists(), f"Missing: {template_w}")

    V, F = _load_obj(template_obj)
    W = np.load(str(template_w)).astype(np.float32)
    _require(W.shape[0] == V.shape[0], f"Vertex/weight mismatch: V={V.shape[0]} vs W={W.shape[0]}")

    # Load betas (and expression dim, if present)
    d = np.load(str(smpl_params_npz), allow_pickle=True)
    _require("betas" in d.files, f"Missing 'betas' in {smpl_params_npz}. Keys: {d.files}")
    betas = d["betas"].astype(np.float32)  # (1,B)
    expr_dim = int(d["expression"].shape[-1]) if "expression" in d.files else 10

    # Import AniGS canonical A-pose definition
    import config as anigs_config  # repo-local config.py

    # Import PhysAvatar SMPLX deformer (for transform_mat)
    phys_root = proj_root / "othercode" / "PhysAvatar"
    _require(phys_root.exists(), f"Missing PhysAvatar at {phys_root}")
    sys.path.insert(0, str(phys_root))
    sys.path.insert(0, str(phys_root / "utils"))
    from utils.smplx_deformer import SmplxDeformer  # type: ignore

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    deformer = SmplxDeformer(
        gender=args.gender,
        num_betas=int(betas.shape[-1]),
        num_expression_coeffs=int(expr_dim),
        smplx_model_path=str(args.smplx_model_path),
        vposer_ckpt_path=None,  # not needed (we pass axis-angle body_pose)
        device=device,
    )

    # Build an SMPL-X param dict for the AniGS canonical A-pose.
    # Note: we keep transl/orient/scale = identity, and set all face/hand params to zero.
    smplx_param_apose = {
        "trans": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "orient": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "body_pose": anigs_config.cano_smpl_body_pose.to(device=device, dtype=torch.float32)[None],  # (1,63)
        "beta": torch.from_numpy(betas).to(device=device, dtype=torch.float32),
        "left_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
        "right_hand_pose": torch.zeros((1, 45), dtype=torch.float32, device=device),
        "jaw_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "expr": torch.zeros((1, expr_dim), dtype=torch.float32, device=device),
        "left_eye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "right_eye_pose": torch.zeros((1, 3), dtype=torch.float32, device=device),
        "scale": torch.tensor([1.0], dtype=torch.float32, device=device),
    }

    with torch.no_grad():
        smplx_out = deformer.smplx_forward(smplx_param_apose)
        transform_mat = smplx_out.transform_mat  # (1,J,4,4)

    V_t = torch.from_numpy(V).to(device=device, dtype=torch.float32)
    W_t = torch.from_numpy(W).to(device=device, dtype=torch.float32)

    # LBS into AniGS canonical A-pose
    V_apose = _pose_vertices_with_weights(
        vertices_tpose=V_t,
        lbs_weights=W_t,
        transform_mat=transform_mat,
        global_transl=smplx_param_apose["trans"],
        scale=smplx_param_apose["scale"],
    )[0].detach().cpu().numpy().astype(np.float32)

    out_ply.parent.mkdir(parents=True, exist_ok=True)
    trimesh.Trimesh(vertices=V_apose, faces=F, process=False).export(out_ply)
    print("wrote:", out_ply)

    if args.save_debug_obj:
        out_obj = Path(args.save_debug_obj)
        out_obj.parent.mkdir(parents=True, exist_ok=True)
        trimesh.Trimesh(vertices=V_apose, faces=F, process=False).export(out_obj)
        print("wrote:", out_obj)


if __name__ == "__main__":
    main()

