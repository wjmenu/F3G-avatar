#!/usr/bin/env python3
"""
Convert dataset_rex SMPL-X params stored in .npz into a PhysAvatar-compatible .pth dict.

Input (npz keys observed in dataset_rex/smpl_params.npz):
  - betas: (1, 10)
  - body_pose: (T, 63)
  - global_orient: (T, 3)
  - transl: (T, 3)
  - left_hand_pose: (T, 45)
  - right_hand_pose: (T, 45)
  - jaw_pose: (T, 3)
  - expression: (T, 10)

Output (.pth dict) keys expected by PhysAvatar SmplxDeformer.smplx_forward:
  - trans (1,3)
  - orient (1,3)
  - body_pose (1,63)
  - beta (1,B)
  - left_hand_pose (1,45)
  - right_hand_pose (1,45)
  - jaw_pose (1,3)
  - expr (1,E)
  - left_eye_pose (1,3) zeros
  - right_eye_pose (1,3) zeros
  - scale (1,) float tensor (default 1.0)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RuntimeError(msg)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--npz", required=True, help="Path to dataset_rex smpl_params.npz")
    p.add_argument("--frame", type=int, default=0, help="Frame index to extract (default 0)")
    p.add_argument("--out_pth", required=True, help="Output .pth path")
    p.add_argument("--scale", type=float, default=1.0, help="Global scale factor (default 1.0)")
    args = p.parse_args()

    npz_path = Path(args.npz)
    out_pth = Path(args.out_pth)
    _require(npz_path.exists(), f"Missing npz: {npz_path}")

    d = np.load(str(npz_path))
    needed = ["betas", "body_pose", "global_orient", "transl", "left_hand_pose", "right_hand_pose", "jaw_pose", "expression"]
    for k in needed:
        _require(k in d.files, f"Missing key '{k}' in {npz_path}. Keys: {d.files}")

    T = d["body_pose"].shape[0]
    frame = int(args.frame)
    _require(0 <= frame < T, f"frame={frame} out of range [0, {T-1}]")

    # Shapes
    betas = d["betas"].astype(np.float32)           # (1,B)
    body_pose = d["body_pose"][frame].astype(np.float32)[None, :]  # (1,63)
    global_orient = d["global_orient"][frame].astype(np.float32)[None, :]  # (1,3)
    transl = d["transl"][frame].astype(np.float32)[None, :]        # (1,3)
    lhand = d["left_hand_pose"][frame].astype(np.float32)[None, :] # (1,45)
    rhand = d["right_hand_pose"][frame].astype(np.float32)[None, :]# (1,45)
    jaw = d["jaw_pose"][frame].astype(np.float32)[None, :]         # (1,3)
    expr = d["expression"][frame].astype(np.float32)[None, :]      # (1,E)

    param = {
        "trans": torch.from_numpy(transl),
        "orient": torch.from_numpy(global_orient),
        "body_pose": torch.from_numpy(body_pose),
        "beta": torch.from_numpy(betas),
        "left_hand_pose": torch.from_numpy(lhand),
        "right_hand_pose": torch.from_numpy(rhand),
        "jaw_pose": torch.from_numpy(jaw),
        "expr": torch.from_numpy(expr),
        "left_eye_pose": torch.zeros((1, 3), dtype=torch.float32),
        "right_eye_pose": torch.zeros((1, 3), dtype=torch.float32),
        "scale": torch.tensor([float(args.scale)], dtype=torch.float32),
    }

    out_pth.parent.mkdir(parents=True, exist_ok=True)
    torch.save(param, str(out_pth))
    print("wrote:", out_pth)
    print("beta_dim:", param["beta"].shape[-1], "expr_dim:", param["expr"].shape[-1])


if __name__ == "__main__":
    main()

