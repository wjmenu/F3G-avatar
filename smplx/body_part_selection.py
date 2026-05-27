# -*- coding: utf-8 -*-
"""
Arm/hand vertex selection for SMPL-X using LBS (linear blend skinning) weights.
Vertices are assigned to the joint that has the maximum weight; we keep only
those whose dominant joint is an arm or hand joint (excluding head, neck, torso).
SMPL-X joint indices from joint_names.py: 16-21 = arms, 25-54 = hands.
"""

from __future__ import annotations

import numpy as np

# SMPL-X JOINT_NAMES indices: 16 left_shoulder, 17 right_shoulder, 18 left_elbow,
# 19 right_elbow, 20 left_wrist, 21 right_wrist; 22 jaw, 23-24 eyes; 25-39 left hand,
# 40-54 right hand. So arm+hand = 16-21 and 25-54 (excludes head=15, neck=12, etc.).
ARM_HAND_JOINT_INDICES = tuple(range(16, 22)) + tuple(range(25, 55))


def get_arm_hand_vertex_mask_from_lbs(lbs_weights) -> np.ndarray:
    """
    Return a boolean mask of shape (num_vertices,) where True = arm or hand vertex.

    Uses the LBS weight matrix: for each vertex, the joint with maximum weight
    determines body part. Only vertices dominated by arm/hand joints (16-21, 25-54)
    are marked True, so head, neck, and upper torso are excluded.

    Parameters
    ----------
    lbs_weights : array-like, shape (V, J+1)
        Linear blend skinning weights from the SMPL-X model (e.g. model.lbs_weights).

    Returns
    -------
    mask : np.ndarray, shape (V,), dtype=bool
    """
    w = np.asarray(lbs_weights)
    if w.ndim == 3:
        w = w[0]
    # (V,) index of dominant joint per vertex
    dominant_joint = np.argmax(w, axis=1)
    return np.isin(dominant_joint, ARM_HAND_JOINT_INDICES)
