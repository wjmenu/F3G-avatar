"""
Image utilities (interpolation, masking, etc.).
"""
import torch
import torch.nn.functional as F


def interpolate_image_masked(
    x,
    target_h,
    target_w,
    eps=0.01,
    threshold=0.5,
    fill=0.0,
):
    """
    Bilinearly interpolate a (H, W, C) image and mask out invalid (near-zero) regions.

    Invalid pixels are those with L2 norm <= eps. They are interpolated normally
    but then replaced with `fill` where the interpolated validity mask is below
    `threshold`. Works for any value range (e.g. [0,1] or [-1,1]); set `fill`
    accordingly.

    Args:
        x: (H, W, C) tensor, any range.
        target_h, target_w: output spatial size.
        eps: pixels with norm <= eps are considered invalid for the mask.
        threshold: interpolated mask values below this are replaced with fill.
        fill: value to write where mask < threshold.

    Returns:
        (target_h, target_w, C) tensor, same dtype/device as x.
    """
    # Validity: near-zero = invalid
    valid = (x.pow(2).sum(dim=-1, keepdim=True) > eps ** 2).to(x.dtype)

    # (H, W, C) -> (1, C, H, W)
    x_batch = x.permute(2, 0, 1).unsqueeze(0)
    valid_batch = valid.permute(2, 0, 1).unsqueeze(0)

    x_interp = F.interpolate(
        x_batch,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )
    valid_interp = F.interpolate(
        valid_batch,
        size=(target_h, target_w),
        mode="bilinear",
        align_corners=False,
    )

    out = x_interp.squeeze(0).permute(1, 2, 0)
    mask = valid_interp.squeeze(0).permute(1, 2, 0)
    out = torch.where(mask >= threshold, out, torch.full_like(out, fill))

    return out
