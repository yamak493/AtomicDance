"""Rotation conversions that work with or without PyTorch3D installed.

PyTorch3D ships CUDA extensions and has no wheels for the Python/CUDA
combinations used by hosted notebooks such as Google Colab, where building it
from source takes tens of minutes and often fails outright. Every PyTorch3D
symbol this project needs lives in ``pytorch3d.transforms.rotation_conversions``
and is implemented in plain PyTorch, so the pure-PyTorch fallbacks below are
used whenever PyTorch3D cannot be imported.

The fallbacks are ported from PyTorch3D (BSD-3-Clause, Copyright (c) Meta
Platforms, Inc. and affiliates) and are numerically equivalent to the originals.
Import these helpers from this module rather than from ``pytorch3d.transforms``
so the project keeps running on either setup.
"""

import torch
import torch.nn.functional as F

__all__ = [
    "USING_PYTORCH3D",
    "axis_angle_to_matrix",
    "axis_angle_to_quaternion",
    "matrix_to_axis_angle",
    "matrix_to_quaternion",
    "matrix_to_rotation_6d",
    "quaternion_apply",
    "quaternion_invert",
    "quaternion_multiply",
    "quaternion_raw_multiply",
    "quaternion_to_axis_angle",
    "quaternion_to_matrix",
    "rotation_6d_to_matrix",
    "standardize_quaternion",
]


def _standardize_quaternion(quaternions):
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def _quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def _sqrt_positive_part(x):
    """``torch.sqrt(max(0, x))`` with a zero subgradient where ``x`` is zero."""
    positive = torch.zeros_like(x)
    mask = x > 0
    positive[mask] = torch.sqrt(x[mask])
    return positive


def _matrix_to_quaternion(matrix):
    if matrix.shape[-1] != 3 or matrix.shape[-2] != 3:
        raise ValueError("Invalid rotation matrix shape {}.".format(matrix.shape))

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )

    # The four candidate quaternions, each numerically stable for a different
    # dominant component; the largest ``q_abs`` entry selects the best one.
    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    selected = quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))
    return _standardize_quaternion(selected)


def _sin_half_over_angle(angles, half_angles):
    """``sin(angles / 2) / angles`` with a Taylor expansion near zero."""
    eps = 1e-6
    small = angles.abs() < eps
    result = torch.empty_like(angles)
    result[~small] = torch.sin(half_angles[~small]) / angles[~small]
    # sin(x/2)/x = 0.5 - x^2 / 48 + O(x^4)
    result[small] = 0.5 - (angles[small] * angles[small]) / 48
    return result


def _axis_angle_to_quaternion(axis_angle):
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    return torch.cat(
        (torch.cos(half_angles), axis_angle * _sin_half_over_angle(angles, half_angles)),
        dim=-1,
    )


def _quaternion_to_axis_angle(quaternions):
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    return quaternions[..., 1:] / _sin_half_over_angle(angles, half_angles)


def _axis_angle_to_matrix(axis_angle):
    return _quaternion_to_matrix(_axis_angle_to_quaternion(axis_angle))


def _matrix_to_axis_angle(matrix):
    return _quaternion_to_axis_angle(_matrix_to_quaternion(matrix))


def _matrix_to_rotation_6d(matrix):
    batch_dim = matrix.shape[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def _rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def _quaternion_raw_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def _quaternion_multiply(a, b):
    return _standardize_quaternion(_quaternion_raw_multiply(a, b))


def _quaternion_invert(quaternion):
    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def _quaternion_apply(quaternion, point):
    if point.shape[-1] != 3:
        raise ValueError("Points are not in 3D, {}.".format(point.shape))
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = _quaternion_raw_multiply(
        _quaternion_raw_multiply(quaternion, point_as_quaternion),
        _quaternion_invert(quaternion),
    )
    return out[..., 1:]


try:  # Prefer the upstream implementation when the extension is available.
    from pytorch3d.transforms import (axis_angle_to_matrix,
                                      axis_angle_to_quaternion,
                                      matrix_to_axis_angle,
                                      matrix_to_quaternion,
                                      matrix_to_rotation_6d, quaternion_apply,
                                      quaternion_invert, quaternion_multiply,
                                      quaternion_raw_multiply,
                                      quaternion_to_axis_angle,
                                      quaternion_to_matrix,
                                      rotation_6d_to_matrix,
                                      standardize_quaternion)

    USING_PYTORCH3D = True
except ImportError:  # pragma: no cover - exercised by whichever env is in use
    axis_angle_to_matrix = _axis_angle_to_matrix
    axis_angle_to_quaternion = _axis_angle_to_quaternion
    matrix_to_axis_angle = _matrix_to_axis_angle
    matrix_to_quaternion = _matrix_to_quaternion
    matrix_to_rotation_6d = _matrix_to_rotation_6d
    quaternion_apply = _quaternion_apply
    quaternion_invert = _quaternion_invert
    quaternion_multiply = _quaternion_multiply
    quaternion_raw_multiply = _quaternion_raw_multiply
    quaternion_to_axis_angle = _quaternion_to_axis_angle
    quaternion_to_matrix = _quaternion_to_matrix
    rotation_6d_to_matrix = _rotation_6d_to_matrix
    standardize_quaternion = _standardize_quaternion

    USING_PYTORCH3D = False
