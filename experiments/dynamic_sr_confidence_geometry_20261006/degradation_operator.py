"""Actual bicubic-AA degradation, its adjoint, and differentiable null projector.

The image-domain projector is tied to PyTorch's *linear* downsampling operator,
without clamp or PNG quantization. Runtime HR-axis products are sparse; Gram
systems are solved only at LR scale. This does not guarantee that renderer/Adam
parameter updates leave any LR render unchanged.
"""
from __future__ import annotations

import hashlib
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


def _impulse_axis(n: int, m: int) -> torch.Tensor:
    # Each input channel is one impulse. A horizontal axis avoids degenerate
    # height-only interpolation paths and also constructs the identical h kernel.
    impulses = torch.eye(n, dtype=torch.float64).reshape(1, n, 1, n)
    out = F.interpolate(impulses, size=(1, m), mode="bicubic",
                        align_corners=False, antialias=True)
    return out[0, :, 0].T.contiguous()


def _axis_product(x: torch.Tensor, matrix: torch.Tensor, axis: int) -> torch.Tensor:
    axis = axis % x.ndim
    moved = x.movedim(axis, 0)
    result = torch.sparse.mm(matrix, moved.reshape(moved.shape[0], -1))
    return result.reshape(matrix.shape[0], *moved.shape[1:]).movedim(0, axis)


def _axis_solve(x: torch.Tensor, factor: torch.Tensor, axis: int,
                mode: str) -> torch.Tensor:
    axis = axis % x.ndim
    moved = x.movedim(axis, 0)
    rhs = moved.reshape(moved.shape[0], -1)
    if mode == "cholesky":
        solved = torch.cholesky_solve(rhs, factor)
    else:
        solved = factor @ rhs
    return solved.reshape(moved.shape).movedim(0, axis)


class _NullProject(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, operator):
        ctx.operator = operator
        return operator._q_impl(x)

    @staticmethod
    def backward(ctx, gradient):
        # Q is self-adjoint. Avoid saving HR intermediates; explicitly apply the
        # same projector to the pixel gradient, including after spatial weighting.
        return ctx.operator._q_impl(gradient), None


class ActualDegradation(nn.Module):
    """Separable D0, D0^T, D0^dagger and Q for CHW / NCHW images.

    Construction and factorization always use CPU float64. Input floating dtype
    and device must match the module. float16/bfloat16 are deliberately rejected:
    Gram solving and null-space leakage must be checked in float32 or float64.
    """

    def __init__(self, hr_size, lr_size, *, device=None,
                 dtype=torch.float32, pinv_rtol=1e-12):
        super().__init__()
        self.hr_size = tuple(map(int, hr_size))
        self.lr_size = tuple(map(int, lr_size))
        if len(self.hr_size) != 2 or len(self.lr_size) != 2:
            raise ValueError("hr_size/lr_size must each have two dimensions")
        if any(m <= 0 or m > n for n, m in zip(self.hr_size, self.lr_size)):
            raise ValueError("Require positive LR <= HR on both axes")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("Validated runtime dtypes are float32 and float64")
        if not 0 < pinv_rtol < 1:
            raise ValueError("pinv_rtol must lie strictly between 0 and 1")
        self.pinv_rtol = float(pinv_rtol)
        self._axis_info = {}
        for name, n, m in zip(("h", "w"), self.hr_size, self.lr_size):
            a = _impulse_axis(n, m)
            gram = a @ a.T
            eigenvalues, eigenvectors = torch.linalg.eigh(gram)
            cutoff = float(eigenvalues[-1]) * self.pinv_rtol
            keep = eigenvalues > cutoff
            rank = int(keep.sum())
            if rank == m:
                factor = torch.linalg.cholesky(gram)
                mode = "cholesky"
            else:
                safe_inverse = torch.where(keep, eigenvalues.clamp_min(cutoff).reciprocal(),
                                           torch.zeros_like(eigenvalues))
                factor = (eigenvectors * safe_inverse) @ eigenvectors.T
                mode = "eigen_pinv"
            nnz = int(a.count_nonzero())
            self._axis_info[name] = {
                "input": n, "output": m, "nonzero": nnz,
                "maximum_support": int((a != 0).sum(1).max()),
                "gram_rank": rank, "gram_solver": mode,
                "gram_min_eigenvalue": float(eigenvalues[0]),
                "gram_max_eigenvalue": float(eigenvalues[-1]),
                "gram_condition": float(eigenvalues[-1] / eigenvalues[keep][0]),
                "pinv_absolute_cutoff": cutoff,
                "axis_float64_sha256": hashlib.sha256(a.numpy().tobytes()).hexdigest(),
            }
            # COO sparse.mm supports autograd for dense RHS on CPU and CUDA.
            # No dense HR-axis matrix remains registered in the runtime module.
            self.register_buffer("a_" + name, a.to_sparse().coalesce())
            self.register_buffer("at_" + name, a.T.to_sparse().coalesce())
            self.register_buffer("factor_" + name, factor)
        self.to(device=device, dtype=dtype)

    def _check(self, x, size):
        if x.ndim not in (3, 4) or tuple(x.shape[-2:]) != size:
            raise ValueError(f"Expected CHW/NCHW ending in {size}, got {tuple(x.shape)}")
        if x.dtype != self.factor_h.dtype or x.device != self.factor_h.device:
            raise ValueError("Image dtype/device must match ActualDegradation")

    def D0(self, x):
        """Linear bicubic-antialias downsampling; no clamp or quantization."""
        self._check(x, self.hr_size)
        return _axis_product(_axis_product(x, self.a_w, -1), self.a_h, -2)

    def D(self, x):
        """Historical training degradation, linear downsampling then clamp."""
        return self.D0(x).clamp(0, 1)

    def transpose(self, y):
        """True Euclidean adjoint of D0; not bicubic upsampling."""
        self._check(y, self.lr_size)
        return _axis_product(_axis_product(y, self.at_h, -2), self.at_w, -1)

    def pinv(self, y):
        """D0^T (D0 D0^T)^dagger y via two LR-axis Gram systems."""
        self._check(y, self.lr_size)
        solved = _axis_solve(y, self.factor_h, -2, self._axis_info["h"]["gram_solver"])
        solved = _axis_solve(solved, self.factor_w, -1, self._axis_info["w"]["gram_solver"])
        return self.transpose(solved)

    def _q_impl(self, x):
        return x - self.pinv(self.D0(x))

    def Q(self, x):
        """I - D0^dagger D0, with the same symmetric operator in backward."""
        self._check(x, self.hr_size)
        return _NullProject.apply(x, self)

    def detail_loss(self, render, teacher, confidence, k_r=1.0):
        """k_R mean(c |Q(render-teacher)|); caller applies the SR coefficient 0.1.

        confidence is frozen HW / 1HW (or N1HW for batches), shared over RGB;
        the mean covers every RGB pixel and is not normalized by weight sum.
        """
        self._check(render, self.hr_size)
        self._check(teacher, self.hr_size)
        if tuple(render.shape) != tuple(teacher.shape):
            raise ValueError("Teacher/render shapes must match")
        c = confidence.detach()
        if c.ndim == 2:
            c = c.unsqueeze(0)
        if render.ndim == 4 and c.ndim == 3:
            c = c.unsqueeze(0)
        expected = (1, *self.hr_size) if render.ndim == 3 else (render.shape[0], 1, *self.hr_size)
        if tuple(c.shape) != expected:
            raise ValueError(f"Confidence must share spatial weights across RGB: expected {expected}")
        if c.dtype != render.dtype or c.device != render.device:
            raise ValueError("Confidence dtype/device must match render")
        return float(k_r) * (c * self.Q(render - teacher.detach()).abs()).mean()

    def project_to_lr(self, x, y):
        """Float visualization source Pi_y(x); deliberately no output clamp."""
        return x + self.pinv(y - self.D0(x))

    def fusion_preview(self, teacher, lr, confidence):
        """T*=B+Q(c*(T-B)); visualization only, not the training target."""
        batch = lr.unsqueeze(0) if lr.ndim == 3 else lr
        up = F.interpolate(batch, size=self.hr_size, mode="bicubic",
                           align_corners=False, antialias=True)
        up = up[0] if lr.ndim == 3 else up
        base = self.project_to_lr(up, lr)
        return base + self.Q(confidence.detach() * (teacher.detach() - base))

    def metadata(self):
        return {"kind": "bicubic_actual_linear_nullspace", "torch_version": torch.__version__,
                "antialias": True, "align_corners": False,
                "construction_dtype": "float64", "runtime_dtype": str(self.factor_h.dtype),
                "runtime_device": str(self.factor_h.device), "runtime_axis_products": "sparse_coo",
                "runtime_gram_solve_scale": "LR", "pinv_relative_cutoff": self.pinv_rtol,
                "clamp_in_D_only": True, "quantization": False,
                "hr_size": list(self.hr_size), "lr_size": list(self.lr_size),
                "axes": self._axis_info}


def calibrate_pixel_gradient_scale(operator: ActualDegradation,
                                   pairs: Iterable[tuple[torch.Tensor, torch.Tensor]],
                                   expected_observations: int = 32):
    """Single registered c=1 pixel-gradient calibration, before confidence.

    pairs contain detached U6000 renders and frozen teachers, never HR. RMS is
    aggregated across all pixels/observations, then full-SR / Q RMS is returned.
    No module parameters or training RNG are updated. The caller fixes and saves
    the exact 32-observation list and reuses one k_R for R and RG.
    """
    full_sq, q_sq, count, observations = 0.0, 0.0, 0, 0
    for render, teacher in pairs:
        x = render.detach().requires_grad_(True)
        full = (x - teacher.detach()).abs().mean()
        full_grad, = torch.autograd.grad(full, x)
        detail = operator.Q(x - teacher.detach()).abs().mean()
        detail_grad, = torch.autograd.grad(detail, x)
        full_sq += float(full_grad.double().square().sum())
        q_sq += float(detail_grad.double().square().sum())
        count += x.numel()
        observations += 1
    if observations != expected_observations or not count:
        raise ValueError(f"Expected {expected_observations} fixed observations, got {observations}")
    full_rms, q_rms = (full_sq / count) ** .5, (q_sq / count) ** .5
    if not torch.isfinite(torch.tensor([full_rms, q_rms])).all():
        raise FloatingPointError("Nonfinite pixel gradients; repair operator before training")
    # Accuracy registration is absolute at the gradient's natural 1/N scale.
    precision_floor = torch.finfo(operator.factor_h.dtype).eps * full_rms
    fallback = q_rms <= precision_floor
    k_r = 1.0 if fallback else full_rms / q_rms
    return {"k_R": k_r, "full_sr_pixel_gradient_rms": full_rms,
            "q_c1_pixel_gradient_rms": q_rms, "observations": observations,
            "confidence_used": False, "gradient_precision_floor": precision_floor,
            "fallback_k_R_1": fallback}
