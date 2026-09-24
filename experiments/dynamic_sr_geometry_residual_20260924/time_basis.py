"""Fixed eight-term global Legendre/local cubic B-spline time bases."""
import math
import torch
from torch import nn

KNOTS = [0., 0., 0., 0., .2, .4, .6, .8, 1., 1., 1., 1.]
FRAMES = list(range(0, 120, 2))


def raw_basis(tau, kind):
    if not torch.isfinite(tau).all() or bool(((tau < 0) | (tau > 1)).any()):
        raise ValueError('Geometry residual supports only frame0..118 (tau in [0,1])')
    if kind == 'G':
        z = 2 * tau - 1
        terms = [torch.ones_like(z), z]
        for k in range(1, 7):
            terms.append(((2*k+1)*z*terms[-1]-k*terms[-2])/(k+1))
        return torch.stack(terms, -1)
    if kind != 'L':
        raise ValueError(kind)
    q = KNOTS
    terms = [((tau >= q[i]) & (tau < q[i+1])).to(tau.dtype) for i in range(11)]
    for p in range(1, 4):
        terms = [
            ((tau-q[i])/(q[i+p]-q[i])*terms[i] if q[i+p] > q[i] else torch.zeros_like(tau)) +
            ((q[i+p+1]-tau)/(q[i+p+1]-q[i+1])*terms[i+1] if q[i+p+1] > q[i+1] else torch.zeros_like(tau))
            for i in range(11-p)]
    value = torch.stack(terms, -1)
    endpoint = torch.zeros_like(value); endpoint[..., -1] = 1
    return torch.where((tau == 1)[..., None], endpoint, value)


class TimeBasis(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        tau = torch.tensor(FRAMES, dtype=torch.float64) / 118
        raw = raw_basis(tau, kind)
        rms = raw.square().mean(0).sqrt()
        if not bool(torch.isfinite(rms).all() and (rms > 0).all()):
            raise ValueError('Invalid time-basis RMS')
        self.register_buffer('rms64', rms)
        self.register_buffer('knots64', torch.tensor(KNOTS if kind == 'L' else [], dtype=torch.float64))
        self.register_buffer('normalized60', (raw/(math.sqrt(8)*rms)).float())

    def forward(self, frame):
        frame = float(frame)
        if not math.isfinite(frame) or not 0 <= frame <= 118:
            raise ValueError(f'Frame {frame} outside declared short window')
        if frame.is_integer() and int(frame) % 2 == 0:
            return self.normalized60[int(frame)//2]
        tau = torch.tensor(frame/118, device=self.rms64.device, dtype=torch.float64)
        return (raw_basis(tau, self.kind)/(math.sqrt(8)*self.rms64)).to(self.normalized60.dtype)

    def protocol(self):
        matrix = self.normalized60.double()
        return dict(kind=self.kind, k=8, degree=3 if self.kind=='L' else 7,
                    knots=self.knots64.tolist(), rms=self.rms64.tolist(), frames=FRAMES,
                    tau='frame_id/118', original_time='manifest time unchanged; frame_id/300',
                    rank=int(torch.linalg.matrix_rank(matrix)), condition=float(torch.linalg.cond(matrix)),
                    mean_row_square_norm=float(matrix.square().sum(-1).mean()),
                    normalization='raw / (sqrt(8) * per-column RMS over 60 train times); no whitening')
