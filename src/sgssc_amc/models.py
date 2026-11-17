"""Inference architecture extracted from the project SGSSC adaptation.

The encoder/readout and ordered linear spectral head retain the original
parameter names and computations. This is an independent implementation,
not the IJCAI authors' official code. Clean training routines are in training.py.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from numbers import Integral
from typing import Sequence
import torch
from torch import Tensor, nn

POSITION_INIT_STD = 0.02
SPECTRAL_DECODER_HEADS = 4
NUM_CLASSES = 11

@dataclass(frozen=True)
class SGSSCConfig:
    length: int = 512
    patch: int = 16
    dim: int = 128
    depth: int = 8
    heads: int = 8
    ff: int = 1024
    dropout: float = 0.1
    norm_eps: float = 1e-5

    def __post_init__(self):
        for name in ("length", "patch", "dim", "depth", "heads", "ff"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.length % self.patch:
            raise ValueError("length must be divisible by patch")
        if self.dim % self.heads or self.dim % SPECTRAL_DECODER_HEADS:
            raise ValueError("dim must be divisible by encoder and decoder attention heads")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must be finite and in [0, 1)")
        if not math.isfinite(self.norm_eps) or self.norm_eps <= 0:
            raise ValueError("norm_eps must be finite and positive")

    @property
    def tokens(self) -> int:
        return self.length // self.patch

    @property
    def fusion_tokens(self) -> int:
        return self.depth * self.tokens


def _check_iq(y: Tensor) -> None:
    if not isinstance(y, Tensor) or y.ndim != 3 or y.shape[-1] != 2:
        raise ValueError("expected a real floating tensor shaped [batch, length, 2] in I,Q order")
    if not y.is_floating_point():
        raise TypeError("I/Q coordinates must use a real floating dtype")
    if y.shape[0] < 1 or y.shape[1] < 1:
        raise ValueError("batch and time dimensions must be nonempty")


def patch_iq(y: Tensor, patch: int = 16) -> Tensor:
    """Return [B,N,2*patch], ordered I0,Q0,I1,Q1,... inside each patch."""
    _check_iq(y)
    if isinstance(patch, bool) or not isinstance(patch, Integral) or patch < 1 or y.shape[1] % patch:
        raise ValueError("patch must be positive and divide the input length")
    # 连续时间点按I/Q交替展开：[B,512,2] -> [B,32,32]，不把I与Q拆成独立token。
    return y.reshape(y.shape[0], y.shape[1] // patch, 2 * patch)


def mask_iq(y: Tensor, ratio: float = 0.7,
            generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
    """Zero exactly round(ratio*L) time points per example, with shared I/Q mask.

    The caller owns the generator and its seed. This operation neither adds
    noise nor mutates the shared input view. The bool mask is True at masked
    points and has shape [B,L], independent of the patch boundaries.
    """
    _check_iq(y)
    if not math.isfinite(ratio) or not 0 <= ratio <= 1:
        raise ValueError("ratio must be finite and in [0, 1]")
    # 每例无放回选取遮挡点；默认round(0.7*512)=358，掩码不依赖patch边界。
    count = round(float(ratio) * y.shape[1])
    mask = torch.zeros(y.shape[:2], dtype=torch.bool, device=y.device)
    if count:
        for example in range(y.shape[0]):
            indices = torch.randperm(y.shape[1], generator=generator, device=y.device)[:count]
            mask[example, indices] = True
    # 同一时间点的I/Q同步置零；返回新张量，完整输入保持不变。
    return y.masked_fill(mask.unsqueeze(-1), 0), mask


def spectrum_target(y: Tensor) -> Tensor:
    """Full complex orthonormal DFT of an I/Q window, negative frequencies first.

    Output is [B,L,2] with real/imaginary coordinates; norm='ortho' preserves
    the full complex energy. There is no magnitude reduction, half spectrum,
    or per-example spectral normalization. The public spectral task uses the
    saved no-added-noise reference windows before applying its fixed masks.
    """
    _check_iq(y)
    if y.dtype not in (torch.float32, torch.float64):
        raise TypeError("CPU complex DFT coordinates require float32 or float64 I/Q")
    # 保留完整输入的所有复频点实部/虚部，不改成幅度谱。
    z = torch.complex(y[..., 0], y[..., 1])
    # 正交FFT保持能量尺度，fftshift把负频率移到前半段。
    transformed = torch.fft.fftshift(torch.fft.fft(z, dim=1, norm="ortho"), dim=1)
    return torch.stack((transformed.real, transformed.imag), dim=-1)


class _EncoderLayer(nn.Module):
    def __init__(self, config: SGSSCConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.attention = nn.MultiheadAttention(config.dim, config.heads,
                                               dropout=config.dropout, batch_first=True)
        self.dropout_attention = nn.Dropout(config.dropout)
        self.norm_ff = nn.LayerNorm(config.dim, eps=config.norm_eps)
        self.ff = nn.Sequential(nn.Linear(config.dim, config.ff), nn.GELU(),
                                nn.Dropout(config.dropout), nn.Linear(config.ff, config.dim))
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, tokens: Tensor) -> Tensor:
        normalized = self.norm_attention(tokens)
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + self.dropout_attention(attended)
        return tokens + self.dropout_ff(self.ff(self.norm_ff(tokens)))


class SGSSCEncoder(nn.Module):
    """Independent pre-LN blocks; return every layer's post-residual tokens."""

    def __init__(self, config: SGSSCConfig | None = None):
        super().__init__()
        self.config = config or SGSSCConfig()
        self.embedding = nn.Linear(2 * self.config.patch, self.config.dim)
        self.position = nn.Parameter(torch.empty(1, self.config.tokens, self.config.dim))
        nn.init.normal_(self.position, std=POSITION_INIT_STD)
        self.layers = nn.ModuleList([_EncoderLayer(self.config) for _ in range(self.config.depth)])

    def forward(self, y: Tensor) -> list[Tensor]:
        _check_iq(y)
        if y.shape[1] != self.config.length:
            raise ValueError(f"encoder expects length {self.config.length}")
        # 每组32个实数投影到128维，再加入可学习的位置编码。
        tokens = self.embedding(patch_iq(y, self.config.patch)) + self.position
        # 保留H1...H8供下游多层融合；预训练频谱头只读取最后一层H8。
        features = []
        for layer in self.layers:
            tokens = layer(tokens)
            features.append(tokens)
        return features

    def encode_all(self, y: Tensor) -> Tensor:
        """Convenient differentiable [B,depth,N,D] layout for feature caching."""
        return torch.stack(self(y), dim=1)


class SGSSCReadout(nn.Module):
    """Feature-mean token gating, sigmoid weights, unnormalized weighted sum."""

    def __init__(self, config: SGSSCConfig | None = None):
        super().__init__()
        self.config = config or SGSSCConfig()
        tokens = self.config.fusion_tokens
        self.attention = nn.Sequential(nn.Linear(tokens, max(1, tokens // 4)), nn.GELU(),
                                       nn.Linear(max(1, tokens // 4), tokens), nn.Sigmoid())
        self.classifier = nn.Linear(self.config.dim, NUM_CLASSES)

    def extract_features(self, features: Sequence[Tensor] | Tensor) -> Tensor:
        """Return the label-supervised pooled representation before classification.

        The original learned token gates pool all encoder layers into [B, D]
        ([B, 128] for the released checkpoint). These features include the
        downstream readout's label supervision; they are not encoder-only
        self-supervised representations. Parameter names and pooling are
        identical to the original classification forward pass.
        """
        # 八层特征沿token轴拼接，默认得到[B,256,128]，保留浅层与深层信息。
        if isinstance(features, Tensor):
            expected = (self.config.depth, self.config.tokens, self.config.dim)
            if features.ndim != 4 or tuple(features.shape[1:]) != expected:
                raise ValueError("stacked features must have shape [B,depth,N,D]")
            fused = features.reshape(features.shape[0], self.config.fusion_tokens, self.config.dim)
        else:
            if len(features) != self.config.depth:
                raise ValueError("readout requires exactly one feature tensor per encoder layer")
            if any(f.ndim != 3 or tuple(f.shape[1:]) != (self.config.tokens, self.config.dim) for f in features):
                raise ValueError("layer features must have shape [B,N,D]")
            fused = torch.cat(tuple(features), dim=1)
        # 沿128维特征取均值，MLP256->64->256+sigmoid生成token权重；加权和不再归一化。
        weights = self.attention(fused.mean(dim=-1))
        pooled = torch.sum(fused * weights.unsqueeze(-1), dim=1)
        return pooled

    def forward(self, features: Sequence[Tensor] | Tensor) -> Tensor:
        # 输出11类logits，训练时直接交给CrossEntropyLoss，不预先做softmax。
        return self.classifier(self.extract_features(features))


class FlattenSpectralHead(nn.Module):
    """按时间 token 顺序展平 H8，再线性预测完整复频谱。

    默认 [B,32,128] → [B,4096] → Linear(4096,1024) → [B,512,2]。
    输出最后一维仍按实部、虚部排列；具体 FFT 坐标由外部目标定义。
    """

    def __init__(self, config: SGSSCConfig | None = None):
        super().__init__()
        self.config = config or SGSSCConfig()
        # Linear 使用 PyTorch 默认初始化及偏置；没有额外激活或归一化。
        self.projection = nn.Linear(self.config.tokens * self.config.dim,
                                    self.config.length * 2, bias=True)

    def forward(self, memory: Tensor) -> Tensor:
        if not isinstance(memory, Tensor) or memory.ndim != 3:
            raise ValueError("flat spectral head requires [B,N,D] final encoder features")
        if memory.shape[0] < 1 or tuple(memory.shape[1:]) != (self.config.tokens, self.config.dim):
            raise ValueError("flat spectral head requires a nonempty batch and configured [N,D]")
        if not memory.is_floating_point():
            raise TypeError("flat spectral head requires real floating features")
        # reshape 按 token0 的全部通道、token1 的全部通道……排列，不做平均池化。
        ordered = memory.reshape(memory.shape[0], self.config.tokens * self.config.dim)
        return self.projection(ordered).reshape(memory.shape[0], self.config.length, 2)
