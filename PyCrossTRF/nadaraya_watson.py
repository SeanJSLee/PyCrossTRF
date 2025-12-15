# nadaraya_watson_torch.py
# Nadaraya–Watson kernel regression in PyTorch (CUDA-ready) with LOO CV bandwidth selection.


# nadaraya_watson_torch_streaming.py
from __future__ import annotations
from typing import Iterable, Literal, Optional, Sequence, Tuple, Union
import torch
from torch import Tensor

KernelName = Literal["gaussian", "epanechnikov", "triangular", "uniform"]
DeviceLike = Union[str, torch.device]

def _ensure_2d(x: Tensor) -> Tensor:
    return x.view(-1, 1) if x.ndim == 1 else x

def _as_device(device: Optional[DeviceLike]) -> torch.device:
    return torch.device("cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu"))

def _kernel_1d(u: Tensor, name: KernelName) -> Tensor:
    if name == "gaussian":
        return torch.exp(-0.5 * (u**2))
    elif name == "epanechnikov":
        return torch.clamp(1.0 - (u**2), min=0.0)
    elif name == "triangular":
        return torch.clamp(1.0 - torch.abs(u), min=0.0)
    elif name == "uniform":
        return (torch.abs(u) <= 1.0).to(u.dtype)
    raise ValueError(f"Unknown kernel: {name}")

def _product_kernel(diff: Tensor, h: Tensor, kernel: KernelName) -> Tensor:
    u = diff / h  # (B, n_chunk, d)
    return _kernel_1d(u, kernel).prod(dim=-1)  # (B, n_chunk)

@torch.inference_mode()
def _weighted_sum_and_rowsums_streaming(
    Xq: Tensor,
    X: Tensor,
    Y: Tensor,
    h: Tensor,
    kernel: KernelName,
    *,
    query_batch_size: Optional[int] = None,
    train_chunk_size: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    """
    Streams over both queries and the training set to avoid allocating (B,N,d).

    Returns:
        wsum: (Q, m)  with wsum[i] = sum_j K(xq_i, x_j) * y_j
        rsum: (Q,)    with rsum[i] = sum_j K(xq_i, x_j)
    """
    Q, N = Xq.shape[0], X.shape[0]
    m = 1 if Y.ndim == 1 else Y.shape[1]
    if query_batch_size is None:
        query_batch_size = min(Q, 8192)
    if train_chunk_size is None:
        # safe default; adjust upward if you have more VRAM
        train_chunk_size = min(N, 65536)

    out_wsum = torch.zeros((Q, m), device=Xq.device, dtype=Xq.dtype)
    out_rsum = torch.zeros(Q, device=Xq.device, dtype=Xq.dtype)

    for qs in range(0, Q, query_batch_size):
        qe = min(qs + query_batch_size, Q)
        Xq_block = Xq[qs:qe]  # (B,d)
        wsum_block = torch.zeros((qe - qs, m), device=Xq.device, dtype=Xq.dtype)
        rsum_block = torch.zeros((qe - qs,), device=Xq.device, dtype=Xq.dtype)

        for ts in range(0, N, train_chunk_size):
            te = min(ts + train_chunk_size, N)
            X_tr = X[ts:te]                       # (n_chunk,d)
            Y_tr = Y[ts:te] if Y.ndim == 2 else Y[ts:te].view(-1, 1)  # (n_chunk,m)

            diff = Xq_block.unsqueeze(1) - X_tr.unsqueeze(0)          # (B,n_chunk,d)
            Kblk = _product_kernel(diff, h, kernel)                   # (B,n_chunk)

            rsum_block += Kblk.sum(dim=1)                             # (B,)
            wsum_block += Kblk @ Y_tr                                  # (B,m)

            # free ASAP
            del diff, Kblk, X_tr, Y_tr

        out_wsum[qs:qe] = wsum_block
        out_rsum[qs:qe] = rsum_block

        del Xq_block, wsum_block, rsum_block

    return out_wsum, out_rsum

def _diag_kernel_value() -> float:
    # With our kernels K(0)=1 in each dim, product -> 1.
    return 1.0

class NadarayaWatson:
    """
    Nadaraya–Watson regression with product kernels and streaming over data.
    Bandwidth chosen by LOO-CV unless provided.

    LOO identity for linear smoothers:
        yhat^(-i) = (yhat_i - S_ii * y_i) / (1 - S_ii),
    where S_ii = K_ii / sum_j K_ij and K_ii = 1 here.
    """
    def __init__(
        self,
        kernel: KernelName = "gaussian",
        bandwidth: Optional[Union[float, Sequence[float]]] = None,
        *,
        per_dimension: bool = False,
        device: Optional[DeviceLike] = None,
        dtype: torch.dtype = torch.float32,
        query_batch_size: Optional[int] = None,
        train_chunk_size: Optional[int] = None,
        fallback_to_cpu_for_cv: bool = True,
    ) -> None:
        self.kernel = kernel
        self.fixed_bandwidth = bandwidth
        self.per_dimension = per_dimension
        self.device = _as_device(device)
        self.dtype = dtype
        self.query_batch_size = query_batch_size
        self.train_chunk_size = train_chunk_size
        self.fallback_to_cpu_for_cv = fallback_to_cpu_for_cv

        self.h_: Optional[Tensor] = None
        self.X_: Optional[Tensor] = None
        self.Y_: Optional[Tensor] = None

    def _to_device_dtype(self, *tensors: Tensor) -> Tuple[Tensor, ...]:
        return tuple(t.to(self.device, self.dtype, non_blocking=True) for t in tensors)

    def _prepare_h_grid(
        self, X: Tensor, grid: Optional[Iterable[Union[float, Sequence[float]]]] = None
    ) -> Sequence[Tensor]:
        n, d = X.shape
        if grid is not None:
            hs = []
            for h in grid:
                t = torch.as_tensor(h, dtype=self.dtype, device=self.device)
                if t.ndim == 0 and self.per_dimension:
                    t = t.repeat(d)
                if t.ndim == 1 and not self.per_dimension:
                    t = t.mean()
                hs.append(t)
            return hs
        std = X.std(dim=0, unbiased=False).clamp_min(torch.finfo(self.dtype).eps)
        rate = n ** (-1.0 / (d + 4.0))
        h0_vec = 1.06 * std * rate
        base = h0_vec if self.per_dimension else torch.sqrt((h0_vec**2).mean())
        multipliers = torch.tensor([0.25, 0.5, 1.0, 2.0, 4.0], device=self.device, dtype=self.dtype)
        return [(base * m) for m in multipliers]

    def fit(
        self,
        X: Union[Tensor, "numpy.ndarray"],
        Y: Union[Tensor, "numpy.ndarray"],
        *,
        bandwidth_grid: Optional[Iterable[Union[float, Sequence[float]]]] = None,
    ) -> "NadarayaWatson":
        X = torch.as_tensor(X, dtype=self.dtype)
        Y = torch.as_tensor(Y, dtype=self.dtype)
        X = _ensure_2d(X)
        if Y.ndim == 1:
            Y = Y.view(-1, 1)
        X, Y = self._to_device_dtype(X, Y)
        self.X_, self.Y_ = X, Y

        if self.fixed_bandwidth is not None:
            h = torch.as_tensor(self.fixed_bandwidth, dtype=self.dtype, device=self.device)
            if h.ndim == 0 and self.per_dimension:
                h = h.repeat(X.shape[1])
            if h.ndim == 1 and not self.per_dimension:
                h = h.mean()
            self.h_ = h
            return self

        h_grid = self._prepare_h_grid(X, bandwidth_grid)
        best_mse, best_h = float("inf"), None

        # Optionally evaluate CV on CPU to save VRAM
        if self.fallback_to_cpu_for_cv and self.device.type == "cuda":
            X_cpu = X.detach().to("cpu")
            Y_cpu = Y.detach().to("cpu")
            for h in h_grid:
                mse = self._loo_mse_for_h(h.detach().to("cpu"), X_cpu, Y_cpu, device_override=torch.device("cpu"))
                if mse < best_mse:
                    best_mse, best_h = mse, h
        else:
            for h in h_grid:
                mse = self._loo_mse_for_h(h, X, Y, device_override=self.device)
                if mse < best_mse:
                    best_mse, best_h = mse, h

        assert best_h is not None
        self.h_ = best_h.to(self.device)
        return self

    @torch.inference_mode()
    def predict(self, Xq: Union[Tensor, "numpy.ndarray"]) -> Tensor:
        assert self.X_ is not None and self.Y_ is not None and self.h_ is not None, "Call fit() first."
        Xq = torch.as_tensor(Xq, dtype=self.dtype, device=self.device)
        Xq = _ensure_2d(Xq)
        wsum, rsum = _weighted_sum_and_rowsums_streaming(
            Xq, self.X_, self.Y_, self.h_, self.kernel,
            query_batch_size=self.query_batch_size, train_chunk_size=self.train_chunk_size
        )
        eps = torch.finfo(self.dtype).eps
        yhat = wsum / (rsum.clamp_min(eps).unsqueeze(-1))
        return yhat.squeeze(-1) if self.Y_.shape[1] == 1 else yhat

    @torch.inference_mode()
    def _loo_mse_for_h(self, h: Tensor, X: Tensor, Y: Tensor, *, device_override: torch.device) -> float:
        # Compute predictions at TRAIN points using streaming
        wsum, rsum = _weighted_sum_and_rowsums_streaming(
            X, X, Y, h, self.kernel,
            query_batch_size=self.query_batch_size, train_chunk_size=self.train_chunk_size
        )
        eps = torch.finfo(X.dtype).eps
        yhat = wsum / (rsum.clamp_min(eps).unsqueeze(-1))       # (n,m)

        # S_ii = K_ii / sum_j K_ij, with K_ii = 1
        sii = (torch.as_tensor(_diag_kernel_value(), dtype=X.dtype, device=device_override) /
               rsum.clamp_min(eps))                              # (n,)

        # LOO
        numer = yhat - sii.unsqueeze(-1) * Y
        denom = (1.0 - sii).clamp_min(eps).unsqueeze(-1)
        yhat_loo = numer / denom

        mse = torch.mean((Y - yhat_loo) ** 2).item()
        return float(mse)

# # ------------ quick sanity test -------------
# if __name__ == "__main__":
#     torch.manual_seed(0)
#     dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     X = torch.linspace(-3, 3, 30_000, device=dev).view(-1, 1)   # large-ish
#     f = torch.sin(X) + 0.5 * torch.cos(2 * X)
#     Y = f + 0.2 * torch.randn_like(f)

#     nw = NadarayaWatson(kernel="gaussian", device=dev, dtype=torch.float32,
#                         query_batch_size=4096, train_chunk_size=32768,
#                         fallback_to_cpu_for_cv=True)
#     nw.fit(X, Y)                       # CV may run on CPU to save VRAM
#     yhat = nw.predict(X[:10])          # predict a few points
#     print("Selected h:", float(nw.h_) if nw.h_.ndim == 0 else nw.h_.tolist())
#     print("Pred sample:", yhat[:3].flatten().tolist())


# tnwr = NadarayaWatson(
#     kernel="gaussian",
#     device="cuda",                 # or "cpu"
#     dtype=torch.float32,           # use float16 only if you really need memory
#     query_batch_size=2048,         # tune up/down per VRAM
#     train_chunk_size=10240,        # tune up/down per VRAM
#     fallback_to_cpu_for_cv=True,   # CV on CPU to keep GPU RAM free
# )
# tnwr.fit(df_ctrf['temp_q'].to_numpy(), df_ctrf['ln_mortality_30days_harvest'].to_numpy())
