import torch
import torch.nn.functional as F
from torch import Tensor
import triton
import triton.language as tl


def _mamba2_pytorch_fwd(A: Tensor, delta: Tensor, x: Tensor, B: Tensor, C: Tensor, h0: Tensor):
    batch_size, length, n_heads = A.shape
    _, _, d_state, d_head = h0.shape
    x = x.view(batch_size, length, n_heads, d_head)

    _A = torch.exp(-torch.exp(A) * F.softplus(delta))   # (B, T, H)
    _B = B[:, :, None, :] * F.softplus(delta[:, :, :, None])        # (B, T, H, N)
    y = []
    h_curr = h0
    for i in range(0, length):
        h_next = _A[:, i, :, None, None] * h_curr + _B[:, i, :, :, None] * x[:, i, :, None, :]   # (B, H, N, P)
        y.append(torch.sum(C[:, i, None, :, None] * h_next, dim=(-2)))     # (B, H, N, P) -> (B, H, P)
        h_curr = h_next
    return torch.stack(y, dim=1), h_curr

@triton.jit
def softplus(x: tl.tensor) -> tl.tensor:
    return tl.log(1 + tl.exp(x))

@triton.jit
def sigmoid(x: tl.tensor) -> tl.tensor:
    return 1 / (1 + tl.exp(-x))

@triton.jit
def _mamba2_fwd_kernel(
    # ptr
    A_ptr, delta_ptr, x_ptr, y_ptr, B_ptr, C_ptr, h_ptr,
    # shapes
    length, d_head: tl.constexpr, d_state: tl.constexpr,
    # strides
    stride_A_B, stride_A_T, stride_A_H,                 # (B, T, H)
    stride_delta_B, stride_delta_T, stride_delta_H,     # (B, T, H)
    stride_x_B, stride_x_T, stride_x_H, stride_x_P,     # (B, T, H, P)
    stride_y_B, stride_y_T, stride_y_H, stride_y_P,     # (B, T, H, P)
    stride_B_B, stride_B_T, stride_B_N,                 # (B, T, N)
    stride_C_B, stride_C_T, stride_C_N,                 # (B, T, N)
    stride_h_B, stride_h_T, stride_h_H, stride_h_N, stride_h_P,     # (B, T//C, H, N, P)
    BLOCK_LEN: tl.constexpr, data_type: tl.constexpr
):
    pid_B = tl.program_id(0)
    pid_H = tl.program_id(1)
    offs_T = tl.arange(0, BLOCK_LEN)
    offs_P = tl.arange(0, d_head)
    offs_N = tl.arange(0, d_state)

    # set ptrs
    A_ptrs = A_ptr + pid_B * stride_A_B + pid_H * stride_A_H + offs_T[:] * stride_A_T                                       # (T)
    delta_ptrs = delta_ptr + pid_B * stride_delta_B + pid_H * stride_delta_H + offs_T[:] * stride_delta_T                   # (T)
    x_ptrs = x_ptr + pid_B * stride_x_B + pid_H * stride_x_H + offs_T[:, None] * stride_x_T + offs_P[None, :] * stride_x_P  # (T, P)
    y_ptrs = y_ptr + pid_B * stride_y_B + pid_H * stride_y_H + offs_T[:, None] * stride_y_T + offs_P[None, :] * stride_y_P  # (T, P)
    B_ptrs = B_ptr + pid_B * stride_B_B + offs_T[:, None] * stride_B_T + offs_N[None, :] * stride_B_N                       # (T, N)
    C_ptrs = C_ptr + pid_B * stride_C_B + offs_T[:, None] * stride_C_T + offs_N[None, :] * stride_C_N                       # (T, N)
    h_ptrs = h_ptr + pid_B * stride_h_B + pid_H * stride_h_H + offs_N[:, None] * stride_h_N + offs_P[None, :] * stride_h_P  # (N, P)

    # pre-load
    h: tl.tensor = tl.load(h_ptrs)
    
    for block_id in range(0, tl.cdiv(length, BLOCK_LEN), 1):
        offs_block = block_id * BLOCK_LEN
        block_mask = (offs_T + offs_block) < length

        # loads
        A_raw = -tl.exp(tl.load(A_ptrs, mask=block_mask, other=0).cast(tl.float32))         # (T)
        delta = softplus(tl.load(delta_ptrs, mask=block_mask, other=0).cast(tl.float32))    # (T)
        x = tl.load(x_ptrs, mask=block_mask[:, None], other=0).cast(tl.float32)     # (T, P)
        B = tl.load(B_ptrs, mask=block_mask[:, None], other=0).cast(tl.float32)     # (T, N)
        C = tl.load(C_ptrs, mask=block_mask[:, None], other=0).cast(tl.float32)     # (T, N)

        A_log = A_raw * delta
        A_log = tl.where(block_mask, A_log, 0)
        B = B * delta[:, None]
        A_log_cumsum = tl.cumsum(A_log)
        A_log_sum = tl.sum(A_log)

        A_mask = A_log_cumsum[:, None] - A_log_cumsum[None, :]
        A_mask = tl.exp(A_mask)
        A_mask = tl.where(offs_T[:, None] >= offs_T[None, :], A_mask, 0)

        M = tl.dot(C, tl.trans(B, (1, 0))) * A_mask
        y = tl.dot(M, x)
        y += tl.dot(C * tl.exp(A_log_cumsum)[:, None], h)

        right_term_A = tl.exp(A_log_sum - A_log_cumsum)[:, None]
        right_term_AB = B * right_term_A
        h = h * tl.exp(A_log_sum) + tl.dot(tl.trans(right_term_AB, (1, 0)), x)

        # cast dtype (dtype of h always be float32 so not need to cast dtype)
        if data_type == "bfloat16":
            y = y.cast(tl.bfloat16)
        elif data_type == "float16":
            y = y.cast(tl.float16)
        elif data_type == "float64":
            y = y.cast(tl.float64)
        # stores
        h_ptrs += 1 * stride_h_T
        tl.store(h_ptrs, h)
        tl.store(y_ptrs, y, mask=block_mask[:, None])

        # move ptrs
        A_ptrs += BLOCK_LEN * stride_A_T
        x_ptrs += BLOCK_LEN * stride_x_T
        y_ptrs += BLOCK_LEN * stride_y_T
        B_ptrs += BLOCK_LEN * stride_B_T
        C_ptrs += BLOCK_LEN * stride_C_T

@triton.jit
def _mamba2_bwd_kernel(
    # ptr
    A_ptr, delta_ptr, x_ptr, B_ptr, C_ptr, h_ptr,
    dA_ptr, ddelta_ptr, dx_ptr, dy_ptr, dB_ptr, dC_ptr, dhn_ptr, dh0_ptr,
    # shapes
    length, d_head: tl.constexpr, d_state: tl.constexpr,
    # strides
    stride_A_B, stride_A_T, stride_A_H,                 # (B, T, H)
    stride_delta_B, stride_delta_T, stride_delta_H,     # (B, T, H)
    stride_x_B, stride_x_T, stride_x_H, stride_x_P,     # (B, T, H, P)
    stride_y_B, stride_y_T, stride_y_H, stride_y_P,     # (B, T, H, P)
    stride_B_B, stride_B_T, stride_B_N,                 # (B, T, N)
    stride_C_B, stride_C_T, stride_C_N,                 # (B, T, N)
    stride_h_B, stride_h_T, stride_h_H, stride_h_N, stride_h_P,     # (B, T//C, H, N, P)
    stride_dhn_B, stride_dhn_H, stride_dhn_N, stride_dhn_P, # (B, H, N, P)v
    stride_dh0_B, stride_dh0_H, stride_dh0_N, stride_dh0_P, # (B, H, N, P)
    BLOCK_LEN: tl.constexpr, data_type: tl.constexpr
):
    pid_B = tl.program_id(0)
    pid_H = tl.program_id(1)
    offs_T = tl.arange(0, BLOCK_LEN)
    offs_P = tl.arange(0, d_head)
    offs_N = tl.arange(0, d_state)

    # set ptrs
    A_ptrs = A_ptr + pid_B * stride_A_B + pid_H * stride_A_H + offs_T[:] * stride_A_T       # (C)
    delta_ptrs = delta_ptr + pid_B * stride_delta_B + pid_H * stride_delta_H + offs_T[:] * stride_delta_T       # (C)
    x_ptrs = x_ptr + pid_B * stride_x_B + pid_H * stride_x_H + offs_T[:, None] * stride_x_T + offs_P[None, :] * stride_x_P          # (C, P)
    B_ptrs = B_ptr + pid_B * stride_B_B + offs_T[:, None] * stride_B_T + offs_N[None, :] * stride_B_N       # (C, N)
    C_ptrs = C_ptr + pid_B * stride_C_B + offs_T[:, None] * stride_C_T + offs_N[None, :] * stride_C_N       # (C, N)
    h_ptrs = h_ptr + pid_B * stride_h_B + pid_H * stride_h_H + offs_N[:, None] * stride_h_N + offs_P[None, :] * stride_h_P          # (N, P)

    dA_ptrs = dA_ptr + pid_B * stride_A_B + pid_H * stride_A_H + offs_T[:] * stride_A_T     # (C)
    ddelta_ptrs = ddelta_ptr + pid_B * stride_delta_B + pid_H * stride_delta_H + offs_T[:] * stride_delta_T     # (C)
    dx_ptrs = dx_ptr + pid_B * stride_x_B + pid_H * stride_x_H + offs_T[:, None] * stride_x_T + offs_P[None, :] * stride_x_P        # (C, P)
    dy_ptrs = dy_ptr + pid_B * stride_y_B + pid_H * stride_y_H + offs_T[:, None] * stride_y_T + offs_P[None, :] * stride_y_P        # (C, P)
    dB_ptrs = dB_ptr + pid_B * stride_B_B + offs_T[:, None] * stride_B_T + offs_N[None, :] * stride_B_N     # (C, N)
    dC_ptrs = dC_ptr + pid_B * stride_C_B + offs_T[:, None] * stride_C_T + offs_N[None, :] * stride_C_N     # (C, N)
    dhn_ptrs = dhn_ptr + pid_B * stride_dhn_B + pid_H * stride_dhn_H + offs_N[:, None] * stride_dhn_N + offs_P[None, :] * stride_dhn_P      # (N, P)
    dh0_ptrs = dh0_ptr + pid_B * stride_dh0_B + pid_H * stride_dh0_H + offs_N[:, None] * stride_dh0_N + offs_P[None, :] * stride_dh0_P      # (N, P)    

    # pre-load
    dh = tl.load(dhn_ptrs).cast(tl.float32)   # (N, P)

    for i in range(tl.cdiv(length, BLOCK_LEN) - 1, -1, -1):
        block_offs = i * BLOCK_LEN
        block_mask = (offs_T + block_offs) < length

        # loads
        A_pure_raw = tl.load(A_ptrs + block_offs * stride_A_T, mask=block_mask, other=0).cast(tl.float32)
        delta_raw = tl.load(delta_ptrs + block_offs * stride_delta_T, mask=block_mask, other=0).cast(tl.float32)
        x = tl.load(x_ptrs + block_offs * stride_x_T, mask=block_mask[:, None], other=0).cast(tl.float32)       # (C, P)
        B_raw = tl.load(B_ptrs + block_offs * stride_B_T, mask=block_mask[:, None], other=0).cast(tl.float32)   # (C, N)
        C = tl.load(C_ptrs + block_offs * stride_C_T, mask=block_mask[:, None], other=0).cast(tl.float32)       # (C, N)
        h = tl.load(h_ptrs + i * stride_h_T).cast(tl.float32)       # (N, P)
        dy = tl.load(dy_ptrs + block_offs * stride_y_T, mask=block_mask[:, None], other=0).cast(tl.float32)     # (C, P)

        A_raw = -tl.exp(A_pure_raw)
        delta = softplus(delta_raw)
        A_log = A_raw * delta   # (C)
        A_log = tl.where(block_mask, A_log, 0)
        B = B_raw * delta[:, None]
        A_log_cumsum = tl.cumsum(A_log) # (C)
        A_reversed_cumsum = tl.exp(tl.sum(A_log) - A_log_cumsum)

        # A_reversed_mask and A_mask
        A_reversed_mask = A_log_cumsum[None, :] - A_log_cumsum[:, None]
        A_reversed_mask = tl.exp(A_reversed_mask)
        A_reversed_mask = tl.where(offs_T[None, :] >= offs_T[:, None], A_reversed_mask, 0)
        A_mask = A_log_cumsum[:, None] - A_log_cumsum[None, :]
        A_mask = tl.exp(A_mask)
        A_mask = tl.where(offs_T[:, None] >= offs_T[None, :], A_mask, 0)

        # dB
        dB_mask = tl.dot(x, tl.trans(dy, (1, 0))) * A_reversed_mask
        dB = tl.dot(dB_mask, C) + tl.dot(x * A_reversed_cumsum[:, None], tl.trans(dh, (1, 0)))

        # dx
        dx_mask = tl.dot(B, tl.trans(C, (1, 0))) * A_reversed_mask
        dx = tl.dot(dx_mask, dy) + tl.dot(B * A_reversed_cumsum[:, None], dh)

        # dC
        dC_mask = tl.dot(dy, tl.trans(x, (1, 0))) * A_mask
        dC = tl.dot(dC_mask, B) + tl.dot(dy * tl.exp(A_log_cumsum)[:, None], tl.trans(h, (1, 0)))

        # dA = dh_t * h_{t-1}
        # dh for dA => \sum_{i = t}^{C}A_{i : t} C_i^\top dy_i + A_{C : t} dh_{end}
        C_reduced = tl.sum(C, axis=1)   # (C)
        dy_reduced = tl.sum(dy, axis=1) # (C)
        dh_end_reduced = tl.sum(tl.sum(dh, axis=0), axis=0) # (1)
        dh_for_dA = tl.sum(A_reversed_mask * C_reduced[None, :] * dy_reduced[None, :], axis=1)\
                    + A_reversed_cumsum * dh_end_reduced    # (C)
        # h_{t-1} for dA => \frac{\sum_{s = c}^{t} A_{t : s} B_s^\top x_s - A_{t:t} B_t^\top x_t+ A_{t: c-1} h_{c-1}}{A_t}
        B_reduced = tl.sum(B, axis=1)   # (C)
        x_reduced = tl.sum(x, axis=1)   # (C)
        h_start_reduced = tl.sum(tl.sum(h, axis=0), axis=0) # (1)
        h_for_dA = (tl.sum(A_mask * B_reduced[None, :] * x_reduced[None, :], axis=1)\
                    - B_reduced * x_reduced\
                    + tl.exp(A_log_cumsum) * h_start_reduced\
                    ) / tl.exp(A_log)
        dA = dh_for_dA * h_for_dA   # (C)

        # final gradients calculation
        dA_pure_raw = dA * tl.exp(A_log) * A_raw * delta    # (C)
        ddelta_raw = (dA * tl.exp(A_log) * A_raw + tl.sum(dB * B_raw, axis=1)) * sigmoid(delta_raw)     # (C)
        dB_raw = dB * delta[:, None]    # (C, N)

        # update dh
        dh = tl.dot(tl.exp(A_log_cumsum)[None, :] * tl.trans(C, (1, 0)), dy) + tl.exp(tl.sum(A_log)) * dh

        # cast type
        if data_type == "bfloat16":
            dA_pure_raw = dA_pure_raw.cast(tl.bfloat16)
            ddelta_raw = ddelta_raw.cast(tl.bfloat16)
            dx = dx.cast(tl.bfloat16)
            dB_raw = dB_raw.cast(tl.bfloat16)
            dC = dC.cast(tl.bfloat16)
        elif data_type == "float16":
            dA_pure_raw = dA_pure_raw.cast(tl.float16)
            ddelta_raw = ddelta_raw.cast(tl.float16)
            dx = dx.cast(tl.float16)
            dB_raw = dB_raw.cast(tl.float16)
            dC = dC.cast(tl.float16)
        elif data_type == "float64":
            dA_pure_raw = dA_pure_raw.cast(tl.float64)
            ddelta_raw = ddelta_raw.cast(tl.float64)
            dx = dx.cast(tl.float64)
            dB_raw = dB_raw.cast(tl.float64)
            dC = dC.cast(tl.float64)
        # store
        tl.store(dA_ptrs + block_offs * stride_A_T, dA_pure_raw, mask=block_mask)
        tl.store(ddelta_ptrs + block_offs * stride_delta_T, ddelta_raw, mask=block_mask)
        tl.store(dx_ptrs + block_offs * stride_x_T, dx, mask=block_mask[:, None])
        tl.atomic_add(dB_ptrs + block_offs * stride_B_T, dB_raw, mask=block_mask[:, None])
        tl.atomic_add(dC_ptrs + block_offs * stride_C_T, dC, mask=block_mask[:, None])
    if data_type == "bfloat16":
        dh = dh.cast(tl.bfloat16)
    elif data_type == "float16":
        dh = dh.cast(tl.float16)
    elif data_type == "float64":
        dh = dh.cast(tl.float64)
    tl.store(dh0_ptrs, dh)



def mamba2_fwd(A: Tensor, delta: Tensor, x: Tensor, B: Tensor, C: Tensor, h0: Tensor, pytorch: bool = False) -> tuple[Tensor, Tensor]:
    if pytorch:
        return _mamba2_pytorch_fwd(A, delta, x, B, C, h0)
    else:
        return Mamba2Autograd.apply(A, delta, x, B, C, h0) # type: ignore

class Mamba2Autograd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A: Tensor, delta: Tensor, x: Tensor, B: Tensor, C: Tensor, h0: Tensor):
        # get size
        batch_size, length, n_heads = A.shape
        batch_size, n_heads, d_state, d_head = h0.shape
        # view x
        x = x.view(batch_size, length, n_heads, d_head)
        BLOCK_LEN = 64
        # create hiddens
        hiddens = torch.zeros(
            size=(batch_size, triton.cdiv(length, BLOCK_LEN) + 1, n_heads, d_state, d_head),
            dtype=torch.float32, device=A.device,   # the dtype of hiddens should always be float32
        )
        hiddens[:, 0, :, :, :] += h0.to(torch.float32)
        ctx.save_for_backward(A, delta, x, B, C, hiddens)
        ctx.BLOCK_LEN = BLOCK_LEN
        y = torch.empty_like(x)
        # get input dtype
        if x.dtype == torch.bfloat16:
            data_type = "bfloat16"
        elif x.dtype == torch.float16:
            data_type = "float16"
        elif x.dtype == torch.float64:
            data_type = "float64"
        else:
            data_type = "float32"
        ctx.data_type = data_type

        _mamba2_fwd_kernel[(batch_size, n_heads)](
            A, delta, x, y, B, C, hiddens,
            length, d_head, d_state, # type: ignore
            A.stride(0), A.stride(1), A.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            y.stride(0), y.stride(1), y.stride(2), y.stride(3),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            hiddens.stride(0), hiddens.stride(1), hiddens.stride(2), hiddens.stride(3), hiddens.stride(4),
            BLOCK_LEN, data_type # type: ignore
        )
        hn = torch.zeros_like(h0)
        hn += hiddens[:, -1, :, :, :]
        return y, hn

    @staticmethod
    def backward(ctx, dy: Tensor, dhn: Tensor): # type: ignore
        batch_size, length, n_heads, d_head = dy.shape
        batch_size, n_heads, d_state, d_head = dhn.shape
        BLOCK_LEN = ctx.BLOCK_LEN
        data_type = ctx.data_type
        A, delta, x, B, C, hiddens, = ctx.saved_tensors
        A: Tensor; delta: Tensor; x: Tensor; B: Tensor; C: Tensor; hiddens: Tensor  # type hints
        dA = torch.empty_like(A)
        ddelta = torch.empty_like(delta)
        dx = torch.empty_like(x)
        dB = torch.zeros_like(B)
        dC = torch.zeros_like(C)
        dh0 = torch.empty_like(dhn)

        _mamba2_bwd_kernel[(batch_size, n_heads)](
            # ptrs
            A, delta, x, B, C, hiddens,
            dA, ddelta, dx, dy, dB, dC, dhn, dh0,
            length, d_head, d_state,    # type: ignore
            A.stride(0), A.stride(1), A.stride(2),
            delta.stride(0), delta.stride(1), delta.stride(2),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dy.stride(0), dy.stride(1), dy.stride(2), dy.stride(3),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1), C.stride(2),
            hiddens.stride(0), hiddens.stride(1), hiddens.stride(2), hiddens.stride(3), hiddens.stride(4),
            dhn.stride(0), dhn.stride(1), dhn.stride(2), dhn.stride(3),
            dh0.stride(0), dh0.stride(1), dh0.stride(2), dh0.stride(3),
            BLOCK_LEN, data_type
        )
        return dA, ddelta, dx, dB, dC, dh0