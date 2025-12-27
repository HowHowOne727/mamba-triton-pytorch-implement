import torch
import triton
import triton.language as tl


# set your config here
CHECKPOINT = 16
FORWARD_BLOCK_SIZE_D = 16
BACKWARD_BLOCK_SIZE_D = 8

@triton.jit
def mamba_fusion_ssm_kernel(
        delta, A, B, C, x, y, h, h_out,   # ptrs
        Batch_size, length, d_model,   # ints
        delta_stride_B, delta_stride_L, delta_stride_D,
        A_stride_D, A_stride_N,
        B_stride_B, B_stride_L, B_stride_N,
        C_stride_D, C_stride_N,
        x_stride_B, x_stride_L, x_stride_D,
        y_stride_B, y_stride_L, y_stride_D,
        h_stride_B, h_stride_L, h_stride_D, h_stride_N,
        h_out_stride_B, h_out_stride_D, h_out_stride_N,
        BLOCK_SIZE_D: tl.constexpr, n_state: tl.constexpr, checkpoint: tl.constexpr
):
    """
    mamba_fusion_ssm_kernel 的 Docstring
    
    :param delta: (B , L)
    :param A: (D , N)
    :param B: (D , L , N)
    :param C: (D , N)
    :param x: (B , L , D)
    :param y: (B , L , D)
    :param h0: (B , D , N)
    :param h_out: (B , D , N)
    """
    pid_B = tl.program_id(0)
    pid_D = tl.program_id(1)

    offs_B = pid_B + tl.arange(0 , 1)
    offs_L = tl.arange(0, checkpoint)
    offs_D = pid_D * BLOCK_SIZE_D + tl.arange(0 , BLOCK_SIZE_D)
    offs_N = tl.arange(0 , n_state)

    # load delta
    delta_ptr = delta + (offs_B[:, None, None] * delta_stride_B + offs_L[None, :, None] * delta_stride_L + offs_D[None, None, :] * delta_stride_D)   # (B, L, D)

    # load hidden
    h0_ptr = h + (offs_B[:, None, None] * h_stride_B + offs_D[None, :, None] * h_stride_D + offs_N[None, None, :] * h_stride_N)   # (B, D, N)
    h_cp_ptr = h + (offs_B[:, None, None] * h_stride_B + offs_D[None, :, None] * h_stride_D + offs_N[None, None, :] * h_stride_N)     # (B, D, N)
    
    hidden_tile: tl.tensor = tl.load(h0_ptr)     # (B, D, N)

    # load A
    A_ptr = A + (offs_D[: , None] * A_stride_D + offs_N[None , :] * A_stride_N)
    A_tile: tl.tensor = tl.load(A_ptr)    # (D , N)

    # set B ptr
    B_ptr = B + (offs_B[:, None, None] * B_stride_B + offs_L[None, :, None] * B_stride_L + offs_N[None, None, :] * B_stride_N)   # (B, L, N)

    # load C
    C_ptr = C + (offs_D[: , None] * C_stride_D + offs_N[None , :] * C_stride_N)
    C_tile: tl.tensor = tl.load(C_ptr)    # (D , N)

    # x and y
    x_ptr = x + (offs_B[:, None, None] * x_stride_B + offs_L[None, :, None] * x_stride_L + offs_D[None, None, :] * x_stride_D)  # (B, L, D)
    y_ptr = y + (offs_B[:, None, None] * y_stride_B + offs_L[None, :, None] * y_stride_L + offs_D[None, None, :] * y_stride_D)  # (B, L, D)

    for steps in range(0, length, checkpoint):     # main loop
        mask_L = offs_L + steps < length

        # chunked loads
        x_tile = tl.load(x_ptr + steps * x_stride_L, mask=mask_L[None, :, None])              # (B, L, D)
        delta_tile = tl.load(delta_ptr + steps * delta_stride_L, mask=mask_L[None, :, None])[:, :, :, None]     # (B, L, D) -> (B, L, D, N)
        B_tile = tl.load(B_ptr + steps * B_stride_L, mask=mask_L[None, :, None])[:, :, None, :]          # (B, L, N) -> (B, L, D, N)

        _out = tl.zeros_like(x_tile)    # (B, L, D)

        if steps > 0:   # save hidden checkpoint
            tl.store(h_cp_ptr + (steps//checkpoint) * h_stride_L, hidden_tile)

        A_hat = tl.exp(delta_tile * A_tile[None, None, :, :])     # (B, L, D, N)
        B_hat = delta_tile * B_tile * x_tile[:, :, :, None]     # (B, L, D, N)

        for i in range(checkpoint):
            A_hat_t = tl.sum(tl.where(((offs_L == i) & mask_L)[None, :, None, None], A_hat, 0.0), axis=1)    # (B, D, N)
            B_hat_t = tl.sum(tl.where(((offs_L == i) & mask_L)[None, :, None, None], B_hat, 0.0), axis=1)    # (B, D, N)
            hidden_tile = A_hat_t * hidden_tile + B_hat_t

            _out = tl.where(offs_L[None, :, None] == i, tl.sum(hidden_tile * C_tile[None, :, :], axis=-1)[:, None, :], _out)

        tl.store(y_ptr + steps * y_stride_L , _out, mask=mask_L[None, :, None])

    # save h_out
    h_out_ptr = h_out + (offs_B[:, None, None] * h_out_stride_B + offs_D[None, :, None] * h_out_stride_D + offs_N[None, None, :] * h_out_stride_N)
    tl.store(h_out_ptr , hidden_tile)

@triton.jit
def mamba_fusion_ssm_backward_kernel(
        # tensor pointers
        delta, A, B, C, x, h_cp,     # input value, h_cp : hidden checkpoints
        g_delta, g_A, g_B, g_C, g_x, g_h0,  # gradients to store
        g_y, g_h_out,           # upstream gradients

        Batch_size, Length, d_model,
        delta_stride_B, delta_stride_L, delta_stride_D,
        A_stride_D, A_stride_N,
        B_stride_B, B_stride_L, B_stride_N,
        C_stride_D, C_stride_N,
        x_stride_B, x_stride_L, x_stride_D,
        h_cp_stride_B, h_cp_stride_L, h_cp_stride_D, h_cp_stride_N,
        g_h0_stride_B, g_h0_stride_D, g_h0_stride_N,

        g_y_stride_B, g_y_stride_L, g_y_stride_D,
        g_h_out_stride_B, g_h_out_stride_D, g_h_out_stride_N,
        BLOCK_SIZE_D: tl.constexpr, n_states: tl.constexpr, checkpoints: tl.constexpr
):
    pid_B = tl.program_id(0)
    pid_D = tl.program_id(1)

    offs_B = pid_B + tl.arange(0 , 1)
    offs_L = tl.arange(0, checkpoints)
    offs_D = pid_D * BLOCK_SIZE_D + tl.arange(0 , BLOCK_SIZE_D)
    offs_N = tl.arange(0 , n_states)

    # load A, C
    A_ptr = A + (offs_D[:, None] * A_stride_D + offs_N[None, :] * A_stride_N)
    C_ptr = C + (offs_D[:, None] * C_stride_D + offs_N[None, :] * C_stride_N)

    B_ptr = B + (offs_B[:, None, None] * B_stride_B + offs_L[None, :, None] * B_stride_L + offs_N[None, None, :] * B_stride_N)    # (B, L, N)
    g_B_ptr = g_B + (offs_B[:, None, None] * B_stride_B + offs_L[None, :, None] * B_stride_L + offs_N[None, None, :] * B_stride_N)

    A_tile = tl.load(A_ptr)
    C_tile = tl.load(C_ptr)

    g_A_tile = tl.zeros_like(A_tile[None , : , :])
    g_C_tile = tl.zeros_like(C_tile[None , : , :])

    # pre-calculate pointers
    delta_ptr = delta + (offs_B[:, None, None] * delta_stride_B + offs_L[None, :, None] * delta_stride_L + offs_D[None, None, :] * delta_stride_D)     # (B, L, D)
    g_delta_ptr = g_delta + (offs_B[:, None, None] * delta_stride_B + offs_L[None, :, None] * delta_stride_L + offs_D[None, None, :] * delta_stride_D)

    x_ptr = x + (offs_B[:, None, None] * x_stride_B + offs_L[None, :, None] * x_stride_L + offs_D[None, None, :] * x_stride_D)     # (B, L, D)
    g_x_ptr = g_x + (offs_B[:, None, None] * x_stride_B + offs_L[None, :, None] * x_stride_L + offs_D[None, None, :] * x_stride_D)

    h_cp_ptr = h_cp + (offs_B[: , None , None] * h_cp_stride_B + offs_D[None , : , None] * h_cp_stride_D + offs_N[None , None , :] * h_cp_stride_N)     # (B, D, N)

    g_y_ptr = g_y + (offs_B[:, None, None] * g_y_stride_B + offs_L[None, :, None] * g_y_stride_L + offs_D[None , None , :] * g_y_stride_D)      # (B, L, D)

    # load g_h_out
    g_h_out_ptr = g_h_out + (offs_B[: , None , None] * g_h_out_stride_B + offs_D[None , : , None] * g_h_out_stride_D + offs_N[None , None , :] * g_h_out_stride_N)  # (B, D, N)
    g_h_tile = tl.load(g_h_out_ptr)     # initialize with the last one

    for block_id in range((Length // checkpoints) , -1 , -1):
        mask_L = offs_L + block_id * checkpoints < Length
        # loads
        x_tile = tl.load(x_ptr + block_id * checkpoints * x_stride_L, mask=mask_L[None, :, None])   # (B, L, D)
        delta_tile = tl.load(delta_ptr + block_id * checkpoints * delta_stride_L, mask=mask_L[None, :, None])[:, :, :, None]   # (B, L, D) -> (B, L, D, N)
        B_tile = tl.load(B_ptr + block_id * checkpoints * B_stride_L, mask=mask_L[None, :, None])[:, :, None, :]   # (B, L, N) -> (B, L, D, N)
        hidden_curr = tl.load(h_cp_ptr + block_id * h_cp_stride_L)  # (B, D, N)
        g_y_tile = tl.load(g_y_ptr + block_id * checkpoints * g_y_stride_L, mask=mask_L[None, :, None])        # (B, L, D)

        A_hat = tl.exp(delta_tile * A_tile[None, None, :, :])    # (B, L, D, N)
        A_hat = tl.where(mask_L[None, :, None, None], A_hat, 1.0)
        B_hat = delta_tile * B_tile * x_tile[:, :, :, None]

        hidden_all = tl.zeros_like(A_hat)   # (B, L, D, N)
        hidden_all = tl.where(offs_L[None, :, None, None] == 0, hidden_curr[:, None, :, :], hidden_all) # set first to checkpoint hidden

        # forward for h_block
        for i in range(checkpoints):
            A_hat_t = tl.sum(tl.where(((offs_L == i) & mask_L)[None, :, None, None], A_hat, 0.0), axis=1)    # (B, D, N)
            B_hat_t = tl.sum(tl.where(((offs_L == i) & mask_L)[None, :, None, None], B_hat, 0.0), axis=1)    # (B, D, N)
            hidden_curr = A_hat_t * hidden_curr + B_hat_t

            g_C_tile += tl.sum(tl.where(offs_L[None, :, None] == i, g_y_tile, 0.0), axis=1)[:, :, None] * hidden_curr   # compute C gradient here

            hidden_all = tl.where(offs_L[None, :, None, None] == (i + 1) , hidden_curr[: , None , : , :] , hidden_all)   # write recomputed hidden to SRAM
        
        # backward
        g_hidden_all = g_y_tile[:, :, :, None] * C_tile[None, None, :, :]

        for i in range(checkpoints - 1 , -1 , -1):  # reverse loop for calculate hidden gradient
            g_h_tile += tl.sum(tl.where(offs_L[None, :, None, None] == i, g_hidden_all, 0.0), axis=1)
            g_hidden_all = tl.where(offs_L[None, :, None, None] == i, g_h_tile[:, None, :, :], g_hidden_all)
            g_h_tile *= tl.sum(tl.where(offs_L[None, :, None, None] == i, A_hat, 0.0), axis=1)

        g_delta_tile = tl.sum(g_hidden_all * (hidden_all * A_hat * A_tile[None, None, :, :] + B_tile * x_tile[:, :, :, None]), axis=-1)
        g_A_tile += tl.sum(g_hidden_all * A_hat * delta_tile, axis=1)
        g_B_tile = tl.sum(g_hidden_all * delta_tile * x_tile[:, :, :, None], axis=2)    # (B, L, D, N)
        g_x_tile = tl.sum(g_hidden_all * delta_tile * B_tile, axis=-1)                   # (B, L, D, N)

        # stores
        tl.store(g_delta_ptr + block_id * checkpoints * delta_stride_L, g_delta_tile, mask=mask_L[None, :, None])
        tl.store(g_B_ptr + block_id * checkpoints * B_stride_L, g_B_tile, mask=mask_L[None, :, None])
        tl.store(g_x_ptr + block_id * checkpoints * x_stride_L, g_x_tile, mask=mask_L[None, :, None])
    
    # save A, C, h0 gradient
    g_A_tile = tl.sum(g_A_tile , axis=0)    # (B , D , N) to (D , N)
    g_C_tile = tl.sum(g_C_tile , axis=0)

    g_A_ptr = g_A + (offs_D[: , None] * A_stride_D + offs_N[None , :] * A_stride_N)
    g_C_ptr = g_C + (offs_D[: , None] * C_stride_D + offs_N[None , :] * C_stride_N)

    tl.atomic_add(g_A_ptr , g_A_tile)
    tl.atomic_add(g_C_ptr , g_C_tile)

    g_h0_ptr = g_h0 + (offs_B[: , None , None] * g_h0_stride_B + offs_D[None , : , None] * g_h0_stride_D + offs_N[None , None , :] * g_h0_stride_N)
    tl.store(g_h0_ptr , g_h_tile)


def mamba_fusion_ssm(
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor,
        h0: torch.Tensor|None = None,
        checkpoint_len: int = CHECKPOINT,
        forward_block_size_d: int = FORWARD_BLOCK_SIZE_D,
        backward_block_size_d: int = BACKWARD_BLOCK_SIZE_D
) -> tuple[torch.Tensor, torch.Tensor]:
    if h0 is None:
        h0 = torch.zeros(size=(x.size(0) , x.size(-1) , A.size(-1)) , dtype=x.dtype , device=x.device)
    y, h_out = MambaFusionSSMFunction.apply(delta , A , B , C , x , h0 , checkpoint_len, forward_block_size_d, backward_block_size_d) # type: ignore
    return y, h_out

class MambaFusionSSMFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        delta: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        x: torch.Tensor,
        h0: torch.Tensor,
        checkpoint_len:int,
        forward_block_size_d: int,
        backward_block_size_d: int
    ):
        BLOCK_SIZE_D = forward_block_size_d
        Batch_size , Length , D_model = x.shape
        n_state = A.size(-1)

        h_checkpoint = torch.zeros(size=(Batch_size , (Length // checkpoint_len) + 1 , D_model , n_state) , dtype=x.dtype , device=x.device)
        h_checkpoint[: , 0 , : , :] = h0

        ctx.checkpoint_len = checkpoint_len
        ctx.backward_block_size_d = backward_block_size_d
        ctx.save_for_backward(delta , A , B , C , x , h_checkpoint)

        y = torch.empty_like(x, device=x.device)
        h_out = torch.empty_like(h0, device=h0.device)

        grid = (Batch_size , triton.cdiv(D_model , BLOCK_SIZE_D))
        mamba_fusion_ssm_kernel[grid](delta, A, B, C, x, y, h_checkpoint, h_out,
                                    Batch_size, Length, D_model,
                                    delta.stride(0), delta.stride(1), delta.stride(2),
                                    A.stride(0) , A.stride(1),
                                    B.stride(0), B.stride(1), B.stride(2),
                                    C.stride(0), C.stride(1),
                                    x.stride(0), x.stride(1), x.stride(2),
                                    y.stride(0), y.stride(1), y.stride(2),
                                    h_checkpoint.stride(0), h_checkpoint.stride(1), h_checkpoint.stride(2), h_checkpoint.stride(3),
                                    h_out.stride(0), h_out.stride(1), h_out.stride(2),
                                    BLOCK_SIZE_D, n_state, checkpoint_len) # type: ignore

        return y , h_out
    
    @staticmethod
    def backward(ctx, grad_y: torch.Tensor, grad_h_out: torch.Tensor): # type: ignore
        checkpoint_len = ctx.checkpoint_len
        delta, A, B, C, x, h_checkpoint, = ctx.saved_tensors
        delta: torch.Tensor
        A: torch.Tensor
        B: torch.Tensor
        C: torch.Tensor
        x: torch.Tensor
        h_checkpoint: torch.Tensor

        BLOCK_SIZE_D = ctx.backward_block_size_d
        Batch_size, Length, D_model = x.shape
        n_state = A.size(-1)

        g_delta = torch.zeros_like(delta)
        g_A = torch.zeros_like(A)
        g_B = torch.zeros_like(B)
        g_C = torch.zeros_like(C)
        g_x = torch.zeros_like(x)
        g_h0 = torch.zeros(size=(Batch_size , D_model , n_state) , dtype=h_checkpoint.dtype , device=h_checkpoint.device)

        grid = (Batch_size , triton.cdiv(D_model , BLOCK_SIZE_D))
        mamba_fusion_ssm_backward_kernel[grid](
            delta, A, B, C, x, h_checkpoint,
            g_delta, g_A, g_B, g_C, g_x, g_h0,
            grad_y, grad_h_out,
            Batch_size, Length, D_model,
            delta.stride(0), delta.stride(1), delta.stride(2),
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1), B.stride(2),
            C.stride(0), C.stride(1),
            x.stride(0), x.stride(1), x.stride(2),
            h_checkpoint.stride(0), h_checkpoint.stride(1), h_checkpoint.stride(2), h_checkpoint.stride(3),
            g_h0.stride(0), g_h0.stride(1), g_h0.stride(2),
            grad_y.stride(0), grad_y.stride(1), grad_y.stride(2),
            grad_h_out.stride(0), grad_h_out.stride(1), grad_h_out.stride(2),
            BLOCK_SIZE_D, n_state, checkpoint_len   # type: ignore
        )

        return g_delta, g_A, g_B, g_C, g_x, g_h0, None, None, None