import torch
import triton
import triton.testing
from mamba.ssm_kernel import mamba_fusion_ssm


def benchmark_mamba(L: int, D: int, N: int, configs: list[dict]):
    r"""
    note that D should be d_model * expansion_factor in your model
    """
    DEVICE = "cuda"
    x = torch.randn(size=(1, L, D), device=DEVICE, requires_grad=True)
    delta = torch.randn(size=(1, L, D), device=DEVICE, requires_grad=True)
    A = -torch.rand(size=(D, N), device=DEVICE, requires_grad=True)
    B = torch.randn(size=(1, L, N), device=DEVICE, requires_grad=True)
    C = torch.randn(size=(D, N), device=DEVICE, requires_grad=True)

    for i in range(len(configs)):
        config = configs[i]
        checkpoint = config['checkpoint']
        forward_block_size_d = config['forward_block_size_d']
        backward_block_size_d = config['backward_block_size_d']

        def full_step():
            out, hn = mamba_fusion_ssm(delta, A, B, C, x, checkpoint_len=checkpoint, forward_block_size_d=forward_block_size_d, backward_block_size_d=backward_block_size_d)
            y = out.sum()
            y.backward(retain_graph=True)

        ms = triton.testing.do_bench(full_step)
        configs[i]['ms'] = ms
        print(configs[i])
    
    print("\n\n\nresult:")
    configs.sort(key=lambda c : c['ms'])
    for c in configs:
        print(c)

def _exhaustive_next(curr: list[int], lens: list[int]) -> tuple[list[int], bool]:
    assert len(curr) == len(lens)
    is_end = False
    for i in range(len(curr)-1, -1, -1):
        curr[i] += 1
        if curr[i] == lens[i]:
            curr[i] = 0
            if i == 0:
                is_end = True
        else:
            break
    return curr, is_end

def make_config(checkpoints: list[int], forward_block_size_ds: list[int], backward_block_size_ds: list[int]):
    idxs = [0, 0, 0]
    lens = [len(checkpoints), len(forward_block_size_ds), len(backward_block_size_ds)]
    configs = []
    is_end = False
    while not is_end:
        configs.append({'checkpoint' : checkpoints[idxs[0]], 'forward_block_size_d' : forward_block_size_ds[idxs[1]], 'backward_block_size_d': backward_block_size_ds[idxs[2]]})
        idxs, is_end = _exhaustive_next(idxs, lens)
    return configs
    
    
# example bench
if __name__ == "__main__":
    configs = make_config([16, 32, 64], [4, 8, 16, 32], [2, 4, 8, 16, 32])
    print(len(configs))
    benchmark_mamba(1024, 768, 16, configs)