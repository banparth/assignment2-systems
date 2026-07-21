import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock
from torch.utils.checkpoint import checkpoint

device = torch.device("cuda")

# num_layers for this model is 32
d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
                            positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length)).to(device=device)


# Fuse as much torch.compile will allow
block = block
x = torch.randn((4, context_length, d_model), requires_grad=True, device=device)

torch.cuda.synchronize()
# Now logs the number of bytes saved
total_size_bytes = 0
def pack_hook(t: torch.Tensor):
    if isinstance(t, torch.nn.Parameter): # Skip logging parameters to avoid double counting
        return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    return t

def full(x) -> torch.Tensor:
    for i in range (0, 32, 2):
        def mid(x):  
            for j in range (0, 2):
                x = block(x)
            return x
        x = checkpoint(mid, x, use_reentrant=False)
        
    # x =  checkpoint(block, x, use_reentrant=False)
    # x = checkpoint( block, x, use_reentrant=False)
    # x = checkpoint( block, x, use_reentrant=False)
    # x = checkpoint( block, x, use_reentrant=False)
    return x

def run(x):
    y = full(x)
    y.sum().backward()
    
for _ in range(5):
    run(x)

# Run forward pass, saving for backward
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        run(x)
        torch.cuda.memory._dump_snapshot(f"memory_dump/memory_residual.pickle")
        torch.cuda.memory._record_memory_history(enabled=None)


print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes /(1024**2):.2f} MiB")