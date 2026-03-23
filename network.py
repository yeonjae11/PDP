import torch.distributed as dist
import torch, time

# 예: 1MB와 100MB 크기의 텐서를 준비
for size in [1_000_000, 100_000_000]:
    tensor = torch.ones(size, dtype=torch.float32, device='cuda')
    dist.barrier()  # 동기화
    start = time.time()
    dist.all_reduce(tensor)  # 모든 GPU에 대해 All-Reduce 수행
    dist.barrier()
    elapsed_ms = (time.time() - start) * 1000
    print(f"Size {size} bytes: {elapsed_ms:.3f} ms")