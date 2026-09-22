"""Two-node torch.distributed connectivity check (gloo then NCCL) via env://."""
import os, time, socket, datetime, torch, torch.distributed as dist
host = socket.gethostname().split(".")[0]
def run(backend):
    t = time.time()
    dist.init_process_group(backend, init_method="env://", timeout=datetime.timedelta(seconds=120))
    x = torch.ones(1, device="cuda" if backend == "nccl" else "cpu")
    dist.all_reduce(x)
    print(f"{host} rank {os.environ['RANK']}: {backend} all_reduce ok -> {x.item()} in {time.time()-t:.1f}s", flush=True)
    dist.destroy_process_group()
for b in ("gloo", "nccl"):
    try:
        run(b)
    except Exception as e:
        print(f"{host} rank {os.environ['RANK']}: {b} FAILED {str(e)[:300]}", flush=True)
