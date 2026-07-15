"""Exploratory E5 probe: does the swap-in/swap-out ratio rho survive contention?

Measures pinned D2H (swap-out) and H2D (swap-in) of a KV-block-sized buffer
(1) idle, (2) under a concurrent GPU matmul load (other requests' inference),
(3) under a concurrent host-memory-bandwidth load (a tool-call container on the
host), (4) both. Reports median ms, GB/s, and rho for each condition.

Exploratory only — not a committed pipeline artifact. If rho moves materially
under contention, a rigorous E5 harness is warranted.
"""

import json
import statistics
import threading
import time

import torch

BYTES_PER_TOKEN = 98304  # Qwen3-Coder-30B bf16 KV, from the rho measurement
SIZES_TOKENS = [8192, 65536, 131072]  # 768 MiB, 6 GiB, 12 GiB
REPEATS = 20


def time_transfers(nbytes: int, repeats: int) -> tuple[float, float]:
    gpu = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    host = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    for _ in range(3):  # warmup
        host.copy_(gpu, non_blocking=True); torch.cuda.synchronize()
        gpu.copy_(host, non_blocking=True); torch.cuda.synchronize()
    out_ms, in_ms = [], []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(repeats):
        torch.cuda.synchronize(); start.record()
        host.copy_(gpu, non_blocking=True)  # swap-out D2H
        end.record(); torch.cuda.synchronize()
        out_ms.append(start.elapsed_time(end))
        torch.cuda.synchronize(); start.record()
        gpu.copy_(host, non_blocking=True)  # swap-in H2D
        end.record(); torch.cuda.synchronize()
        in_ms.append(start.elapsed_time(end))
    del gpu, host
    torch.cuda.empty_cache()
    return statistics.median(out_ms), statistics.median(in_ms)


class Load:
    def __init__(self, gpu: bool, host: bool):
        self.gpu, self.host = gpu, host
        self.stop = threading.Event()
        self.threads: list[threading.Thread] = []

    def _gpu_loop(self):
        # Separate stream so it competes for SMs/HBM, not the default stream.
        s = torch.cuda.Stream()
        a = torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16)
        with torch.cuda.stream(s):
            while not self.stop.is_set():
                a = a @ a
                a = a / a.norm()

    def _host_loop(self):
        # Saturate host DRAM bandwidth with large pageable memcpys.
        import numpy as np
        x = np.empty(1 << 30, dtype=np.uint8)  # 1 GiB
        y = np.empty(1 << 30, dtype=np.uint8)
        while not self.stop.is_set():
            y[:] = x

    def __enter__(self):
        if self.gpu:
            t = threading.Thread(target=self._gpu_loop, daemon=True)
            t.start(); self.threads.append(t)
        if self.host:
            for _ in range(8):  # 8 CPU threads hammering DRAM
                t = threading.Thread(target=self._host_loop, daemon=True)
                t.start(); self.threads.append(t)
        time.sleep(2)  # let load ramp
        return self

    def __exit__(self, *a):
        self.stop.set()
        time.sleep(0.5)


def run():
    assert torch.cuda.is_available()
    conditions = {
        "idle": (False, False),
        "gpu_load": (True, False),
        "host_load": (False, True),
        "both": (True, True),
    }
    results = {"device": torch.cuda.get_device_name(0), "conditions": {}}
    for name, (g, h) in conditions.items():
        rows = []
        with Load(gpu=g, host=h):
            for tok in SIZES_TOKENS:
                nbytes = tok * BYTES_PER_TOKEN
                out_ms, in_ms = time_transfers(nbytes, REPEATS)
                rows.append({
                    "tokens": tok,
                    "MiB": round(nbytes / 1024 / 1024, 1),
                    "swap_out_ms": round(out_ms, 3),
                    "swap_in_ms": round(in_ms, 3),
                    "out_GBps": round(nbytes / 1e9 / (out_ms / 1e3), 2),
                    "in_GBps": round(nbytes / 1e9 / (in_ms / 1e3), 2),
                    "rho": round(in_ms / out_ms, 4),
                })
        results["conditions"][name] = rows
        rho_med = statistics.median(r["rho"] for r in rows)
        out_med = statistics.median(r["out_GBps"] for r in rows)
        print(f"{name:10s} rho={rho_med:.3f}  out={out_med:.1f} GB/s  "
              f"(sizes {[r['rho'] for r in rows]})")
    with open("/root/rho_contention_probe.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote /root/rho_contention_probe.json")


if __name__ == "__main__":
    run()
