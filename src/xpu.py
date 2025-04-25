import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch


def worker(size):
    a = np.random.randn(size, size)
    b = np.random.randn(size, size)
    return np.dot(a, b)


def cpu(size):
    start = time.time()
    _ = worker(size)
    return time.time() - start


def parallel_cpu(size, workers=None):
    if workers is None:
        workers = cpu_count()  # Automatically detect number of CPU cores
    with Pool(workers) as p:
        start = time.time()
        _ = p.map(worker, [size] * 100)
        return (time.time() - start) / 100


def gpu(size):
    if not torch.cuda.is_available():
        return float("nan")

    device = torch.device("cuda")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm up
    for _ in range(5):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        _ = torch.mm(a, b)
    torch.cuda.synchronize()
    return (time.time() - start) / 100


def hybrid(size):
    if not torch.cuda.is_available():
        return float("nan")

    device = torch.device("cuda")
    split_point = size // 2  # Split work 50/50

    # Create full matrices
    a = torch.randn(size, size)
    b = torch.randn(size, size)

    # GPU part (first half)
    a_gpu = a[:split_point].to(device)
    b_gpu = b[:, :split_point].to(device)

    # CPU part (second half)
    a_cpu = a[split_point:]
    b_cpu = b[:, split_point:]

    # Warm up
    for _ in range(5):
        gpu_res = torch.mm(a_gpu, b_gpu)
        cpu_res = torch.mm(a_cpu, b_cpu)
    torch.cuda.synchronize()

    start = time.time()
    for _ in range(100):
        gpu_res = torch.mm(a_gpu, b_gpu)
        cpu_res = torch.mm(a_cpu, b_cpu)
        # Combine results properly
        combined = torch.cat(
            [
                torch.cat(
                    [gpu_res.cpu(), torch.zeros(split_point, size - split_point)], dim=1
                ),
                torch.cat(
                    [torch.zeros(size - split_point, split_point), cpu_res], dim=1
                ),
            ],
            dim=0,
        )
    torch.cuda.synchronize()
    return (time.time() - start) / 100


def benchmark(size=2048):
    print(f"\n{'='*40}")
    print(f"Matrix Multiplication Benchmark (Size: {size}x{size})")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"CPU cores: {cpu_count()}")
    print(f"{'='*40}\n")

    results = {}

    # Single CPU core
    torch.set_num_threads(1)
    results["Single CPU"] = cpu(size)

    # Multi-core CPU
    torch.set_num_threads(torch.get_num_threads())
    results["Multi-CPU"] = parallel_cpu(size)  # Now automatically uses all cores

    # GPU and Hybrid
    if torch.cuda.is_available():
        results["GPU"] = gpu(size)
        results["Hybrid"] = hybrid(size)

    # Print results
    print("{:<20} {:<15}".format("Configuration", "Time per matmul (s)"))
    print("-" * 35)
    for config, time_val in results.items():
        if not np.isnan(time_val):
            print("{:<20} {:.6f}".format(config, time_val))
        else:
            print("{:<20} {}".format(config, "N/A"))

    # Speedup comparison
    if "GPU" in results and not np.isnan(results["GPU"]):
        print("\nSpeedup Factors (vs Single CPU):")
        base = results["Single CPU"]
        for config, time_val in results.items():
            if not np.isnan(time_val):
                print(f"{config:<20} {base/time_val:.2f}x")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    benchmark(size=2048)
