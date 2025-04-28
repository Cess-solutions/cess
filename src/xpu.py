import time
from multiprocessing import Pool, cpu_count

import numpy as np
import torch


class XPU:
    def __init__(self):
        """Initialize the XPU processor"""
        pass

    @staticmethod
    def worker(size):
        """Worker function for parallel processing"""
        a = np.random.randn(size, size)
        b = np.random.randn(size, size)
        return np.dot(a, b)

    def cpu(self, size):
        """Single CPU core benchmark"""
        start = time.time()
        _ = self.worker(size)
        return time.time() - start

    def parallel_cpu(self, size, workers=None):
        """Parallel CPU benchmark"""
        if workers is None:
            workers = cpu_count()  # Automatically detect number of CPU cores
        with Pool(workers) as p:
            start = time.time()
            _ = p.map(self.worker, [size] * 100)
            return (time.time() - start) / 100

    def gpu(self, size):
        """GPU benchmark"""
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

    def hybrid(self, size):
        """Hybrid CPU/GPU benchmark"""
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
            torch.mm(a_gpu, b_gpu)  # Removed unused assignment
            torch.mm(a_cpu, b_cpu)  # Removed unused assignment
        torch.cuda.synchronize()

        start = time.time()
        for _ in range(100):
            # Calculate results without storing in unused variables
            torch.mm(a_gpu, b_gpu)
            torch.mm(a_cpu, b_cpu)
        torch.cuda.synchronize()
        return (time.time() - start) / 100

    def benchmark(self, size=2048):
        """Run complete benchmark suite"""
        print(f"\n{'=' * 40}")
        print(f"Matrix Multiplication Benchmark (Size: {size}x{size})")
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CPU cores: {cpu_count()}")
        print(f"{'=' * 40}\n")

        results = {}

        # Single CPU core
        torch.set_num_threads(1)
        results["Single CPU"] = self.cpu(size)

        # Multi-core CPU
        torch.set_num_threads(torch.get_num_threads())
        results["Multi-CPU"] = self.parallel_cpu(size)

        # GPU and Hybrid
        if torch.cuda.is_available():
            results["GPU"] = self.gpu(size)
            results["Hybrid"] = self.hybrid(size)

        # Print results
        print(f"{'Configuration':<20} {'Time per operation (s)':<15}")
        print("-" * 35)
        for config, time_val in results.items():
            if not np.isnan(time_val):
                print(f"{config:<20} {time_val:.6f}")  # Changed to f-string
            else:
                print(f"{config:<20} {'N/A'}")  # Changed to f-string

        # Speedup comparison
        if "GPU" in results and not np.isnan(results["GPU"]):
            print("\nSpeedup Factors (vs Single CPU):")
            base = results["Single CPU"]
            for config, time_val in results.items():
                if not np.isnan(time_val):
                    print(f"{config:<20} {base / time_val:.2f}x")


if __name__ == "__main__":
    import multiprocessing

    multiprocessing.freeze_support()
    xpu = XPU()
    xpu.benchmark(size=2048)
