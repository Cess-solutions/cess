# Matrix Multiplication Benchmark (CPU vs GPU vs Hybrid)

A Python benchmark comparing matrix multiplication performance across:
- **Single-Core CPU**
- **Multi-Core CPU (Parallelized)**
- **GPU (CUDA)**
- **Hybrid (CPU + GPU)**

Uses **NumPy**, **PyTorch**, and Python's `multiprocessing` for optimized computations.

## 🚀 Features
- **Automatic hardware detection** (CPU cores, GPU availability)
- **Warm-up runs** to avoid cold-start bias
- **Parallel CPU computation** using `multiprocessing.Pool`
- **GPU-accelerated matrix multiplication** via PyTorch (`torch.mm`)
- **Hybrid CPU+GPU mode** for splitting workloads

## ⚙️ Installation
1. **Requirements**:
   - Python 3.10+
   - NVIDIA GPU (for CUDA support) *(Optional)*

2. **Install dependencies**:
   ```bash
    pip install torch numpy
    ```
## 🛠️  Command Usage

To run the matrix operation script, use the following command:

```bash
python xpu.py --size <matrix_size>
```
## 📜 License

MIT License. Free for academic and commercial use.

## 🔗 Links

- [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.mm.html)
- [NumPy xpu](https://numpy.org/doc/stable/reference/generated/numpy.xml.html)

### 🎯 Key Takeaway

This benchmark helps **choose the best hardware** for matrix-heavy workloads (AI, simulations, etc.)
