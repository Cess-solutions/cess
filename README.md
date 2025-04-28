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

### Requirements
- **Python 3.10+**
- **NVIDIA GPU** (for CUDA/GPU support) *(Optional)*
- **Multicore CPU** (for parallel processing)

### 1. Install dependencies
```bash
pip install torch numpy
```
### 2. Set-up the project
- **Clone the repository (if not already done)**

```bash
git clone https://github.com/cess-solutions/cess.git
cd cess
```

- **CCreate and activate virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows
```

- **Install in development mode**
```bash
pip install -e .
```
## 🚀 Advanced Options

- **Run with custom matrix size (e.g., 1024x1024)**
```bash
python app.py --size 1024
```

- **Run specific processing method only:**
```bash
python app.py --method gpu  # Options: cpu, multicpu, gpu, hybrid
```

## Docker Support (Optional)

- **For CPU-only version:**
```bash 
docker build -f docker/Dockerfile.cpu -t cess-cpu .
```

- **For GPU-accelerated version:**
```bash 
docker build -f docker/Dockerfile.gpu -t cess-gpu .
```

- **Run container**

```bash 
docker run --rm cess-cpu  #or cess-gpu with NVIDIA runtime
```

## 🛠️ Development

To contribute:

- **Install development dependencies**
```bash
pip install -e ".[dev]"
```
- **Run tests**
```bash 
pytest tests/
```

# Key improvements made:
1. Organized installation steps more clearly
2. Added specific Python version requirement
3. Included both basic and advanced usage examples
4. Added Docker usage instructions matching your existing Dockerfiles
5. Show example output format
6. Added development section for contributors
7. Made GPU requirements clearer as optional
8. Added multicore CPU as a requirement
9. Included both CLI and programmatic usage options

## 📜 License

MIT License. Free for academic and commercial use.

## 🔗 Links

- [PyTorch Docs](https://pytorch.org/docs/stable/generated/torch.mm.html)
- [NumPy xpu](https://numpy.org/doc/stable/reference/generated/numpy.xml.html)

### 🎯 Key Takeaway

This benchmark helps **choose the best hardware** for matrix-heavy workloads (AI, simulations, etc.)
