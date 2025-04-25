# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-04-20

Initial stable release of CESS platform.

### Features
- **Automatic hardware detection** (CPU cores, GPU availability)
- **Warm-up runs** to avoid cold-start bias
- **Parallel CPU computation** using `multiprocessing.Pool`
- **GPU-accelerated matrix multiplication** via PyTorch (`torch.mm`)
- **Hybrid CPU+GPU mode** for splitting workloads

Key improvements:

1. **CI/CD Pipeline**:
   - Added GitHub Actions workflows for testing, publishing to PyPI, and building/pushing Docker images
   - Includes proper secret management for PyPI and Docker Hub

2. **Docker Support**:
   - Added both CPU and GPU Dockerfiles
   - Docker build and push automation in CI/CD

3. **Proper Software Structure**:
   - Maintained your original XPU structure
   - Added comprehensive test setup with pytest fixtures
   - Included example usage files

4. **Project Metadata**:
   - Complete pyproject.toml with all necessary metadata
   - README.md with installation and usage instructions
   - Proper .gitignore file

5. **Documentation**:
   - Basic documentation in README
   - Ready for expansion with proper docs

This creates a production-ready Python package that:
- Can be installed via pip
- Has automated testing
- Can be published to PyPI
- Has Docker images for both CPU and GPU
- Follows Python packaging best practices
- Includes issue templates for GitHub

To use this, simply run the script and it will create the complete project structure with all necessary files pre-populated with reasonable defaults.
