# TurboCat Installation Guide

## Quick Install

The simplest way to install TurboCat:

```bash
pip install turbocat
```

Or from source:

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
pip install .
```

---

## Requirements

### Minimum

| Component | Version |
|-----------|---------|
| Python | 3.8+ |
| C++ Compiler | GCC 10+ / Clang 12+ / MSVC 2019+ |
| CMake | 3.18+ |

### Recommended (for best performance)

| Component | Purpose |
|-----------|---------|
| OpenMP | Parallel training |
| AVX2/AVX-512 | SIMD acceleration (x86) |
| ARM NEON | SIMD acceleration (Apple Silicon) |

---

## Platform-Specific Instructions

### macOS

```bash
# Install OpenMP for parallel training
brew install libomp

# Install TurboCat
pip install turbocat
```

### Ubuntu / Debian

```bash
# Install dependencies
sudo apt update
sudo apt install -y build-essential cmake python3-dev python3-pip libomp-dev

# Install TurboCat
pip install turbocat
```

### Windows

1. Install Visual Studio 2019+ with C++ workload
2. Install CMake from https://cmake.org/download/
3. Run: `pip install turbocat`

---

## Building from Source

### Step 1: Clone

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
```

### Step 2: Install (pip method - recommended)

```bash
pip install .
```

### Step 3: Verify

```python
import turbocat
print(turbocat.__version__)  # Should print 0.3.0
```

---

## Manual CMake Build

For development or custom builds:

```bash
mkdir build && cd build

# Configure
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTURBOCAT_BUILD_PYTHON=ON \
    -DTURBOCAT_BUILD_TESTS=ON

# Build
cmake --build . --parallel

# Run tests
./turbocat_tests
```

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `TURBOCAT_BUILD_PYTHON` | ON | Build Python bindings |
| `TURBOCAT_BUILD_TESTS` | ON | Build unit tests |
| `TURBOCAT_BUILD_BENCHMARKS` | ON | Build benchmarks |
| `TURBOCAT_USE_OPENMP` | ON | Enable OpenMP |
| `CMAKE_BUILD_TYPE` | Release | Debug / Release |

---

## Verifying Installation

### Python Test

```python
from turbocat import TurboCatClassifier
import numpy as np

# Quick test
X = np.random.randn(100, 10).astype(np.float32)
y = (X[:, 0] > 0).astype(np.float32)

clf = TurboCatClassifier(n_estimators=10, verbosity=0)
clf.fit(X, y)

print("TurboCat installed successfully!")
print(f"Trained {clf.n_trees} trees")
```

### Check SIMD Support

```bash
# Check CPU features (Linux)
grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u

# Check CPU features (macOS)
sysctl -a | grep cpu.features
```

---

## Troubleshooting

### "OpenMP not found"

**macOS:**
```bash
brew install libomp
export OpenMP_ROOT=$(brew --prefix libomp)
pip install . --no-cache-dir
```

**Ubuntu:**
```bash
sudo apt install libomp-dev
```

### "ModuleNotFoundError: No module named 'turbocat'"

```bash
# Ensure you're using the correct Python
python -m pip install turbocat

# Or add build directory to path
export PYTHONPATH=$PYTHONPATH:/path/to/turbocat/build
```

### Slow Performance

1. Ensure Release build:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. Check SIMD is enabled (look for in CMake output):
   ```
   TurboCat: ARM NEON support enabled
   # or
   TurboCat: AVX2 support enabled
   ```

3. Enable OpenMP:
   ```bash
   export OMP_NUM_THREADS=$(nproc)
   ```

### Build Errors on Apple Silicon

```bash
# Use Homebrew's libomp
export LDFLAGS="-L$(brew --prefix libomp)/lib"
export CPPFLAGS="-I$(brew --prefix libomp)/include"
pip install . --no-cache-dir
```

---

## Development Setup

For contributors:

```bash
# Clone with submodules
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dev dependencies
pip install -e ".[dev]"

# Build in debug mode
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DTURBOCAT_BUILD_TESTS=ON
make -j8

# Run tests
./turbocat_tests
ctest --output-on-failure
```

---

## Uninstalling

```bash
pip uninstall turbocat
```

---

## Support

- **Issues**: https://github.com/ispromadhka/Turbo-Cat/issues
- **Documentation**: https://github.com/ispromadhka/Turbo-Cat#readme
