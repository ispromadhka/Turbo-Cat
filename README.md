# 🐱⚡ TurboCat

**Next-Generation Gradient Boosting Framework**

TurboCat is a high-performance gradient boosting library that achieves **superior model quality** compared to CatBoost by implementing cutting-edge research algorithms.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🚀 Key Innovations

- **GradTree** — Gradient-based global tree optimization (AAAI 2024), not greedy splitting
- **Robust Focal Loss** — Better handling of class imbalance and label noise
- **Tsallis Entropy Splitting** — Non-extensive entropy for improved splits
- **LDAM Loss** — Label-distribution-aware margin loss
- **GOSS Sampling** — Gradient-based One-Side Sampling for efficiency
- **SIMD Optimizations** — AVX2/AVX-512 vectorized histogram building

## 📊 Benchmark Results

Comparison on synthetic dataset (10K samples, 20 features, 20 trees, depth=8):

| Metric | TurboCat | CatBoost | Winner |
|--------|----------|----------|--------|
| **Accuracy** | 0.9255 | 0.9005 | 🏆 TurboCat (+2.5%) |
| **Precision** | 0.9215 | 0.9006 | 🏆 TurboCat (+2.1%) |
| **Recall** | 0.9244 | 0.8922 | 🏆 TurboCat (+3.2%) |
| **F1-Score** | 0.9229 | 0.8964 | 🏆 TurboCat (+2.6%) |
| **ROC-AUC** | 0.9821 | 0.9713 | 🏆 TurboCat (+1.1%) |
| **PR-AUC** | 0.9817 | 0.9707 | 🏆 TurboCat (+1.1%) |
| Log Loss | 0.5996 | 0.4908 | CatBoost |

**TurboCat wins 6/7 quality metrics!**

> ⚠️ TurboCat prioritizes model quality over training speed. Current version is ~25x slower than CatBoost due to GradTree's gradient-based optimization. Speed improvements are planned.

## 📦 Installation

### From source (recommended for now)

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Install Python package
cd ../python
pip install -e .
```

### Requirements

- C++20 compiler (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+
- Python 3.8+
- NumPy

### Optional dependencies

- OpenMP (for parallel training)
- Eigen3 (auto-downloaded if not found)

## 🔥 Quick Start

```python
from turbocat import TurboCatClassifier
import numpy as np

# Create classifier
clf = TurboCatClassifier(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=6,
    loss='focal',        # Robust Focal Loss
    use_goss=True,       # GOSS sampling
    verbosity=0
)

# Train
clf.fit(X_train, y_train)

# Predict
proba = clf.predict_proba(X_test)
predictions = clf.predict(X_test)

# Check number of trees
print(f"Trees trained: {clf.n_trees}")
```

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of boosting iterations |
| `learning_rate` | 0.05 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `max_bins` | 255 | Histogram bins |
| `subsample` | 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Feature sampling ratio |
| `min_child_weight` | 1.0 | Minimum leaf hessian sum |
| `lambda_l2` | 1.0 | L2 regularization |
| `loss` | 'logloss' | Loss function: `logloss`, `focal`, `ldam`, `tsallis` |
| `use_goss` | True | Enable GOSS sampling |
| `use_gradtree` | False | Enable GradTree optimization |
| `early_stopping_rounds` | 50 | Early stopping patience |
| `n_threads` | -1 | Thread count (-1 = all) |
| `seed` | 42 | Random seed |

## 🏗️ Project Structure

```
turbocat/
├── include/turbocat/     # C++ headers
├── src/
│   ├── core/             # Types, config, dataset
│   ├── tree/             # Tree building, GradTree, splitters
│   ├── boosting/         # Loss functions, booster
│   └── utils/            # SIMD, threading utilities
├── python/               # Python bindings (pybind11)
├── tests/                # Unit tests
└── benchmarks/           # Performance benchmarks
```

## 🔬 Research References

- **GradTree**: Marton et al., "GradTree: Learning Axis-Aligned Decision Trees with Gradient Descent", AAAI 2024
- **Focal Loss**: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
- **LDAM**: Cao et al., "Learning Imbalanced Datasets with Label-Distribution-Aware Margin Loss", NeurIPS 2019
- **Tsallis Entropy**: Maszczyk & Duch, "Comparison of Shannon, Renyi and Tsallis Entropy", 2008
- **GOSS**: Ke et al., "LightGBM: A Highly Efficient Gradient Boosting Decision Tree", NeurIPS 2017

## 🗺️ Roadmap

- [ ] GPU support (CUDA)
- [ ] Apple Metal support
- [ ] Probability calibration (Platt scaling)
- [ ] Categorical feature encoding
- [ ] Feature importance
- [ ] SHAP integration
- [ ] Pre-built wheels for pip install
- [ ] Oblivious trees for faster inference

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

Contributions welcome! Please open an issue or PR.

---

*Built with ❤️ for the ML community*
