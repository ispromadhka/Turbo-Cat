# TurboCat

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
# English

**Next-generation gradient boosting that beats CatBoost on quality (27 vs 9 wins) with up to 17x faster training.**

TurboCat is a high-performance C++ gradient boosting library with Python bindings, implementing cutting-edge research techniques: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, SIMD-optimized gradients, and symmetric trees.

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/ispromadhka/Turbo-Cat)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class classification** - Now supported!
- [x] **Parallel training** - OpenMP enabled
- [x] **SIMD optimization** - AVX2/AVX-512 and ARM NEON support
- [x] **Symmetric trees** - `mode="large"` for fast inference
- [x] **Tree mode selection** - `mode` parameter: "small", "large", "auto"
- [ ] **GPU support** - CUDA and Metal acceleration
- [ ] **Model serialization** - Save/load trained models
- [ ] **ONNX export** - For production deployment

---

## Benchmark Results (v0.3.0)

Comprehensive benchmark comparing TurboCat vs CatBoost on 6 datasets.

**Configuration**: `n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8`

### Classification Results

| Dataset | Samples | Features | TC AUC | CB AUC | AUC Diff | Train Speed | Winner |
|---------|---------|----------|--------|--------|----------|-------------|--------|
| Breast Cancer | 569 | 30 | **0.9950** | 0.9940 | +0.10% | **16.7x** | TC |
| Small | 5,000 | 20 | **0.9875** | 0.9873 | +0.02% | **3.0x** | TC |
| Medium | 20,000 | 30 | **0.9932** | 0.9919 | +0.13% | **1.3x** | TC |
| Large | 50,000 | 40 | **0.9934** | 0.9933 | +0.02% | 0.86x | TC |
| Imbalanced | 30,000 | 25 | **0.9725** | 0.9697 | +0.28% | **1.2x** | TC |
| High-Dim | 10,000 | 100 | 0.9902 | **0.9910** | -0.08% | **1.8x** | CB |

### All Quality Metrics Summary

| Metric | TurboCat Wins | CatBoost Wins |
|--------|---------------|---------------|
| ROC-AUC | 5 | 1 |
| Accuracy | 4 | 2 |
| Precision | 4 | 2 |
| Recall | 4 | 2 |
| F1-Score | 4 | 2 |
| LogLoss | 6 | 0 |
| **Total Quality** | **27** | **9** |

### Speed Summary

| Metric | TurboCat Wins | CatBoost Wins |
|--------|---------------|---------------|
| Training Speed | 5 | 1 |
| Inference Speed | 1 | 5 |
| **Total Speed** | **6** | **6** |

### Final Score

| Category | TurboCat | CatBoost | Winner |
|----------|----------|----------|--------|
| Quality Metrics | **27** | 9 | TurboCat |
| Speed Metrics | 6 | 6 | Tie |
| **OVERALL** | **33** | **15** | **TurboCat** |

---

## Strengths

### 1. Classification Quality - Key Advantage

TurboCat significantly outperforms CatBoost on classification:
- **AUC**: Wins on 5 out of 6 datasets
- **LogLoss**: Wins on ALL 6 datasets (up to +19% better)
- **All Metrics**: 27 vs 9 quality wins

### 2. Training Speed

- **Small datasets** (< 1K): up to **17x** faster
- **Medium datasets** (1K-20K): **1.3-3x** faster
- **Large datasets** (50K+): comparable

### 3. Imbalanced Data

On imbalanced datasets, TurboCat excels:

| Dataset | TC AUC | CB AUC | Difference |
|---------|--------|--------|------------|
| Imbalanced 30K | **97.25%** | 96.97% | **+0.28%** |
| Recall (minority) | **85.97%** | 83.39% | **+2.58%** |

### 4. Advanced Features

- **SIMD Optimizations**: AVX2/AVX-512 (x86) and NEON (ARM/Apple Silicon)
- **Symmetric Trees**: O(1) leaf lookup for fast inference
- **Tree Mode Selection**: Choose between quality and speed

---

## Weaknesses

### 1. Inference Speed (with regular trees)

When using `mode="small"` (regular trees):
- Inference is slower than CatBoost
- Use `mode="large"` for faster inference at slight quality trade-off

### 2. Symmetric Tree Training

When using `mode="large"` (symmetric trees):
- Training is slower than CatBoost's optimized symmetric implementation
- Inference becomes competitive

---

## Tree Modes

TurboCat v0.3.0 introduces the `mode` parameter:

| Mode | Tree Type | Best For |
|------|-----------|----------|
| `"small"` | Regular trees | **Best quality**, faster training |
| `"large"` | Symmetric trees | Faster inference, comparable quality |
| `"auto"` | Auto-select | Chooses based on data size |

```python
# For best quality (default)
clf = TurboCatClassifier(mode="small")

# For faster inference
clf = TurboCatClassifier(mode="large")

# Auto-select based on data size
clf = TurboCatClassifier(mode="auto")
```

---

## When to Use

### Recommended:

- **Binary/Multi-class classification** - TurboCat's strength
- **Fraud detection, medical diagnosis** - imbalanced classes
- **Real-time training** - up to 17x faster
- **Quality-critical applications** - better AUC and LogLoss

### Consider Alternatives:

- Very large inference batches with `mode="small"` (use `mode="large"` or CatBoost)
- Native categorical feature support (CatBoost)
- GPU training (XGBoost, LightGBM)

---

## Installation

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
pip install .
```

That's it! No manual CMake configuration needed.

### Requirements

- Python 3.8+
- C++20 compiler (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+ (installed automatically if missing)

### Optional (for best performance)

```bash
# macOS (for OpenMP support)
brew install libomp

# Ubuntu/Debian
sudo apt install libomp-dev
```

---

## Quick Start

```python
from turbocat import TurboCatClassifier
import numpy as np

# Create classifier
clf = TurboCatClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    mode="small"  # Best quality
)

# Train
clf.fit(X_train, y_train)

# Predict probabilities
proba = clf.predict_proba(X_test)

# Predict classes
predictions = clf.predict(X_test)
```

### Regression

```python
from turbocat import TurboCatRegressor

reg = TurboCatRegressor(n_estimators=500, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### Multi-class Classification

```python
from turbocat import TurboCatClassifier

# Multi-class is automatically detected
clf = TurboCatClassifier(n_estimators=500)
clf.fit(X_train, y_train)  # y can have >2 classes
proba = clf.predict_proba(X_test)  # Returns (n_samples, n_classes)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 1000 | Number of boosting iterations |
| `learning_rate` | 0.1 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `max_bins` | 255 | Histogram bins |
| `subsample` | 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Feature sampling ratio |
| `min_child_weight` | 1.0 | Minimum leaf hessian sum |
| `lambda_l2` | 1.0 | L2 regularization |
| `mode` | "auto" | Tree type: "small", "large", "auto" |
| `use_goss` | False | Use Gradient-based One-Side Sampling |
| `n_threads` | -1 | Number of CPU cores (-1 = all) |
| `verbosity` | 1 | Verbosity level (0=silent) |

---

## Detailed Benchmark (v0.3.0)

```
Dataset              TC AUC    CB AUC    Diff      Train     Winner
────────────────────────────────────────────────────────────────────
Breast Cancer        0.9950    0.9940    +0.10%    16.7x     TurboCat
Small (5K)           0.9875    0.9873    +0.02%     3.0x     TurboCat
Medium (20K)         0.9932    0.9919    +0.13%     1.3x     TurboCat
Large (50K)          0.9934    0.9933    +0.02%     0.86x    TurboCat
Imbalanced (30K)     0.9725    0.9697    +0.28%     1.2x     TurboCat
High-Dim (100f)      0.9902    0.9910    -0.08%     1.8x     CatBoost
────────────────────────────────────────────────────────────────────
TOTAL                Quality: 27 vs 9    Speed: 6 vs 6    TurboCat WINS
```

---

## What's New in v0.3.0

- **17x faster training** on small datasets
- **SIMD-optimized gradients** with accurate sigmoid approximation
- **ARM NEON support** for Apple Silicon
- **Tree mode selection** (`mode` parameter)
- **27 vs 9 quality wins** against CatBoost
- **Improved symmetric trees** with tree batching and prefetching

---

<a name="russian"></a>
# Русский

**Градиентный бустинг нового поколения — превосходит CatBoost по качеству (27 vs 9 побед), обучение до 17x быстрее.**

TurboCat — высокопроизводительная библиотека градиентного бустинга на C++ с Python-привязками, реализующая современные исследовательские техники: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy, SIMD-оптимизированные градиенты и симметричные деревья.

[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](https://github.com/ispromadhka/Turbo-Cat)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class классификация** - Поддерживается!
- [x] **Параллельное обучение** - OpenMP включен
- [x] **SIMD оптимизация** - Поддержка AVX2/AVX-512 и ARM NEON
- [x] **Симметричные деревья** - `mode="large"` для быстрого инференса
- [x] **Выбор типа деревьев** - Параметр `mode`: "small", "large", "auto"
- [ ] **GPU поддержка** - Ускорение на CUDA и Metal
- [ ] **Сериализация моделей** - Сохранение/загрузка
- [ ] **ONNX экспорт** - Для production deployment

---

## Результаты бенчмарков (v0.3.0)

Комплексное сравнение TurboCat vs CatBoost на 6 датасетах.

**Конфигурация**: `n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8`

### Результаты классификации

| Датасет | Samples | Features | TC AUC | CB AUC | Разница | Скорость | Победитель |
|---------|---------|----------|--------|--------|---------|----------|------------|
| Breast Cancer | 569 | 30 | **0.9950** | 0.9940 | +0.10% | **16.7x** | TC |
| Small | 5,000 | 20 | **0.9875** | 0.9873 | +0.02% | **3.0x** | TC |
| Medium | 20,000 | 30 | **0.9932** | 0.9919 | +0.13% | **1.3x** | TC |
| Large | 50,000 | 40 | **0.9934** | 0.9933 | +0.02% | 0.86x | TC |
| Imbalanced | 30,000 | 25 | **0.9725** | 0.9697 | +0.28% | **1.2x** | TC |
| High-Dim | 10,000 | 100 | 0.9902 | **0.9910** | -0.08% | **1.8x** | CB |

### Сводка по метрикам качества

| Метрика | Побед TurboCat | Побед CatBoost |
|---------|----------------|----------------|
| ROC-AUC | 5 | 1 |
| Accuracy | 4 | 2 |
| Precision | 4 | 2 |
| Recall | 4 | 2 |
| F1-Score | 4 | 2 |
| LogLoss | 6 | 0 |
| **Всего качество** | **27** | **9** |

### Финальный счёт

| Категория | TurboCat | CatBoost | Победитель |
|-----------|----------|----------|------------|
| Метрики качества | **27** | 9 | TurboCat |
| Метрики скорости | 6 | 6 | Ничья |
| **ИТОГО** | **33** | **15** | **TurboCat** |

---

## Сильные стороны

### 1. Качество классификации — главное преимущество

TurboCat значительно превосходит CatBoost:
- **AUC**: Побеждает на 5 из 6 датасетов
- **LogLoss**: Побеждает на ВСЕХ 6 датасетах (до +19% лучше)
- **Все метрики**: 27 vs 9 побед по качеству

### 2. Скорость обучения

- **Малые датасеты** (< 1K): до **17x** быстрее
- **Средние датасеты** (1K-20K): **1.3-3x** быстрее
- **Большие датасеты** (50K+): сопоставимо

### 3. Несбалансированные данные

На несбалансированных данных TurboCat отлично работает:

| Датасет | TC AUC | CB AUC | Разница |
|---------|--------|--------|---------|
| Imbalanced 30K | **97.25%** | 96.97% | **+0.28%** |
| Recall (меньшинство) | **85.97%** | 83.39% | **+2.58%** |

### 4. Продвинутые возможности

- **SIMD оптимизация**: AVX2/AVX-512 (x86) и NEON (ARM/Apple Silicon)
- **Симметричные деревья**: O(1) поиск листа для быстрого инференса
- **Выбор режима**: Баланс между качеством и скоростью

---

## Режимы деревьев

TurboCat v0.3.0 вводит параметр `mode`:

| Режим | Тип деревьев | Лучше для |
|-------|--------------|-----------|
| `"small"` | Обычные деревья | **Лучшее качество**, быстрое обучение |
| `"large"` | Симметричные деревья | Быстрый инференс, сравнимое качество |
| `"auto"` | Авто-выбор | Выбирает по размеру данных |

```python
# Для лучшего качества (по умолчанию)
clf = TurboCatClassifier(mode="small")

# Для быстрого инференса
clf = TurboCatClassifier(mode="large")

# Авто-выбор по размеру данных
clf = TurboCatClassifier(mode="auto")
```

---

## Когда использовать

### Рекомендуется:

- **Бинарная/Multi-class классификация** — сила TurboCat
- **Fraud detection, медицинская диагностика** — несбалансированные классы
- **Real-time обучение** — до 17x быстрее
- **Критичные к качеству приложения** — лучше AUC и LogLoss

### Рассмотреть альтернативы:

- Большие батчи инференса с `mode="small"` (используйте `mode="large"` или CatBoost)
- Нативная поддержка категориальных признаков (CatBoost)
- GPU обучение (XGBoost, LightGBM)

---

## Установка

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
pip install .
```

Готово! Никакой ручной настройки CMake не требуется.

### Требования

- Python 3.8+
- C++20 компилятор (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+ (установится автоматически)

### Опционально (для лучшей производительности)

```bash
# macOS (для OpenMP)
brew install libomp

# Ubuntu/Debian
sudo apt install libomp-dev
```

---

## Быстрый старт

```python
from turbocat import TurboCatClassifier
import numpy as np

# Создание классификатора
clf = TurboCatClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    mode="small"  # Лучшее качество
)

# Обучение
clf.fit(X_train, y_train)

# Предсказание вероятностей
proba = clf.predict_proba(X_test)

# Предсказание классов
predictions = clf.predict(X_test)
```

### Регрессия

```python
from turbocat import TurboCatRegressor

reg = TurboCatRegressor(n_estimators=500, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### Multi-class классификация

```python
from turbocat import TurboCatClassifier

# Multi-class определяется автоматически
clf = TurboCatClassifier(n_estimators=500)
clf.fit(X_train, y_train)  # y может иметь >2 классов
proba = clf.predict_proba(X_test)  # Возвращает (n_samples, n_classes)
```

---

## Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `n_estimators` | 1000 | Количество деревьев |
| `learning_rate` | 0.1 | Скорость обучения |
| `max_depth` | 6 | Максимальная глубина дерева |
| `max_bins` | 255 | Количество бинов гистограммы |
| `subsample` | 0.8 | Доля сэмплов для обучения |
| `colsample_bytree` | 0.8 | Доля признаков для дерева |
| `min_child_weight` | 1.0 | Минимальный вес листа |
| `lambda_l2` | 1.0 | L2 регуляризация |
| `mode` | "auto" | Тип деревьев: "small", "large", "auto" |
| `use_goss` | False | Использовать GOSS сэмплирование |
| `n_threads` | -1 | Количество ядер CPU (-1 = все) |
| `verbosity` | 1 | Уровень вывода (0=тихий) |

---

## Детальный бенчмарк (v0.3.0)

```
Датасет              TC AUC    CB AUC    Разница   Скорость  Победитель
────────────────────────────────────────────────────────────────────────
Breast Cancer        0.9950    0.9940    +0.10%    16.7x     TurboCat
Small (5K)           0.9875    0.9873    +0.02%     3.0x     TurboCat
Medium (20K)         0.9932    0.9919    +0.13%     1.3x     TurboCat
Large (50K)          0.9934    0.9933    +0.02%     0.86x    TurboCat
Imbalanced (30K)     0.9725    0.9697    +0.28%     1.2x     TurboCat
High-Dim (100f)      0.9902    0.9910    -0.08%     1.8x     CatBoost
────────────────────────────────────────────────────────────────────────
ИТОГО                Качество: 27 vs 9   Скорость: 6 vs 6   TurboCat WINS
```

---

## Что нового в v0.3.0

- **Обучение в 17x быстрее** на малых датасетах
- **SIMD-оптимизированные градиенты** с точной аппроксимацией сигмоиды
- **Поддержка ARM NEON** для Apple Silicon
- **Выбор режима деревьев** (параметр `mode`)
- **27 vs 9 побед по качеству** против CatBoost
- **Улучшенные симметричные деревья** с пакетной обработкой и prefetching

---

## License

MIT License
