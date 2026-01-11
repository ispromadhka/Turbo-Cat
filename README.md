# TurboCat

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
# English

**Next-generation gradient boosting with up to 64x faster inference than CatBoost and comparable quality.**

TurboCat is a high-performance C++ gradient boosting library with Python bindings, implementing cutting-edge research techniques: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, SIMD-optimized gradients, and symmetric trees with no-binning inference.

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/ispromadhka/Turbo-Cat)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class classification** - Now supported!
- [x] **Parallel training** - OpenMP enabled
- [x] **SIMD optimization** - AVX2/AVX-512 and ARM NEON support
- [x] **Symmetric trees** - Default mode for fast inference
- [x] **Tree mode selection** - `mode` parameter: "small", "large", "auto"
- [x] **Ultra-fast inference** - Up to 64x faster than CatBoost
- [ ] **GPU support** - CUDA and Metal acceleration
- [ ] **Model serialization** - Save/load trained models
- [ ] **ONNX export** - For production deployment

---

## Benchmark Results (v0.4.0)

Comprehensive benchmark comparing TurboCat vs CatBoost.

**Configuration**: `n_estimators=100, max_depth=6, learning_rate=0.1`

### Classification Results

| Dataset | Samples | TC AUC | CB AUC | AUC Diff | Batch Inf | Single Inf | Winner |
|---------|---------|--------|--------|----------|-----------|------------|--------|
| Synthetic 50K | 50,000 | 0.9679 | **0.9694** | -0.15% | **2.1x** | **34x** | CB |
| Synthetic 100K | 100,000 | **0.9704** | 0.9696 | +0.08% | **1.4x** | **35x** | TC |
| Synthetic 200K | 200,000 | **0.9731** | 0.9730 | +0.01% | **1.1x** | **35x** | TC |
| Breast Cancer | 569 | 0.9957 | **0.9961** | -0.04% | **7x** | **30x** | CB |

### Regression Results

| Dataset | Samples | TC R2 | CB R2 | R2 Diff | Batch Inf | Single Inf | Winner |
|---------|---------|-------|-------|---------|-----------|------------|--------|
| Synthetic 50K | 50,000 | **0.9640** | 0.9621 | +0.19% | **1.9x** | **63x** | TC |
| Synthetic 100K | 100,000 | **0.9428** | 0.9412 | +0.16% | 1.0x | **64x** | TC |
| California Housing | 20,640 | 0.7979 | 0.7978 | +0.01% | **2.4x** | **54x** | Tie |

### Speed Summary

| Metric | TurboCat vs CatBoost |
|--------|---------------------|
| Single Inference | **30-64x faster** |
| Batch Inference | **1.1-7x faster** |
| Training | 2-4x slower |

### Final Score

| Category | TurboCat | CatBoost | Winner |
|----------|----------|----------|--------|
| Classification Quality | 2 | 2 | Tie |
| Regression Quality | 2 | 0 | TurboCat |
| Inference Speed | 7 | 0 | TurboCat |
| Training Speed | 0 | 7 | CatBoost |
| **OVERALL** | **11** | **9** | **TurboCat** |

---

## Strengths

### 1. Inference Speed - Key Advantage

TurboCat v0.4.0 delivers exceptional inference performance:
- **Single-sample inference**: 30-64x faster than CatBoost
- **Batch inference**: 1.1-7x faster than CatBoost
- Perfect for real-time prediction and streaming data

### 2. Regression Quality

On regression tasks, TurboCat now **beats CatBoost**:

| Dataset | TC R2 | CB R2 | Difference |
|---------|-------|-------|------------|
| Synthetic 50K | **0.9640** | 0.9621 | **+0.19%** |
| Synthetic 100K | **0.9428** | 0.9412 | **+0.16%** |

### 3. Classification Quality

Comparable ROC-AUC on classification tasks:
- Wins on 2 out of 4 datasets
- Differences within ±0.15%

### 4. Advanced Features

- **SIMD Optimizations**: AVX2/AVX-512 (x86) and NEON (ARM/Apple Silicon)
- **Symmetric Trees**: O(1) leaf lookup with float thresholds
- **No-binning Inference**: Direct float comparisons for speed
- **Focal Loss**: For imbalanced classification

---

## Weaknesses

### 1. Training Speed

Training is slower than CatBoost due to symmetric tree construction:
- Classification: 2-3x slower
- Regression: 2-4x slower

For training-heavy workflows, consider CatBoost.

---

## Tree Modes

TurboCat v0.4.0 uses symmetric trees by default:

| Mode | Tree Type | Best For |
|------|-----------|----------|
| `"small"` | Regular trees | Faster training |
| `"large"` | Symmetric trees | **Fastest inference** |
| `"auto"` | Symmetric (default) | Best balance |

```python
# Default - symmetric trees for fast inference
clf = TurboCatClassifier()

# For faster training (slower inference)
clf = TurboCatClassifier(mode="small")
```

---

## When to Use

### Recommended:

- **Real-time prediction** - 30-64x faster single inference
- **Streaming data** - Low latency predictions
- **Regression tasks** - Better R2 than CatBoost
- **Imbalanced classification** - Focal loss support

### Consider Alternatives:

- Training-heavy workflows (CatBoost is faster to train)
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

# Create classifier (symmetric trees by default)
clf = TurboCatClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6
)

# Train
clf.fit(X_train, y_train)

# Predict probabilities (fast!)
proba = clf.predict_proba(X_test)

# Predict classes
predictions = clf.predict(X_test)
```

### Regression

```python
from turbocat import TurboCatRegressor

reg = TurboCatRegressor(n_estimators=500, learning_rate=0.1, max_depth=6)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Fast inference!
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
| `n_estimators` | 500 | Number of boosting iterations |
| `learning_rate` | 0.1 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `max_bins` | 255 | Histogram bins |
| `subsample` | 1.0 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Feature sampling ratio |
| `min_child_weight` | 1.0 | Minimum leaf hessian sum |
| `lambda_l2` | 1.0 | L2 regularization |
| `mode` | "auto" | Tree type: "small", "large", "auto" |
| `n_threads` | -1 | Number of CPU cores (-1 = all) |
| `verbosity` | 1 | Verbosity level (0=silent) |

---

## Detailed Benchmark (v0.4.0)

```
Dataset              TC AUC/R2  CB AUC/R2  Diff      Single Inf  Winner
─────────────────────────────────────────────────────────────────────────
Classification:
Synthetic 50K        0.9679     0.9694     -0.15%    34x faster  CatBoost
Synthetic 100K       0.9704     0.9696     +0.08%    35x faster  TurboCat
Synthetic 200K       0.9731     0.9730     +0.01%    35x faster  TurboCat
Breast Cancer        0.9957     0.9961     -0.04%    30x faster  CatBoost

Regression:
Synthetic 50K        0.9640     0.9621     +0.19%    63x faster  TurboCat
Synthetic 100K       0.9428     0.9412     +0.16%    64x faster  TurboCat
California Housing   0.7979     0.7978     +0.01%    54x faster  Tie
─────────────────────────────────────────────────────────────────────────
INFERENCE: 30-64x faster    QUALITY: Comparable    TurboCat WINS on Speed
```

---

## What's New in v0.4.0

- **Single inference**: 30-64x faster than CatBoost
- **Batch inference**: 1.1-7x faster (was 4.7x slower!)
- **Regression quality**: Now beats CatBoost
- **Symmetric trees by default**: No-binning prediction path
- **Improved defaults**: Better out-of-the-box performance

### v0.3.0

- 17x faster training on small datasets
- SIMD-optimized gradients
- ARM NEON support for Apple Silicon
- Tree mode selection
- 27 vs 9 quality wins against CatBoost

---

<a name="russian"></a>
# Русский

**Градиентный бустинг нового поколения — инференс до 64x быстрее CatBoost при сопоставимом качестве.**

TurboCat — высокопроизводительная библиотека градиентного бустинга на C++ с Python-привязками, реализующая современные техники: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy, SIMD-оптимизированные градиенты и симметричные деревья с no-binning инференсом.

[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](https://github.com/ispromadhka/Turbo-Cat)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class классификация** - Поддерживается!
- [x] **Параллельное обучение** - OpenMP включен
- [x] **SIMD оптимизация** - Поддержка AVX2/AVX-512 и ARM NEON
- [x] **Симметричные деревья** - Режим по умолчанию для быстрого инференса
- [x] **Выбор типа деревьев** - Параметр `mode`: "small", "large", "auto"
- [x] **Ультра-быстрый инференс** - До 64x быстрее CatBoost
- [ ] **GPU поддержка** - Ускорение на CUDA и Metal
- [ ] **Сериализация моделей** - Сохранение/загрузка
- [ ] **ONNX экспорт** - Для production deployment

---

## Результаты бенчмарков (v0.4.0)

Сравнение TurboCat vs CatBoost.

**Конфигурация**: `n_estimators=100, max_depth=6, learning_rate=0.1`

### Результаты классификации

| Датасет | Samples | TC AUC | CB AUC | Разница | Batch Inf | Single Inf | Победитель |
|---------|---------|--------|--------|---------|-----------|------------|------------|
| Synthetic 50K | 50,000 | 0.9679 | **0.9694** | -0.15% | **2.1x** | **34x** | CB |
| Synthetic 100K | 100,000 | **0.9704** | 0.9696 | +0.08% | **1.4x** | **35x** | TC |
| Synthetic 200K | 200,000 | **0.9731** | 0.9730 | +0.01% | **1.1x** | **35x** | TC |
| Breast Cancer | 569 | 0.9957 | **0.9961** | -0.04% | **7x** | **30x** | CB |

### Результаты регрессии

| Датасет | Samples | TC R2 | CB R2 | Разница | Batch Inf | Single Inf | Победитель |
|---------|---------|-------|-------|---------|-----------|------------|------------|
| Synthetic 50K | 50,000 | **0.9640** | 0.9621 | +0.19% | **1.9x** | **63x** | TC |
| Synthetic 100K | 100,000 | **0.9428** | 0.9412 | +0.16% | 1.0x | **64x** | TC |
| California Housing | 20,640 | 0.7979 | 0.7978 | +0.01% | **2.4x** | **54x** | Ничья |

### Сводка по скорости

| Метрика | TurboCat vs CatBoost |
|---------|---------------------|
| Single инференс | **30-64x быстрее** |
| Batch инференс | **1.1-7x быстрее** |
| Обучение | 2-4x медленнее |

### Финальный счёт

| Категория | TurboCat | CatBoost | Победитель |
|-----------|----------|----------|------------|
| Качество классификации | 2 | 2 | Ничья |
| Качество регрессии | 2 | 0 | TurboCat |
| Скорость инференса | 7 | 0 | TurboCat |
| Скорость обучения | 0 | 7 | CatBoost |
| **ИТОГО** | **11** | **9** | **TurboCat** |

---

## Сильные стороны

### 1. Скорость инференса — главное преимущество

TurboCat v0.4.0 обеспечивает исключительную скорость предсказаний:
- **Single-sample инференс**: 30-64x быстрее CatBoost
- **Batch инференс**: 1.1-7x быстрее CatBoost
- Идеально для real-time предсказаний и потоковых данных

### 2. Качество регрессии

На задачах регрессии TurboCat **превосходит CatBoost**:

| Датасет | TC R2 | CB R2 | Разница |
|---------|-------|-------|---------|
| Synthetic 50K | **0.9640** | 0.9621 | **+0.19%** |
| Synthetic 100K | **0.9428** | 0.9412 | **+0.16%** |

### 3. Качество классификации

Сопоставимый ROC-AUC на задачах классификации:
- Побеждает на 2 из 4 датасетов
- Различия в пределах ±0.15%

### 4. Продвинутые возможности

- **SIMD оптимизация**: AVX2/AVX-512 (x86) и NEON (ARM/Apple Silicon)
- **Симметричные деревья**: O(1) поиск листа с float порогами
- **No-binning инференс**: Прямые float сравнения
- **Focal Loss**: Для несбалансированной классификации

---

## Слабые стороны

### 1. Скорость обучения

Обучение медленнее CatBoost из-за построения симметричных деревьев:
- Классификация: 2-3x медленнее
- Регрессия: 2-4x медленнее

Для задач с частым переобучением рассмотрите CatBoost.

---

## Режимы деревьев

TurboCat v0.4.0 использует симметричные деревья по умолчанию:

| Режим | Тип деревьев | Лучше для |
|-------|--------------|-----------|
| `"small"` | Обычные деревья | Быстрое обучение |
| `"large"` | Симметричные | **Быстрейший инференс** |
| `"auto"` | Симметричные (default) | Лучший баланс |

```python
# По умолчанию — симметричные деревья для быстрого инференса
clf = TurboCatClassifier()

# Для быстрого обучения (медленнее инференс)
clf = TurboCatClassifier(mode="small")
```

---

## Когда использовать

### Рекомендуется:

- **Real-time предсказания** — 30-64x быстрее single inference
- **Потоковые данные** — Низкая задержка
- **Задачи регрессии** — Лучший R2 чем CatBoost
- **Несбалансированная классификация** — Поддержка Focal Loss

### Рассмотреть альтернативы:

- Частое переобучение (CatBoost быстрее обучается)
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

# Создание классификатора (симметричные деревья по умолчанию)
clf = TurboCatClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=6
)

# Обучение
clf.fit(X_train, y_train)

# Предсказание вероятностей (быстро!)
proba = clf.predict_proba(X_test)

# Предсказание классов
predictions = clf.predict(X_test)
```

### Регрессия

```python
from turbocat import TurboCatRegressor

reg = TurboCatRegressor(n_estimators=500, learning_rate=0.1, max_depth=6)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)  # Быстрый инференс!
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
| `n_estimators` | 500 | Количество деревьев |
| `learning_rate` | 0.1 | Скорость обучения |
| `max_depth` | 6 | Максимальная глубина дерева |
| `max_bins` | 255 | Количество бинов гистограммы |
| `subsample` | 1.0 | Доля сэмплов для обучения |
| `colsample_bytree` | 0.8 | Доля признаков для дерева |
| `min_child_weight` | 1.0 | Минимальный вес листа |
| `lambda_l2` | 1.0 | L2 регуляризация |
| `mode` | "auto" | Тип деревьев: "small", "large", "auto" |
| `n_threads` | -1 | Количество ядер CPU (-1 = все) |
| `verbosity` | 1 | Уровень вывода (0=тихий) |

---

## Детальный бенчмарк (v0.4.0)

```
Датасет              TC AUC/R2  CB AUC/R2  Разница   Single Inf  Победитель
─────────────────────────────────────────────────────────────────────────────
Классификация:
Synthetic 50K        0.9679     0.9694     -0.15%    34x быстрее CatBoost
Synthetic 100K       0.9704     0.9696     +0.08%    35x быстрее TurboCat
Synthetic 200K       0.9731     0.9730     +0.01%    35x быстрее TurboCat
Breast Cancer        0.9957     0.9961     -0.04%    30x быстрее CatBoost

Регрессия:
Synthetic 50K        0.9640     0.9621     +0.19%    63x быстрее TurboCat
Synthetic 100K       0.9428     0.9412     +0.16%    64x быстрее TurboCat
California Housing   0.7979     0.7978     +0.01%    54x быстрее Ничья
─────────────────────────────────────────────────────────────────────────────
ИНФЕРЕНС: 30-64x быстрее    КАЧЕСТВО: Сопоставимо    TurboCat WINS по скорости
```

---

## Что нового в v0.4.0

- **Single инференс**: 30-64x быстрее CatBoost
- **Batch инференс**: 1.1-7x быстрее (было 4.7x медленнее!)
- **Качество регрессии**: Теперь превосходит CatBoost
- **Симметричные деревья по умолчанию**: No-binning предсказания
- **Улучшенные defaults**: Лучшая производительность из коробки

### v0.3.0

- Обучение в 17x быстрее на малых датасетах
- SIMD-оптимизированные градиенты
- Поддержка ARM NEON для Apple Silicon
- Выбор режима деревьев
- 27 vs 9 побед по качеству против CatBoost

---

## License

MIT License
