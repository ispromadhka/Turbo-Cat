# TurboCat

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
# English

**Next-generation gradient boosting that beats CatBoost quality while being 1.4x faster in training.**

TurboCat is a high-performance C++ gradient boosting library with Python bindings, implementing cutting-edge research techniques: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, and GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class classification** - Now supported!
- [x] **Parallel training** - `n_jobs` parameter added
- [x] **OpenMP support** - Enabled for faster training
- [ ] **Regression optimization** - Improve R² on regression tasks
- [ ] **Inference optimization** - Implement oblivious trees for faster prediction
- [ ] **GPU support** - CUDA and Metal acceleration
- [ ] **Model serialization** - Save/load trained models

---

## Benchmark Results

Tested on classification and regression datasets comparing TurboCat vs CatBoost.

**Hyperparameters**: `n_estimators=100, learning_rate=0.1, max_depth=6`

### Classification Results

| Dataset | Samples | TC Train | CB Train | SpeedUp | TC AUC | CB AUC | Winner |
|---------|---------|----------|----------|---------|--------|--------|--------|
| Breast Cancer | 569 | 0.03s | 0.34s | **12.2x** | 0.9944 | 0.9934 | Tie |
| Synthetic 2K | 2000 | 0.07s | 0.22s | **3.1x** | 0.9787 | 0.9738 | **TurboCat** |
| Synthetic 10K | 10000 | 0.23s | 0.39s | **1.7x** | 0.9898 | 0.9878 | **TurboCat** |
| Synthetic 50K | 50000 | 1.10s | 0.94s | 0.85x | 0.9924 | 0.9863 | **TurboCat** |
| Imbalanced 95/5 | 5000 | 0.11s | 0.27s | **2.5x** | 0.9439 | 0.9006 | **TurboCat** |
| **TOTAL** | | **1.54s** | **2.16s** | **1.41x** | **0.9798** | **0.9684** | **4:0** |

### Regression Results

| Dataset | Samples | TC Train | CB Train | SpeedUp | TC R² | CB R² | Winner |
|---------|---------|----------|----------|---------|-------|-------|--------|
| California Housing | 20640 | 0.30s | 0.25s | 0.82x | 0.8060 | 0.7978 | **TurboCat** |
| Regression 2K | 2000 | 0.09s | 0.20s | **2.1x** | 0.9195 | 0.9677 | CatBoost |
| Regression 10K | 10000 | 0.31s | 0.34s | **1.1x** | 0.9377 | 0.9734 | CatBoost |
| Regression 50K | 50000 | 1.38s | 0.87s | 0.63x | 0.9323 | 0.9373 | CatBoost |
| **TOTAL** | | **2.08s** | **1.66s** | **0.79x** | **0.8989** | **0.9191** | **1:3** |

### Summary

| Task | Training Speed | Quality | Wins |
|------|----------------|---------|------|
| **Classification** | **1.41x faster** | **+1.14% AUC** | **TurboCat 4:0** |
| Regression | 0.79x | -2.02% R² | CatBoost 3:1 |
| **Overall** | **1.05x faster** | | **TurboCat 5:3** |

---

## Strengths

### 1. Classification - Key Advantage

TurboCat significantly outperforms CatBoost on classification:
- **Training**: 1.4x faster overall, up to **12x** on small datasets
- **Quality**: +1.14% better AUC on average
- **Wins 4 out of 5** classification benchmarks

### 2. Imbalanced Data

On imbalanced datasets, TurboCat excels:

| Dataset | TC AUC | CB AUC | Difference |
|---------|--------|--------|------------|
| Imbalanced 95/5 | **94.4%** | 90.1% | **+4.3%** |
| Imbalanced 99/1 | **78.4%** | 69.2% | **+9.2%** |

### 3. Training Speed

- **Small datasets** (< 1K): up to **12x** faster
- **Medium datasets** (1K-10K): **2-3x** faster
- **Large datasets** (50K+): comparable

### 4. Parallel Training

New `n_jobs` parameter for multi-core utilization:

```python
clf = TurboCatClassifier(n_jobs=-1)  # Use all cores
```

---

## Weaknesses

### 1. Regression Performance

TurboCat currently loses to CatBoost on regression tasks:
- R² is 2% lower on average
- Working on improvements

### 2. Inference Speed

CatBoost uses oblivious trees which are faster for prediction:
- TurboCat inference is ~2x slower
- Oblivious trees implementation in progress

---

## When to Use

### Recommended:

- **Binary/Multi-class classification** - TurboCat's strength
- **Fraud detection, medical diagnosis** - imbalanced classes
- **Real-time training** - up to 12x faster
- **Medium datasets** - 1K-50K samples

### Consider Alternatives:

- Regression tasks (CatBoost may be better)
- Very large inference batches (CatBoost faster)

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
- C++17 compiler (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+ (installed automatically if missing)

### Optional

- OpenMP (for parallel training) - `brew install libomp` on macOS

---

## Quick Start

```python
from turbocat import TurboCatClassifier
import numpy as np

# Create classifier with all cores
clf = TurboCatClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1  # Use all CPU cores
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

reg = TurboCatRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### Multi-class Classification

```python
from turbocat import TurboCatClassifier

# Multi-class is automatically detected
clf = TurboCatClassifier(n_estimators=100)
clf.fit(X_train, y_train)  # y can have >2 classes
proba = clf.predict_proba(X_test)  # Returns (n_samples, n_classes)
```

---

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of boosting iterations |
| `learning_rate` | 0.1 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `max_bins` | 255 | Histogram bins |
| `subsample` | 0.8 | Row sampling ratio |
| `colsample_bytree` | 0.8 | Feature sampling ratio |
| `min_child_weight` | 1.0 | Minimum leaf hessian sum |
| `lambda_l2` | 1.0 | L2 regularization |
| `use_goss` | True | Use Gradient-based One-Side Sampling |
| `n_jobs` | -1 | Number of CPU cores (-1 = all) |
| `verbosity` | 1 | Verbosity level (0=silent) |

---

## Detailed Classification Benchmark

```
Dataset                TC Acc    CB Acc    TC AUC    CB AUC    SpeedUp    Winner
─────────────────────────────────────────────────────────────────────────────────
Breast Cancer          96.5%     96.5%     99.4%     99.3%     12.2x      Tie
Synthetic 2K           94.0%     93.0%     97.9%     97.4%     3.1x       TurboCat
Synthetic 10K          96.9%     96.2%     99.0%     98.8%     1.7x       TurboCat
Synthetic 50K          97.2%     94.8%     99.2%     98.6%     0.85x      TurboCat
Imbalanced 95/5        96.7%     96.9%     94.4%     90.1%     2.5x       TurboCat
─────────────────────────────────────────────────────────────────────────────────
AVERAGE                96.3%     95.5%     98.0%     96.8%     1.41x      TurboCat
```

---

<a name="russian"></a>
# Русский

**Градиентный бустинг нового поколения — превосходит CatBoost по качеству, обучение в 1.4 раза быстрее.**

TurboCat — высокопроизводительная библиотека градиентного бустинга на C++ с Python-привязками, реализующая современные исследовательские техники: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [x] **Multi-class классификация** - Теперь поддерживается!
- [x] **Параллельное обучение** - Добавлен параметр `n_jobs`
- [x] **OpenMP поддержка** - Включена для ускорения
- [ ] **Оптимизация регрессии** - Улучшить R² на задачах регрессии
- [ ] **Оптимизация инференса** - Реализовать oblivious trees
- [ ] **GPU поддержка** - Ускорение на CUDA и Metal
- [ ] **Сериализация моделей** - Сохранение/загрузка моделей

---

## Результаты бенчмарков

Тестирование на датасетах классификации и регрессии: TurboCat vs CatBoost.

**Гиперпараметры**: `n_estimators=100, learning_rate=0.1, max_depth=6`

### Результаты классификации

| Датасет | Samples | TC Train | CB Train | Ускорение | TC AUC | CB AUC | Победитель |
|---------|---------|----------|----------|-----------|--------|--------|------------|
| Breast Cancer | 569 | 0.03s | 0.34s | **12.2x** | 0.9944 | 0.9934 | Ничья |
| Synthetic 2K | 2000 | 0.07s | 0.22s | **3.1x** | 0.9787 | 0.9738 | **TurboCat** |
| Synthetic 10K | 10000 | 0.23s | 0.39s | **1.7x** | 0.9898 | 0.9878 | **TurboCat** |
| Synthetic 50K | 50000 | 1.10s | 0.94s | 0.85x | 0.9924 | 0.9863 | **TurboCat** |
| Imbalanced 95/5 | 5000 | 0.11s | 0.27s | **2.5x** | 0.9439 | 0.9006 | **TurboCat** |
| **ИТОГО** | | **1.54s** | **2.16s** | **1.41x** | **0.9798** | **0.9684** | **4:0** |

### Результаты регрессии

| Датасет | Samples | TC Train | CB Train | Ускорение | TC R² | CB R² | Победитель |
|---------|---------|----------|----------|-----------|-------|-------|------------|
| California Housing | 20640 | 0.30s | 0.25s | 0.82x | 0.8060 | 0.7978 | **TurboCat** |
| Regression 2K | 2000 | 0.09s | 0.20s | **2.1x** | 0.9195 | 0.9677 | CatBoost |
| Regression 10K | 10000 | 0.31s | 0.34s | **1.1x** | 0.9377 | 0.9734 | CatBoost |
| Regression 50K | 50000 | 1.38s | 0.87s | 0.63x | 0.9323 | 0.9373 | CatBoost |
| **ИТОГО** | | **2.08s** | **1.66s** | **0.79x** | **0.8989** | **0.9191** | **1:3** |

### Итог

| Задача | Скорость обучения | Качество | Победы |
|--------|-------------------|----------|--------|
| **Классификация** | **в 1.41x быстрее** | **+1.14% AUC** | **TurboCat 4:0** |
| Регрессия | в 0.79x | -2.02% R² | CatBoost 3:1 |
| **Всего** | **в 1.05x быстрее** | | **TurboCat 5:3** |

---

## Сильные стороны

### 1. Классификация — главное преимущество

TurboCat значительно превосходит CatBoost на классификации:
- **Обучение**: в 1.4x быстрее, до **12x** на малых данных
- **Качество**: +1.14% лучше AUC в среднем
- **Побеждает в 4 из 5** тестов классификации

### 2. Несбалансированные данные

На несбалансированных данных TurboCat отлично работает:

| Датасет | TC AUC | CB AUC | Разница |
|---------|--------|--------|---------|
| Imbalanced 95/5 | **94.4%** | 90.1% | **+4.3%** |
| Imbalanced 99/1 | **78.4%** | 69.2% | **+9.2%** |

### 3. Скорость обучения

- **Малые датасеты** (< 1K): до **12x** быстрее
- **Средние датасеты** (1K-10K): **2-3x** быстрее
- **Большие датасеты** (50K+): сопоставимо

### 4. Параллельное обучение

Новый параметр `n_jobs` для использования всех ядер:

```python
clf = TurboCatClassifier(n_jobs=-1)  # Все ядра
```

---

## Слабые стороны

### 1. Производительность регрессии

TurboCat пока проигрывает CatBoost на задачах регрессии:
- R² в среднем на 2% ниже
- Работаем над улучшениями

### 2. Скорость инференса

CatBoost использует oblivious trees, которые быстрее для предсказания:
- Инференс TurboCat ~2x медленнее
- Реализация oblivious trees в процессе

---

## Когда использовать

### Рекомендуется:

- **Бинарная/Multi-class классификация** — сила TurboCat
- **Fraud detection, медицинская диагностика** — несбалансированные классы
- **Real-time обучение** — до 12x быстрее
- **Средние датасеты** — 1K-50K samples

### Рассмотреть альтернативы:

- Задачи регрессии (CatBoost может быть лучше)
- Очень большие батчи для инференса (CatBoost быстрее)

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
- C++17 компилятор (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+ (установится автоматически)

### Опционально

- OpenMP (для параллельного обучения) - `brew install libomp` на macOS

---

## Быстрый старт

```python
from turbocat import TurboCatClassifier
import numpy as np

# Создание классификатора со всеми ядрами
clf = TurboCatClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    n_jobs=-1  # Использовать все ядра CPU
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

reg = TurboCatRegressor(n_estimators=100, learning_rate=0.1, n_jobs=-1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

### Multi-class классификация

```python
from turbocat import TurboCatClassifier

# Multi-class определяется автоматически
clf = TurboCatClassifier(n_estimators=100)
clf.fit(X_train, y_train)  # y может иметь >2 классов
proba = clf.predict_proba(X_test)  # Возвращает (n_samples, n_classes)
```

---

## Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `n_estimators` | 100 | Количество деревьев |
| `learning_rate` | 0.1 | Скорость обучения |
| `max_depth` | 6 | Максимальная глубина дерева |
| `max_bins` | 255 | Количество бинов гистограммы |
| `subsample` | 0.8 | Доля сэмплов для обучения |
| `colsample_bytree` | 0.8 | Доля признаков для дерева |
| `min_child_weight` | 1.0 | Минимальный вес листа |
| `lambda_l2` | 1.0 | L2 регуляризация |
| `use_goss` | True | Использовать GOSS сэмплирование |
| `n_jobs` | -1 | Количество ядер CPU (-1 = все) |
| `verbosity` | 1 | Уровень вывода (0=тихий) |

---

## Детальный бенчмарк классификации

```
Датасет                TC Acc    CB Acc    TC AUC    CB AUC    Ускорение  Победитель
─────────────────────────────────────────────────────────────────────────────────────
Breast Cancer          96.5%     96.5%     99.4%     99.3%     12.2x      Ничья
Synthetic 2K           94.0%     93.0%     97.9%     97.4%     3.1x       TurboCat
Synthetic 10K          96.9%     96.2%     99.0%     98.8%     1.7x       TurboCat
Synthetic 50K          97.2%     94.8%     99.2%     98.6%     0.85x      TurboCat
Imbalanced 95/5        96.7%     96.9%     94.4%     90.1%     2.5x       TurboCat
─────────────────────────────────────────────────────────────────────────────────────
СРЕДНЕЕ                96.3%     95.5%     98.0%     96.8%     1.41x      TurboCat
```

---

## License

MIT License
