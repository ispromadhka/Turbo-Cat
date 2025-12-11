# 🐱⚡ TurboCat

[🇬🇧 English](#english) | [🇷🇺 Русский](#russian)

---

<a name="english"></a>
# 🇬🇧 English

**Next-generation gradient boosting that matches CatBoost quality while being 3-10x faster.**

TurboCat is a C++ gradient boosting library with Python bindings, implementing cutting-edge research techniques: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, and GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Benchmark Results

Tested on 30 datasets (synthetic, imbalanced, non-linear, high-dimensional, real-world):

### Quality: Parity with CatBoost

| Metric | TurboCat | CatBoost | p-value |
|--------|----------|----------|---------|
| Accuracy | 0.9164 | 0.9171 | 0.87 |
| ROC-AUC | 0.9515 | 0.9568 | 0.17 |
| F1 | **0.8786** | 0.8695 | 0.31 |
| Recall | **0.8657** | 0.8592 | 0.45 |

*No statistically significant difference (t-test, Wilcoxon).*

### Performance: TurboCat is Faster

| Metric | TurboCat vs CatBoost |
|--------|---------------------|
| Training | **3.5x faster** (median 1.8x) |
| Inference | **9.7x faster** (median 6.8x) |
| Max speedup | up to **18.9x** training, **33x** inference |

---

## ✅ Strengths

### 1. Imbalanced Data — Key Advantage

TurboCat performs significantly better on imbalanced datasets:

| Dataset | Recall TC | Recall CB | F1 TC | F1 CB |
|---------|-----------|-----------|-------|-------|
| 70/30 | **91.2%** | 87.4% | **93.6%** | 91.3% |
| 85/15 | **84.7%** | 75.9% | **89.8%** | 84.7% |
| 95/5 | **54.5%** | 45.5% | **70.2%** | 62.1% |
| 99/1 | **15.8%** | 3.5% | **27.3%** | 6.8% |

On extremely imbalanced data (99/1), TurboCat shows **4x higher F1 score**.

### 2. Speed

- Training: Faster on 23/30 datasets
- Inference: Faster on 30/30 datasets
- Particularly effective on small-medium datasets (up to 20x speedup)

### 3. Medium-Large Scale (5K-50K samples)

- Accuracy: 4/5 wins against CatBoost
- ROC-AUC: 4/5 wins

### 4. Special Cases

- **Highly correlated features**: +0.2% ROC-AUC
- **Data with outliers**: +0.3% ROC-AUC
- **High-dim with many informative features**: +3.2% ROC-AUC

---

## ⚠️ Weaknesses

### 1. Noisy Data

On data with >10% label noise, TurboCat loses up to -9.9% ROC-AUC.

### 2. Small Datasets (<1K samples)

CatBoost generalizes better on small samples (1/4 wins by ROC-AUC).

### 3. High-dimensional Sparse Data

With many irrelevant features (200f, 20 informative), CatBoost is slightly better.

---

## 🎯 When to Use

### ✅ Recommended:

- **Fraud detection, medical diagnosis** — imbalanced classes
- **Production deployment** — inference speed is critical
- **Real-time predictions** — up to 33x faster
- **Medium-large datasets** — 5K+ samples

### ⚠️ Consider Alternatives:

- Very noisy data (>10% label noise)
- Very small samples (<500 samples)
- Extreme high-dimensional sparse data

---

## 🛠 Installation

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Requirements

- C++17 compiler (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+
- Python 3.8+
- NumPy

### Optional dependencies

- OpenMP (for parallel training)
- Eigen3 (auto-downloaded if not found)

---

## 🔥 Quick Start

```python
import sys
sys.path.insert(0, 'build')
import _turbocat as tc
import numpy as np

# Create classifier
model = tc.TurboCatClassifier(
    n_estimators=50,
    max_depth=8,
    learning_rate=0.1,
    verbosity=0
)

# Train
model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

# Predict
proba = np.array(model.predict_proba(X_test.astype(np.float32)))
predictions = (proba > 0.5).astype(int)
```

---

## ⚙️ Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_estimators` | 100 | Number of boosting iterations |
| `learning_rate` | 0.1 | Step size shrinkage |
| `max_depth` | 6 | Maximum tree depth |
| `max_bins` | 255 | Histogram bins |
| `subsample` | 1.0 | Row sampling ratio |
| `colsample_bytree` | 1.0 | Feature sampling ratio |
| `min_child_weight` | 1.0 | Minimum leaf hessian sum |
| `lambda_l2` | 1.0 | L2 regularization |
| `verbosity` | 1 | Verbosity level (0=silent) |

---

## 📈 Detailed Benchmark

```
Performance by category (30 datasets):

IMBALANCED:    TC wins Accuracy 4/4, F1 4/4 | Speedup 1.8x train, 5.7x inference
SYNTHETIC:     TC wins ROC-AUC 3/5         | Speedup 1.3x train, 7.3x inference  
SCALE:         TC wins Accuracy 2/3        | Speedup 5.3x train, 9.5x inference
HIGH-DIM:      TC wins Accuracy 2/4        | Speedup 7.1x train, 17.1x inference
SPECIAL:       TC wins Accuracy 3/4        | Speedup 2.0x train, 15.1x inference
```


<a name="russian"></a>
# 🇷🇺 Русский

**Градиентный бустинг нового поколения — качество CatBoost, скорость в 3-10 раз выше.**

TurboCat — библиотека градиентного бустинга на C++ с Python-привязками, реализующая современные исследовательские техники: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📊 Результаты бенчмарков

Тестирование на 30 датасетах (синтетические, несбалансированные, нелинейные, высокоразмерные, реальные):

### Качество: Паритет с CatBoost

| Метрика | TurboCat | CatBoost | p-value |
|---------|----------|----------|---------|
| Accuracy | 0.9164 | 0.9171 | 0.87 |
| ROC-AUC | 0.9515 | 0.9568 | 0.17 |
| F1 | **0.8786** | 0.8695 | 0.31 |
| Recall | **0.8657** | 0.8592 | 0.45 |

*Статистически значимой разницы нет (t-критерий, критерий Уилкоксона).*

### Производительность: TurboCat быстрее

| Метрика | TurboCat vs CatBoost |
|---------|---------------------|
| Обучение | **в 3.5 раза быстрее** (медиана 1.8x) |
| Инференс | **в 9.7 раза быстрее** (медиана 6.8x) |
| Максимум | до **18.9x** обучение, **33x** инференс |

---

## ✅ Сильные стороны

### 1. Несбалансированные данные — главное преимущество

TurboCat значительно лучше на несбалансированных данных:

| Датасет | Recall TC | Recall CB | F1 TC | F1 CB |
|---------|-----------|-----------|-------|-------|
| 70/30 | **91.2%** | 87.4% | **93.6%** | 91.3% |
| 85/15 | **84.7%** | 75.9% | **89.8%** | 84.7% |
| 95/5 | **54.5%** | 45.5% | **70.2%** | 62.1% |
| 99/1 | **15.8%** | 3.5% | **27.3%** | 6.8% |

На экстремально несбалансированных данных (99/1) TurboCat показывает **F1 в 4 раза выше**.

### 2. Скорость

- Обучение: быстрее на 23/30 датасетов
- Инференс: быстрее на 30/30 датасетов
- Особенно эффективен на малых и средних датасетах (до 20x ускорения)

### 3. Средний и большой масштаб (5K-50K samples)

- Accuracy: 4/5 побед над CatBoost
- ROC-AUC: 4/5 побед

### 4. Особые случаи

- **Высококоррелированные признаки**: +0.2% ROC-AUC
- **Данные с выбросами**: +0.3% ROC-AUC
- **Высокоразмерные с информативными признаками**: +3.2% ROC-AUC

---

## ⚠️ Слабые стороны

### 1. Шумные данные

На данных с >10% label noise TurboCat проигрывает до -9.9% ROC-AUC.

### 2. Маленькие датасеты (<1K samples)

CatBoost лучше обобщает на малых выборках (1/4 побед по ROC-AUC).

### 3. Высокоразмерные разреженные данные

При большом количестве нерелевантных признаков (200f, 20 informative) CatBoost немного лучше.

---

## 🎯 Когда использовать

### ✅ Рекомендуется:

- **Fraud detection, медицинская диагностика** — несбалансированные классы
- **Production deployment** — критична скорость инференса
- **Real-time predictions** — до 33x быстрее
- **Средние и большие датасеты** — 5K+ samples

### ⚠️ Рассмотреть альтернативы:

- Очень шумные данные (>10% label noise)
- Очень маленькие выборки (<500 samples)
- Extreme high-dimensional sparse data

---

## 🛠 Установка

```bash
git clone https://github.com/ispromadhka/Turbo-Cat.git
cd Turbo-Cat
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8
```

### Требования

- C++17 компилятор (GCC 10+, Clang 12+, Apple Clang 14+)
- CMake 3.18+
- Python 3.8+
- NumPy

### Опциональные зависимости

- OpenMP (для параллельного обучения)
- Eigen3 (авто-скачивается если не найден)

---

## 🔥 Быстрый старт

```python
import sys
sys.path.insert(0, 'build')
import _turbocat as tc
import numpy as np

# Создание классификатора
model = tc.TurboCatClassifier(
    n_estimators=50,
    max_depth=8,
    learning_rate=0.1,
    verbosity=0
)

# Обучение
model.fit(X_train.astype(np.float32), y_train.astype(np.float32))

# Предсказание
proba = np.array(model.predict_proba(X_test.astype(np.float32)))
predictions = (proba > 0.5).astype(int)
```

---

## ⚙️ Параметры

| Параметр | По умолчанию | Описание |
|----------|--------------|----------|
| `n_estimators` | 100 | Количество деревьев |
| `learning_rate` | 0.1 | Скорость обучения |
| `max_depth` | 6 | Максимальная глубина дерева |
| `max_bins` | 255 | Количество бинов гистограммы |
| `subsample` | 1.0 | Доля сэмплов для обучения |
| `colsample_bytree` | 1.0 | Доля признаков для дерева |
| `min_child_weight` | 1.0 | Минимальный вес листа |
| `lambda_l2` | 1.0 | L2 регуляризация |
| `verbosity` | 1 | Уровень вывода (0=тихий) |

---

## 📈 Детальный бенчмарк

```
Производительность по категориям (30 датасетов):

IMBALANCED:    TC побеждает Accuracy 4/4, F1 4/4 | Ускорение 1.8x train, 5.7x inference
SYNTHETIC:     TC побеждает ROC-AUC 3/5         | Ускорение 1.3x train, 7.3x inference  
SCALE:         TC побеждает Accuracy 2/3        | Ускорение 5.3x train, 9.5x inference
HIGH-DIM:      TC побеждает Accuracy 2/4        | Ускорение 7.1x train, 17.1x inference
SPECIAL:       TC побеждает Accuracy 3/4        | Ускорение 2.0x train, 15.1x inference
```

---

MIT License
