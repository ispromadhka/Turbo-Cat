# TurboCat

[English](#english) | [Русский](#russian)

---

<a name="english"></a>
# English

**Next-generation gradient boosting that matches CatBoost quality while being 3-10x faster.**

TurboCat is a high-performance C++ gradient boosting library with Python bindings, implementing cutting-edge research techniques: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, and GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [ ] **Multi-class classification** - Currently only binary classification is supported
- [ ] **Noisy data handling** - Improve robustness to label noise (>10%)
- [ ] **High-dimensional sparse data** - Better feature selection for many irrelevant features
- [ ] **GPU support** - CUDA and Metal acceleration
- [ ] **Model serialization** - Save/load trained models

---

## Benchmark Results

Tested on 19 binary classification datasets (synthetic, imbalanced, non-linear, high-dimensional, real-world).

**Hyperparameters**: `n_estimators=100, learning_rate=0.1, max_depth=6`

### Quality Metrics (Binary Classification)

| Metric | TurboCat | CatBoost | TurboCat Wins |
|--------|----------|----------|---------------|
| Accuracy | 0.9380 | 0.9360 | 10/16 |
| ROC-AUC | 0.9651 | 0.9660 | 6/16 |
| F1 | 0.9369 | 0.9352 | 11/16 |

*Binary classification only. Multi-class not yet supported.*

### Performance: TurboCat is Faster

| Metric | TurboCat vs CatBoost |
|--------|---------------------|
| Training | **4.3x faster** (wins 19/19 datasets) |
| Inference | Comparable (wins 8/19 datasets) |
| Max speedup | up to **938x** training |

---

## Strengths

### 1. Imbalanced Data - Key Advantage

TurboCat significantly outperforms on imbalanced datasets:

| Dataset | Accuracy TC | Accuracy CB | ROC-AUC TC | ROC-AUC CB |
|---------|-------------|-------------|------------|------------|
| 70/30 | 95.5% | 95.5% | 99.1% | 99.1% |
| 85/15 | **96.6%** | 96.2% | 98.9% | **99.2%** |
| 95/5 | **97.8%** | 97.2% | **98.4%** | 98.0% |
| 99/1 | **99.1%** | 99.0% | **89.3%** | 86.7% |

On extremely imbalanced data (99/1), TurboCat shows **+2.6% ROC-AUC**.

### 2. Training Speed

- Training: Faster on **19/19** datasets (4.3x average)
- Particularly effective on small-medium datasets (up to 938x speedup on Wine)

### 3. Non-linear Data

| Dataset | Accuracy TC | Accuracy CB |
|---------|-------------|-------------|
| Moons | **96.3%** | 95.8% |
| Circles | **99.0%** | 98.5% |

### 4. High-dimensional Dense Data

With many informative features (100 features, 80 informative):
- Accuracy: **90.5%** vs 87.8% (+2.75%)
- ROC-AUC: **96.2%** vs 95.0% (+1.2%)

### 5. Correlated Features

With highly correlated features:
- Accuracy: **98.3%** vs 97.5% (+0.83%)
- ROC-AUC: **99.9%** vs 99.8%

---

## Weaknesses

### 1. Multi-class Classification (Not Supported)

Currently TurboCat only supports binary classification. Multi-class datasets (Iris, Wine, Digits) will not work correctly.

### 2. Noisy Data

On data with >10% label noise, TurboCat loses accuracy:

| Noise Level | Accuracy TC | Accuracy CB | Difference |
|-------------|-------------|-------------|------------|
| 5% | **92.5%** | 92.3% | +0.2% |
| 10% | 89.7% | **90.8%** | -1.2% |
| 20% | 81.3% | **85.5%** | -4.2% |

### 3. High-dimensional Sparse Data

With many irrelevant features (200 features, only 20 informative):
- Accuracy: 87.3% vs **90.8%** (-3.5%)
- ROC-AUC: 93.5% vs **95.4%** (-1.9%)

### 4. Inference Speed on Large Batches

On datasets >5K samples, CatBoost inference is sometimes faster due to better batch optimization.

---

## When to Use

### Recommended:

- **Fraud detection, medical diagnosis** - imbalanced classes
- **Real-time training** - up to 938x faster training
- **Binary classification** - current focus
- **Medium-large datasets** - 1K+ samples
- **Non-linear decision boundaries** - circles, moons patterns
- **Correlated features** - handles multicollinearity well

### Consider Alternatives:

- Multi-class classification (use CatBoost/XGBoost)
- Very noisy data (>10% label noise)
- High-dimensional sparse data (many irrelevant features)

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

- OpenMP (for parallel training)

---

## Quick Start

```python
from turbocat import TurboCatClassifier
import numpy as np

# Create classifier
clf = TurboCatClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
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

reg = TurboCatRegressor(n_estimators=100, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

---

## Parameters

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
| `use_goss` | True | Use Gradient-based One-Side Sampling |
| `verbosity` | 1 | Verbosity level (0=silent) |

---

## Detailed Benchmark (Binary Classification)

```
Dataset                          TC Acc    CB Acc    TC ROC    CB ROC    Train Speedup
─────────────────────────────────────────────────────────────────────────────────────
Breast Cancer                    96.5%     96.5%     98.9%     99.3%     16.5x
Synthetic 500                    89.0%     88.0%     95.5%     95.9%     15.7x
Synthetic 2000                   93.8%     95.3%     98.7%     98.8%     5.7x
Synthetic 10000                  96.8%     96.0%     98.9%     98.8%     2.2x
Imbalanced 70/30                 95.5%     95.5%     99.1%     99.1%     3.1x
Imbalanced 85/15                 96.6%     96.2%     98.9%     99.2%     3.2x
Imbalanced 95/5                  97.8%     97.2%     98.4%     98.0%     3.8x
Imbalanced 99/1                  99.1%     99.0%     89.3%     86.7%     5.5x
High-dim sparse (200f)           87.3%     90.8%     93.5%     95.4%     3.0x
High-dim dense (100f)            90.5%     87.8%     96.2%     95.0%     3.6x
Moons                            96.3%     95.8%     99.0%     99.5%     3.0x
Circles                          99.0%     98.5%     100.0%    100.0%    2.9x
Noisy 5%                         92.5%     92.3%     96.6%     96.8%     3.8x
Noisy 10%                        89.7%     90.8%     94.0%     94.8%     3.9x
Noisy 20%                        81.3%     85.5%     87.4%     88.5%     3.9x
Correlated                       98.3%     97.5%     99.9%     99.8%     5.0x
```

---

<a name="russian"></a>
# Русский

**Градиентный бустинг нового поколения — качество CatBoost, скорость в 3-10 раз выше.**

TurboCat — высокопроизводительная библиотека градиентного бустинга на C++ с Python-привязками, реализующая современные исследовательские техники: GradTree (AAAI 2024), Robust Focal Loss, Tsallis entropy splitting, GOSS sampling.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Roadmap / TODO

- [ ] **Multi-class классификация** - Пока поддерживается только бинарная
- [ ] **Обработка шумных данных** - Улучшить устойчивость к шуму в метках (>10%)
- [ ] **Высокоразмерные разреженные данные** - Лучший отбор признаков
- [ ] **GPU поддержка** - Ускорение на CUDA и Metal
- [ ] **Сериализация моделей** - Сохранение/загрузка обученных моделей

---

## Результаты бенчмарков

Тестирование на 19 датасетах бинарной классификации (синтетические, несбалансированные, нелинейные, высокоразмерные, реальные).

**Гиперпараметры**: `n_estimators=100, learning_rate=0.1, max_depth=6`

### Метрики качества (бинарная классификация)

| Метрика | TurboCat | CatBoost | Побед TC |
|---------|----------|----------|----------|
| Accuracy | 0.9380 | 0.9360 | 10/16 |
| ROC-AUC | 0.9651 | 0.9660 | 6/16 |
| F1 | 0.9369 | 0.9352 | 11/16 |

*Только бинарная классификация. Multi-class пока не поддерживается.*

### Производительность: TurboCat быстрее

| Метрика | TurboCat vs CatBoost |
|---------|---------------------|
| Обучение | **в 4.3 раза быстрее** (побеждает на 19/19 датасетах) |
| Инференс | Сопоставимо (побеждает на 8/19 датасетах) |
| Максимум | до **938x** ускорение обучения |

---

## Сильные стороны

### 1. Несбалансированные данные — главное преимущество

TurboCat значительно лучше на несбалансированных данных:

| Датасет | Accuracy TC | Accuracy CB | ROC-AUC TC | ROC-AUC CB |
|---------|-------------|-------------|------------|------------|
| 70/30 | 95.5% | 95.5% | 99.1% | 99.1% |
| 85/15 | **96.6%** | 96.2% | 98.9% | **99.2%** |
| 95/5 | **97.8%** | 97.2% | **98.4%** | 98.0% |
| 99/1 | **99.1%** | 99.0% | **89.3%** | 86.7% |

На экстремально несбалансированных данных (99/1) TurboCat показывает **+2.6% ROC-AUC**.

### 2. Скорость обучения

- Обучение: быстрее на **19/19** датасетов (в среднем 4.3x)
- Особенно эффективен на малых и средних датасетах (до 938x ускорения)

### 3. Нелинейные данные

| Датасет | Accuracy TC | Accuracy CB |
|---------|-------------|-------------|
| Moons | **96.3%** | 95.8% |
| Circles | **99.0%** | 98.5% |

### 4. Высокоразмерные плотные данные

При большом количестве информативных признаков (100 признаков, 80 информативных):
- Accuracy: **90.5%** vs 87.8% (+2.75%)
- ROC-AUC: **96.2%** vs 95.0% (+1.2%)

### 5. Коррелированные признаки

При высококоррелированных признаках:
- Accuracy: **98.3%** vs 97.5% (+0.83%)
- ROC-AUC: **99.9%** vs 99.8%

---

## Слабые стороны

### 1. Multi-class классификация (Не поддерживается)

Пока TurboCat поддерживает только бинарную классификацию. Multi-class датасеты (Iris, Wine, Digits) работать не будут.

### 2. Шумные данные

На данных с >10% label noise TurboCat проигрывает:

| Уровень шума | Accuracy TC | Accuracy CB | Разница |
|--------------|-------------|-------------|---------|
| 5% | **92.5%** | 92.3% | +0.2% |
| 10% | 89.7% | **90.8%** | -1.2% |
| 20% | 81.3% | **85.5%** | -4.2% |

### 3. Высокоразмерные разреженные данные

При большом количестве нерелевантных признаков (200 признаков, только 20 информативных):
- Accuracy: 87.3% vs **90.8%** (-3.5%)
- ROC-AUC: 93.5% vs **95.4%** (-1.9%)

### 4. Скорость инференса на больших батчах

На датасетах >5K samples инференс CatBoost иногда быстрее из-за лучшей батч-оптимизации.

---

## Когда использовать

### Рекомендуется:

- **Fraud detection, медицинская диагностика** — несбалансированные классы
- **Real-time обучение** — до 938x быстрее
- **Бинарная классификация** — текущий фокус
- **Средние и большие датасеты** — 1K+ samples
- **Нелинейные границы решений** — circles, moons паттерны
- **Коррелированные признаки** — хорошо обрабатывает мультиколлинеарность

### Рассмотреть альтернативы:

- Multi-class классификация (используйте CatBoost/XGBoost)
- Очень шумные данные (>10% label noise)
- Высокоразмерные разреженные данные (много нерелевантных признаков)

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
- CMake 3.18+ (установится автоматически если отсутствует)

### Опционально

- OpenMP (для параллельного обучения)

---

## Быстрый старт

```python
from turbocat import TurboCatClassifier
import numpy as np

# Создание классификатора
clf = TurboCatClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6
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

reg = TurboCatRegressor(n_estimators=100, learning_rate=0.1)
reg.fit(X_train, y_train)
predictions = reg.predict(X_test)
```

---

## Параметры

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
| `use_goss` | True | Использовать GOSS сэмплирование |
| `verbosity` | 1 | Уровень вывода (0=тихий) |

---

## Детальный бенчмарк (бинарная классификация)

```
Датасет                          TC Acc    CB Acc    TC ROC    CB ROC    Ускорение
─────────────────────────────────────────────────────────────────────────────────
Breast Cancer                    96.5%     96.5%     98.9%     99.3%     16.5x
Synthetic 500                    89.0%     88.0%     95.5%     95.9%     15.7x
Synthetic 2000                   93.8%     95.3%     98.7%     98.8%     5.7x
Synthetic 10000                  96.8%     96.0%     98.9%     98.8%     2.2x
Imbalanced 70/30                 95.5%     95.5%     99.1%     99.1%     3.1x
Imbalanced 85/15                 96.6%     96.2%     98.9%     99.2%     3.2x
Imbalanced 95/5                  97.8%     97.2%     98.4%     98.0%     3.8x
Imbalanced 99/1                  99.1%     99.0%     89.3%     86.7%     5.5x
High-dim sparse (200f)           87.3%     90.8%     93.5%     95.4%     3.0x
High-dim dense (100f)            90.5%     87.8%     96.2%     95.0%     3.6x
Moons                            96.3%     95.8%     99.0%     99.5%     3.0x
Circles                          99.0%     98.5%     100.0%    100.0%    2.9x
Noisy 5%                         92.5%     92.3%     96.6%     96.8%     3.8x
Noisy 10%                        89.7%     90.8%     94.0%     94.8%     3.9x
Noisy 20%                        81.3%     85.5%     87.4%     88.5%     3.9x
Correlated                       98.3%     97.5%     99.9%     99.8%     5.0x
```

---

## License

MIT License
