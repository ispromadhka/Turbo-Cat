# 🛠️ Инструкция по установке TurboCat

## Содержание

1. [Требования](#требования)
2. [Установка зависимостей](#установка-зависимостей)
3. [Сборка из исходников](#сборка-из-исходников)
4. [Python биндинги](#python-биндинги)
5. [Проверка установки](#проверка-установки)
6. [Troubleshooting](#troubleshooting)

---

## Требования

### Минимальные

| Компонент | Версия |
|-----------|--------|
| C++ компилятор | GCC 10+ / Clang 12+ / MSVC 2019+ |
| CMake | 3.18+ |
| Git | любая |

### Рекомендуемые (для максимальной производительности)

| Компонент | Версия | Зачем |
|-----------|--------|-------|
| GCC | 11+ | Лучшая поддержка AVX-512 |
| OpenMP | 4.5+ | Многопоточность |
| Eigen3 | 3.4+ | GradTree оптимизация |
| Python | 3.8+ | Python API |

---

## Установка зависимостей

### Ubuntu / Debian

```bash
# Базовые инструменты
sudo apt update
sudo apt install -y build-essential cmake git

# OpenMP
sudo apt install -y libomp-dev

# Eigen3 (опционально, CMake скачает если нет)
sudo apt install -y libeigen3-dev

# Python биндинги
sudo apt install -y python3-dev python3-pip
pip3 install numpy pybind11
```

### Fedora / RHEL / CentOS

```bash
sudo dnf install -y gcc-c++ cmake git
sudo dnf install -y libomp-devel eigen3-devel
sudo dnf install -y python3-devel python3-pip
pip3 install numpy pybind11
```

### macOS

```bash
# Homebrew
brew install cmake libomp eigen

# Python
pip3 install numpy pybind11
```

### Windows

```powershell
# Используйте Visual Studio 2019+ с C++ workload
# Установите CMake: https://cmake.org/download/

# Или через vcpkg:
vcpkg install eigen3 pybind11
```

---

## Сборка из исходников

### Шаг 1: Клонирование

```bash
git clone https://github.com/yourusername/turbocat.git
cd turbocat
```

### Шаг 2: Создание build директории

```bash
mkdir build
cd build
```

### Шаг 3: Конфигурация CMake

**Базовая сборка:**
```bash
cmake .. -DCMAKE_BUILD_TYPE=Release
```

**Полная сборка со всеми опциями:**
```bash
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTURBOCAT_BUILD_PYTHON=ON \
    -DTURBOCAT_BUILD_TESTS=ON \
    -DTURBOCAT_BUILD_BENCHMARKS=ON \
    -DTURBOCAT_USE_OPENMP=ON \
    -DCMAKE_INSTALL_PREFIX=/usr/local
```

**Опции CMake:**

| Опция | По умолчанию | Описание |
|-------|--------------|----------|
| `TURBOCAT_BUILD_PYTHON` | ON | Собирать Python биндинги |
| `TURBOCAT_BUILD_TESTS` | ON | Собирать тесты |
| `TURBOCAT_BUILD_BENCHMARKS` | ON | Собирать бенчмарки |
| `TURBOCAT_USE_OPENMP` | ON | Использовать OpenMP |
| `CMAKE_BUILD_TYPE` | Release | Debug/Release/RelWithDebInfo |

### Шаг 4: Компиляция

```bash
# Используйте все ядра
make -j$(nproc)

# Или для Windows
cmake --build . --config Release --parallel
```

### Шаг 5: Тестирование (опционально)

```bash
# Запуск всех тестов
ctest --output-on-failure

# Или напрямую
./turbocat_tests
```

### Шаг 6: Установка

```bash
# Системная установка
sudo make install

# Или в пользовательскую директорию
cmake .. -DCMAKE_INSTALL_PREFIX=$HOME/.local
make install
```

---

## Python биндинги

### Способ 1: Через pip (рекомендуется)

После сборки:
```bash
cd ../python
pip install -e .
```

### Способ 2: Копирование модуля

```bash
# Найти собранный модуль
find build -name "_turbocat*.so" -o -name "_turbocat*.pyd"

# Скопировать в site-packages
cp build/_turbocat*.so $(python -c "import site; print(site.getsitepackages()[0])")/
```

### Проверка Python установки

```python
import turbocat as tc

# Вывести информацию о библиотеке
tc.print_info()

# Должно показать:
# TurboCat v0.1.0
#   SIMD: AVX-512 (или AVX2)
#   OpenMP: Yes
#   CUDA: No
#   Metal: No
```

---

## Проверка установки

### C++ тест

Создайте файл `test_turbocat.cpp`:

```cpp
#include <turbocat/turbocat.hpp>
#include <iostream>
#include <vector>

int main() {
    turbocat::print_info();
    
    // Простой тест
    std::vector<float> X = {1, 2, 3, 4, 5, 6};  // 2 samples, 3 features
    std::vector<float> y = {0, 1};
    
    turbocat::Config config = turbocat::Config::binary_classification();
    config.boosting.n_estimators = 10;
    
    turbocat::Dataset data;
    data.from_dense(X.data(), 2, 3, y.data());
    data.compute_bins(config);
    
    turbocat::Booster model(config);
    model.train(data);
    
    std::cout << "✅ TurboCat работает!" << std::endl;
    return 0;
}
```

Компиляция:
```bash
g++ -std=c++20 test_turbocat.cpp -o test_turbocat \
    -I/usr/local/include \
    -L/usr/local/lib -lturbocat_core \
    -fopenmp -mavx2
./test_turbocat
```

### Python тест

```python
import turbocat as tc
import numpy as np

# Генерируем данные
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)

# Обучаем модель
model = tc.Booster(task='binary', n_estimators=100)
model.fit(X, y)

# Предсказываем
preds = model.predict_proba(X)
accuracy = ((preds > 0.5) == y).mean()

print(f"✅ Accuracy: {accuracy:.2%}")
```

---

## Troubleshooting

### Ошибка: "AVX-512 not supported"

```bash
# Проверьте поддержку CPU
grep -o 'avx[^ ]*' /proc/cpuinfo | sort -u

# Если нет AVX-512, соберите с AVX2:
cmake .. -DCMAKE_CXX_FLAGS="-mavx2"
```

### Ошибка: "OpenMP not found"

```bash
# Ubuntu
sudo apt install libomp-dev

# macOS (с Homebrew)
brew install libomp
export OpenMP_ROOT=$(brew --prefix)/opt/libomp
```

### Ошибка: "Eigen3 not found"

CMake автоматически скачает Eigen3. Если нужна системная версия:
```bash
sudo apt install libeigen3-dev
cmake .. -DEigen3_DIR=/usr/share/eigen3/cmake
```

### Ошибка Python: "ModuleNotFoundError: No module named 'turbocat'"

```bash
# Проверьте путь к модулю
python -c "import sys; print(sys.path)"

# Добавьте путь к build
export PYTHONPATH=$PYTHONPATH:/path/to/turbocat/build
```

### Медленная работа

1. Убедитесь что собрали в Release:
   ```bash
   cmake .. -DCMAKE_BUILD_TYPE=Release
   ```

2. Проверьте SIMD флаги в выводе CMake:
   ```
   TurboCat: AVX-512 support enabled
   ```

3. Включите OpenMP:
   ```bash
   export OMP_NUM_THREADS=$(nproc)
   ```

---

## Следующие шаги

После успешной установки:

1. 📖 Изучите [README.md](README.md) для примеров использования
2. 🧪 Запустите бенчмарки: `./turbocat_bench`
3. 📊 Попробуйте на своих данных

---

**Вопросы?** Создайте issue на GitHub!
