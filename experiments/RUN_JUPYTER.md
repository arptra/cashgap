# Запуск экспериментов cash-flow в Jupyter

Инструкция рассчитана на Linux EL8, Python 3.8, две NVIDIA GPU и два исходных
Parquet-файла:

- `outflow.parquet`: `tr_date`, `dt_inn` (или `dtinn`), `tr_sum` — списания;
- `inflow.parquet`: `tr_date`, `kt_inn` (или `ktinn`), `tr_sum` — зачисления.

`tr_date` может быть числом вида `20250530`: скрипты распознают формат
`YYYYMMDD` автоматически.

Все команды ниже выполняются в отдельных ячейках Jupyter. Замените пути
`/data/outflow.parquet` и `/data/inflow.parquet` на реальные.

## 1. Получить последнюю версию проекта

Если репозиторий уже склонирован:

```python
%cd /путь/до/cashgap
!git pull origin master
```

Если репозитория на сервере ещё нет:

```python
%cd /папка/для/проекта
!git clone https://github.com/arptra/cashgap.git
%cd cashgap
```

## 2. Проверить Python и GPU

```python
import sys
import torch

print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA доступна:", torch.cuda.is_available())
print("Количество GPU:", torch.cuda.device_count())

for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
```

Дополнительно:

```python
!nvidia-smi
```

Ожидаемый результат: `CUDA доступна: True`, количество GPU — `2`.

## 3. Установить недостающие библиотеки

Каждый скрипт сначала сам проверяет зависимости и при ошибке печатает команду
установки. Для полной установки под Python 3.8 можно выполнить:

```python
%pip install numpy pandas pyarrow scikit-learn threadpoolctl "xgboost==2.1.4" "optuna==3.6.2"
```

Для API-сервера под Python 3.8:

```python
%pip install "fastapi==0.103.2" "uvicorn==0.23.2"
```

Torch с CUDA 12.1:

```python
%pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu121
```

Если драйвер рассчитан на CUDA 11.8:

```python
%pip install "torch==2.3.1" --index-url https://download.pytorch.org/whl/cu118
```

После установки обязательно перезапустите kernel Jupyter и повторите проверку
из раздела 2. Одновременно устанавливать варианты `cu121` и `cu118` не нужно.

## 4. Проверить структуру Parquet

```python
import pyarrow.parquet as pq

OUTFLOW = "/data/outflow.parquet"
INFLOW = "/data/inflow.parquet"

print("OUTFLOW")
print(pq.read_schema(OUTFLOW))
print("INFLOW")
print(pq.read_schema(INFLOW))
```

В первом файле должны быть `tr_date`, `dt_inn`/`dtinn`, `tr_sum`, во втором —
`tr_date`, `kt_inn`/`ktinn`, `tr_sum`. Регистр, подчёркивания и другие знаки в
именах при сопоставлении игнорируются.

## 5. Быстрый пробный запуск

Этот запуск проверяет весь pipeline на небольшой части ИНН и двух тестовых
месяцах. Его результаты не используются как итоговые.

```python
%run experiments/benchmark_monthly_cashflow.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/monthly_smoke" \
  --models trailing_mean,linear_regression,gradient_boosting,torch_mlp_2_layers,torch_mlp_3_layers \
  --test-periods 2 \
  --min-train-months 12 \
  --max-inns 1000 \
  --epochs 3 \
  --batch-size 32768 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1
```

## 6. Полное базовое обучение и проверка на 10 месяцах

Это основной первый запуск. В каждом из десяти fold модели обучаются только на
прошлом, а затем проверяются на очередном будущем месяце. MLP из двух слоёв
работает на `cuda:0`, MLP из трёх слоёв — на `cuda:1`; линейная регрессия,
прогноз по среднему и gradient boosting работают на CPU.

```python
%run experiments/benchmark_monthly_cashflow.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/monthly_benchmark_base" \
  --models trailing_mean,linear_regression,gradient_boosting,torch_mlp_2_layers,torch_mlp_3_layers \
  --test-periods 10 \
  --min-train-months 12 \
  --epochs 100 \
  --batch-size 32768 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1
```

Результаты сохраняются в `artifacts/monthly_benchmark_base`. Основные
бизнес-отчёты полностью на русском:

- `отчет_стабильность_моделей.csv` — рейтинг и стабильность моделей;
- `отчет_метрики_по_периодам.csv` — метрики каждого тестового месяца;
- `отчет_окна_тестирования.csv` — границы обучения, валидации и теста;
- `отчет_прогнозы_по_инн.parquet` — факт и прогноз каждой компании;
- `бизнес_вывод.txt` — короткий вывод с лучшей моделью для каждого потока.

CSV записываются в UTF-8 с BOM и разделителем `;`, поэтому корректно открываются
в русском Excel. Для обратной совместимости рядом остаются технические файлы:

- `monthly_stability_summary.csv` — среднее качество и стабильность за 10 месяцев;
- `monthly_fold_metrics.csv` — метрики каждого месяца;
- `monthly_fold_windows.csv` — границы train/validation/test;
- `monthly_predictions.parquet` — факт и прогноз по каждому ИНН;
- `run_config.json` — параметры запуска.

### Full-parallel профиль: 64 CPU, 500 ГБ RAM, две GPU по 40 ГБ

Этот профиль использует все модели одновременно, выделяет CPU-моделям до 12
потоков каждой и делает обе MLP достаточно широкими для полезной нагрузки GPU.
Месячные признаки разделяются между моделями в RAM, а folds создаются по одному.

```python
%run experiments/benchmark_monthly_cashflow.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/monthly_benchmark_full_parallel" \
  --models trailing_mean,linear_regression,gradient_boosting,torch_mlp_2_layers,torch_mlp_3_layers \
  --test-periods 10 \
  --min-train-months 12 \
  --epochs 150 \
  --batch-size 32768 \
  --boosting-iterations 500 \
  --cpu-threads 12 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1 \
  --mlp2-layers 2048,1024 \
  --mlp3-layers 2048,1536,768
```

В начале запуска лог печатает реальное распределение ресурсов, архитектуры,
число строк, размер batch и количество параметров. Минимальный процент загрузки
GPU нельзя гарантировать независимо от числа строк, но эти настройки увеличивают
полезные матричные вычисления, не добавляя искусственный stress-нагрев.

### Изолированный full-parallel режим при restart kernel

Если один Jupyter-процесс стабилен только на трёх folds, используйте отдельный
runner. Каждый тестовый месяц выполняется в новом дочернем Python-процессе, но
внутри месяца все пять моделей по-прежнему запускаются параллельно. После fold
операционная система полностью освобождает CUDA contexts, VRAM, native thread
pools и RAM. Затем runner потоково объединяет результаты всех месяцев.

```python
%run experiments/benchmark_monthly_isolated.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/monthly_benchmark_isolated" \
  --models trailing_mean,linear_regression,gradient_boosting,torch_mlp_2_layers,torch_mlp_3_layers \
  --test-periods 10 \
  --min-train-months 12 \
  --epochs 100 \
  --batch-size 4096 \
  --boosting-iterations 250 \
  --cpu-threads 8 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1 \
  --mlp2-layers 512,256 \
  --mlp3-layers 768,512,256 \
  --save-model torch_mlp_3_layers \
  --forecast-months 12 \
  --model-output-dir "./artifacts/monthly_benchmark_isolated/saved_model"
```

Объединённые файлы имеют те же имена, что у обычного benchmark. Постоянный лог
каждого дочернего запуска хранится в
`artifacts/monthly_benchmark_isolated/isolated_folds/fold_XX_offset_XX/child.log`.
Если нативная библиотека или ОС завершит конкретный child, основной notebook
останется жив, а runner покажет exit code и путь к последнему логу.

`--save-model` — отдельный параметр финального этапа. После честной проверки на
всех тестовых периодах выбранная модель ещё один раз обучается для эксплуатации
и записывается в файл:

- для MLP: `saved_model/model.pt`;
- для линейной регрессии и бустинга: `saved_model/model.joblib`;
- метаданные и описание: `model_metadata.json`, `описание_модели.json`;
- прогнозы, которые читает API: `forecasts_api.parquet`;
- русский отчёт: `прогнозы_для_api.csv` и `прогнозы_для_api.parquet`.

Значение `--save-model` должно присутствовать в `--models`. Выбирайте победителя
по тестовым метрикам, а не обязательно самую глубокую сеть.

Первый будущий месяц является прямым прогнозом. Месяцы со второго по двенадцатый
рекурсивные: предыдущий прогноз используется как часть истории следующего
месяца, поэтому неопределённость растёт с горизонтом.

Если benchmark уже завершён и победитель выбран, повторять тестовые периоды не
нужно. Сохраните модель отдельной командой:

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "-u", "experiments/export_monthly_model.py",
    "--outflow", "/data/outflow.parquet",
    "--inflow", "/data/inflow.parquet",
    "--output-dir", "./artifacts/monthly_benchmark_isolated/saved_model",
    "--model", "torch_mlp_3_layers",
    "--device", "cuda:1",
    "--epochs", "100",
    "--batch-size", "4096",
    "--mlp3-layers", "768,512,256",
    "--forecast-months", "12",
], check=True)
```

### Стабильный режим при `RuntimeError: exit code -11`

Код `-11` означает нативный `SIGSEGV`, а не Python exception. Если даже
изолированный benchmark падает на 10 периодах, не используйте для этого запуска
`benchmark_monthly_isolated.py`. Запускайте специальный Torch-only pipeline:

```python
%run experiments/benchmark_torch_sequential.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/torch_10_sequential" \
  --model torch_mlp_2_layers \
  --test-periods 10 \
  --min-train-months 12 \
  --epochs 100 \
  --batch-size 32768 \
  --layers 4096,2048 \
  --devices cuda:0,cuda:1 \
  --cpu-threads 8 \
  --amp
```

Здесь нет флага `--parallel`. Периоды идут строго по одному: первый на
`cuda:0`, второй на `cuda:1`, третий снова на `cuda:0` и так далее. Активная GPU
получает широкую сеть `4096,2048`, большой batch и AMP/Tensor Cores. Вторая GPU
в этот момент намеренно свободна — это плата за максимально устойчивый режим.

Главное отличие от прежнего isolated runner: PyArrow/pandas готовят данные один
раз в родительском процессе без CUDA. Каждый test-период запускается в новом
worker, который импортирует только NumPy и PyTorch, читает `.npy` и после записи
результатов завершает процесс без проблемного native CUDA cleanup.

Если сервер или конкретный worker всё-таки завершится, повторите ту же команду,
добавив:

```text
--resume
```

Уже завершённые периоды и подготовленные Parquet не пересчитываются. Подробный
лог и автоматическая диагностика сигнала находятся здесь:

```text
artifacts/torch_10_sequential/sequential_folds/fold_XX_YYYYMM/worker.log
artifacts/torch_10_sequential/sequential_folds/fold_XX_YYYYMM/failure_diagnostics.txt
```

В `worker.log` каждые пять эпох выводятся загрузка GPU, VRAM, мощность и
температура. Все итоговые русские отчёты записываются прямо в
`artifacts/torch_10_sequential`.

## 7. Посмотреть результаты базового обучения

Итоговый рейтинг моделей:

```python
import pandas as pd

summary = pd.read_csv(
    "./artifacts/monthly_benchmark_base/monthly_stability_summary.csv"
)
display(summary.sort_values(["flow", "aggregate_mape_mean_percent"]))
```

Метрики всех десяти месяцев:

```python
fold_metrics = pd.read_csv(
    "./artifacts/monthly_benchmark_base/monthly_fold_metrics.csv"
)
display(fold_metrics.head(30))
```

Факт и прогноз по ИНН:

```python
predictions = pd.read_parquet(
    "./artifacts/monthly_benchmark_base/monthly_predictions.parquet"
)
display(predictions.sample(min(20, len(predictions))))
```

Основной показатель — `aggregate_mape_mean_percent`: средняя процентная ошибка
совокупных зачислений или списаний за месяц. Чем меньше, тем лучше.
`aggregate_mape_std_percent` показывает стабильность, а
`aggregate_mape_worst_percent` — ошибку в худшем тестовом месяце.

Русский рейтинг в notebook:

```python
summary_ru = pd.read_csv(
    "./artifacts/monthly_benchmark_isolated/отчет_стабильность_моделей.csv",
    sep=";",
)
display(summary_ru)
```

## 8. Автотюнинг MLP на двух GPU

Финальные 10 месяцев защищены от автотюнинга. Два Optuna trial запускаются
параллельно: один на `cuda:0`, второй на `cuda:1`.

```python
%run experiments/autotune_cashflow.py \
  --model mlp \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/cashflow_tuning/mlp" \
  --trials 30 \
  --jobs 2 \
  --devices cuda:0,cuda:1 \
  --tuning-periods 3 \
  --holdout-test-periods 10 \
  --min-train-months 12 \
  --epochs 70
```

Лучшие параметры сохранятся в:

```text
artifacts/cashflow_tuning/mlp/best_params.json
```

Повторный запуск той же команды продолжает существующее Optuna study и добавляет
ещё указанное количество trials.

Для указанной конфигурации можно проводить более широкий поиск до шести слоёв и
2048 нейронов в первом слое, запуская примерно четыре trials на каждой GPU:

```python
%run experiments/autotune_cashflow.py \
  --model mlp \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/cashflow_tuning_full" \
  --trials 100 \
  --jobs 8 \
  --devices cuda:0,cuda:1 \
  --tuning-periods 3 \
  --holdout-test-periods 10 \
  --min-train-months 12 \
  --epochs 120
```

## 9. Автотюнинг gradient boosting

Gradient boosting из этого эксперимента обучается на CPU.

```python
%run experiments/autotune_cashflow.py \
  --model gradient_boosting \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/cashflow_tuning/gradient_boosting" \
  --trials 40 \
  --jobs 2 \
  --tuning-periods 3 \
  --holdout-test-periods 10 \
  --min-train-months 12
```

Лучшие параметры сохранятся в:

```text
artifacts/cashflow_tuning/gradient_boosting/best_params.json
```

## 10. Финальное сравнение с подобранными настройками

Tuned MLP запускается на `cuda:0`, обычная трёхслойная MLP — на `cuda:1`.
Базовая двухслойная сеть здесь не запускается повторно: её результат уже есть в
`monthly_benchmark_base`.

```python
%run experiments/benchmark_monthly_cashflow.py \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/monthly_benchmark_tuned" \
  --models trailing_mean,linear_regression,gradient_boosting,torch_mlp_3_layers,torch_mlp_tuned \
  --test-periods 10 \
  --min-train-months 12 \
  --epochs 100 \
  --batch-size 32768 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1 \
  --mlp-params "./artifacts/cashflow_tuning/mlp/best_params.json" \
  --boosting-params "./artifacts/cashflow_tuning/gradient_boosting/best_params.json"
```

Посмотреть финальный рейтинг:

```python
final_summary = pd.read_csv(
    "./artifacts/monthly_benchmark_tuned/monthly_stability_summary.csv"
)
display(final_summary.sort_values(["flow", "aggregate_mape_mean_percent"]))
```

## 11. Дневной прогноз риска отрицательного чистого потока

Это отдельный эксперимент. Он прогнозирует на 14 дней вероятность того, что
дневной чистый поток будет отрицательным, а также приход, расход и чистый поток.
Это proxy ликвидности, а не настоящий кассовый разрыв: без остатков на счетах
нельзя определить, хватит ли денег.

Обучение:

```python
%run experiments/train_cashflow_proxy.py train \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/cashflow_daily" \
  --horizon 14 \
  --test-days 90 \
  --validation-days 90 \
  --epochs 120 \
  --batch-size 4096 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1
```

Показать случайный реальный пример из скрытой исторической test-выборки без
повторного обучения:

```python
%run experiments/train_cashflow_proxy.py demo \
  --output-dir "./artifacts/cashflow_daily" \
  --demo-model best
```

Каждый новый запуск `demo` выбирает другой случай. Для воспроизводимого примера:

```python
%run experiments/train_cashflow_proxy.py demo \
  --output-dir "./artifacts/cashflow_daily" \
  --demo-model best \
  --demo-seed 42
```

Обучить и сразу показать demo одной командой:

```python
%run experiments/train_cashflow_proxy.py train-demo \
  --outflow "/data/outflow.parquet" \
  --inflow "/data/inflow.parquet" \
  --output-dir "./artifacts/cashflow_daily" \
  --horizon 14 \
  --epochs 120 \
  --parallel \
  --mlp2-device cuda:0 \
  --mlp3-device cuda:1
```

## 12. Запустить API прогноза по ИНН и периоду

API читает пакет из `--model-output-dir`. GPU для работы сервера не нужна:
прогнозы на указанный горизонт уже рассчитаны финальной моделью. Запускайте в
отдельном терминале сервера:

```bash
python experiments/forecast_api_server.py \
  --model-dir "./artifacts/monthly_benchmark_isolated/saved_model" \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key "change-me"
```

Из Jupyter можно запустить сервер в фоне отдельным процессом:

```python
import subprocess
import sys

api_process = subprocess.Popen([
    sys.executable,
    "-u",
    "experiments/forecast_api_server.py",
    "--model-dir", "./artifacts/monthly_benchmark_isolated/saved_model",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--api-key", "change-me",
])
print("PID API:", api_process.pid)
```

Остановить сервер из того же kernel:

```python
api_process.terminate()
api_process.wait(timeout=10)
```

Проверка:

```bash
curl -H "X-API-Key: change-me" \
  "http://127.0.0.1:8000/forecast?inn=7701234567&period=2025-06"
```

Или POST-запрос:

```bash
curl -X POST "http://127.0.0.1:8000/forecast" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: change-me" \
  -d '{"inn":"7701234567","period":"2025-06"}'
```

Интерактивная документация Swagger доступна по адресу
`http://АДРЕС_СЕРВЕРА:8000/docs`. Если сервер доступен только локально, можно
убрать `--api-key` и использовать `--host 127.0.0.1`. Не открывайте
`0.0.0.0:8000` во внешнюю сеть без API-ключа и сетевого ограничения доступа.

Успешный ответ содержит прогноз зачислений, списаний, чистого потока, признак
отрицательного потока и тип прогноза — прямой или рекурсивный. Если ИНН или месяц
не входят в сохранённый пакет, сервер вернёт HTTP 404 с русским объяснением.

## Что запускать по порядку

1. Разделы 1–4 — обновление и проверка среды.
2. Раздел 5 — быстрый пробный запуск.
3. Разделы 6–7 — полное базовое обучение и проверка результата.
4. Раздел 8 — автотюнинг MLP.
5. Раздел 9 — автотюнинг gradient boosting.
6. Раздел 10 — финальная честная проверка лучших настроек.
7. Раздел 11 — отдельный дневной бизнес-demo, если он нужен.
8. Раздел 12 — API после появления каталога `saved_model`.

Месячный benchmark сохраняет реальные out-of-sample прогнозы и метрики, а
`--save-model` после проверки обучает отдельный production-checkpoint. Не
подменяйте честную оценку финальным обучением: сначала сравните модели на
тестовых месяцах, затем сохраняйте победителя.
