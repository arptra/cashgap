# CashGap Lab

Локальный model-first Linux-монорепозиторий для установки, запуска, обучения и честного сравнения моделей денежных потоков и cash-gap риска через FastAPI + React.

Проект работает без Docker, GPU, облака и внешнего inference API. Pretrained-веса и competition notebooks скачиваются только по явной команде пользователя; `/models` является стартовой страницей, а каталог источников сохранён как вспомогательный раздел.

## Главное ограничение

`fraud`, `isFraud`, `isFlaggedFraud`, `Is Laundering`, AML- и anomaly-labels — не кассовый разрыв. CashGap Lab программно запрещает назначать их target задачи `cash_gap_classification`.

Supervised-классификация разрешена только когда событие действительно означает нехватку доступной ликвидности:

- target рассчитан генератором по дневной траектории баланса и овердрафта;
- подключена отдельная согласованная корпоративная таблица target;
- источник явно содержит подтверждённое событие недостаточности ликвидности.

Поэтому внешние открытые наборы здесь прежде всего проверяют ingestion, mapping, forecasting, categorization и proxy-сценарии. Полная демонстрация cash-gap classification выполняется на собственной синтетике.

## Возможности

- model registry в `config/models.yaml`: competition recipes, Hugging Face pretrained-модели и локальные обучаемые модели;
- безопасная загрузка только revision-pinned `safetensors`/конфигов без `trust_remote_code`, сторонних pickle/joblib и исполнения Kaggle notebook;
- zero-shot CPU inference Chronos-Bolt/Chronos-2/TimesFM batch-ами, квантильные прогнозы и управляемый локальный cache;
- собственные Shell-style CatBoost/Prophet adapters, прогнозирующие credit/debit отдельно;
- единый benchmark contract: один dataset, target, horizon, test-период и task;
- реестр источников в `config/datasets.yaml` и его копия в SQLite;
- коннекторы Kaggle Dataset, Kaggle Competition, Hugging Face, HTTP/GitHub и local file;
- фоновые jobs со статусами `queued`, `checking_access`, `downloading`, `extracting`, `profiling`, `normalizing`, `completed`, `failed`, `cancelled`;
- адаптеры PaySim, BankSim, IBM AML, Agami Indian Statements, Mindweave US, Shell Cashflow и Transaction Categorization;
- каноническая Parquet-витрина и отдельная таблица месячной ликвидности;
- DuckDB-профилирование, compatibility report и блокировка несовместимых задач в UI/API;
- синтетические дневные потоки, баланс, овердрафт, фактический cash gap и `cash_gap_next_month`;
- четыре model-first cash-gap классификатора, forecasting baselines и сохранённые adapters прежнего lab;
- временное train/validation/test-разделение без случайного перемешивания строк;
- сохранение моделей, метрик, prediction-файлов, split’ов и ошибок;
- model catalog, шестишаговый мастер запуска, эксперименты, task-specific сравнение, результаты/CSV, datasets, sources и настройки.

## Требования

- Linux;
- Python 3.11, 3.12 или 3.13;
- Node.js 18+ и npm;
- около 4 ГБ RAM для синтетики по умолчанию на 3 000 клиентов × 24 месяца.

На macOS проект также запускается для разработки, но целевая среда — Linux CPU.

## Быстрый старт

```bash
git clone https://github.com/arptra/cashgap.git
cd cashgap
python3 start.py
```

`start.py` сам создаёт `.venv`, устанавливает Python- и Node.js-зависимости и запускает FastAPI вместе с Vite. `make` не нужен. При следующих запусках достаточно снова выполнить `python3 start.py`.

Адреса по умолчанию:

- UI: <http://127.0.0.1:5173>
- Swagger: <http://127.0.0.1:8000/docs>
- health: <http://127.0.0.1:8000/api/health>

`python3 start.py` запускает Uvicorn и Vite, а по Ctrl+C завершает оба процесса. Если порт занят, скрипт выберет следующий свободный и напечатает фактический URL. Порты можно задать явно:

```bash
CASHGAP_API_PORT=8010 CASHGAP_UI_PORT=5174 python3 start.py
```

`make setup` автоматически выбирает существующий `.venv`, затем `python3.13`, `python3.12` или `python3.11`. Если совместимый Python доступен под другим именем:

```bash
PYTHON_BIN=/usr/bin/python3.11 make setup
```

На корпоративной машине, где уже установлен только Python 3.13, достаточно:

```bash
git pull
./start-corp.sh
```

Скрипт можно вызвать из любой директории по полному пути. Он сам найдёт Python, создаст `.venv`, установит зависимости и поднимет backend вместе с frontend. Если политика виртуальной машины запрещает прямой запуск файла, используйте `bash start-corp.sh`.

Для ручного запуска в двух терминалах после первой установки:

```bash
# Терминал 1 — backend
.venv/bin/python -m uvicorn app.main:app --app-dir backend --host 127.0.0.1 --port 8000

# Терминал 2 — frontend
cd frontend
npm run dev
```

## Демо целиком

```bash
make demo
```

Команда:

1. создаёт небольшой synthetic dataset;
2. строит дневной скрытый слой, месячную canonical-витрину, liquidity и target;
3. на одном forecasting holdout запускает Seasonal Naive, Shell-style CatBoost lag recipe и Chronos-Bolt Tiny на CPU;
4. на одном temporal classification split обучает LogisticRegression и CatBoost;
5. печатает реальные MAE/WAPE и PR-AUC/Recall/F1;
6. оставляет dataset, benchmarks, runs, metrics и prediction artifacts доступными в SQLite и UI.

Первый запуск Chronos скачает около 35 MB безопасных весов, если они ещё не установлены. При сетевой ошибке run получает явный `FAILED`, без случайного или baseline fallback.

Размер можно уменьшить или изменить через CLI:

```bash
./scripts/demo.sh --clients 100 --months 12 --seed 7
```

## Настройка Kaggle

Поддерживаются стандартный файл Kaggle и environment variables.

Вариант 1 — файл:

```bash
mkdir -p ~/.kaggle
cp /безопасный/путь/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

Вариант 2 — переменные текущей shell-сессии:

```bash
export KAGGLE_USERNAME='your_username'
export KAGGLE_KEY='your_key'
```

Credentials не записываются в SQLite. Kaggle competition загружается отдельным методом; если правила соревнования ещё не приняты на сайте Kaggle, job завершится понятной ошибкой.

## Настройка Hugging Face

Публичные datasets работают без токена. Для private/gated источника:

```bash
export HF_TOKEN='hf_...'
```

Tabular-наборы читаются streaming-режимом через `datasets.load_dataset`. Сложные многотабличные наборы скачиваются через `snapshot_download`; для Agami разрешены JSON/JSONL, а PDF, изображения и OCR исключены. Revision/commit hash сохраняется в метаданных источника.

Файл `.env.example` служит шпаргалкой. Скрипты намеренно не читают и не сохраняют secrets автоматически — перед запуском экспортируйте нужные переменные.

## Загрузка и нормализация через UI

1. Откройте страницу **Источники данных**.
2. Нажмите **Проверить доступ**. Для local/HTTP укажите путь или URL.
3. Проверьте лицензию и правила, затем отметьте checkbox подтверждения.
4. Нажмите **Скачать** и дождитесь завершения background job.
5. На странице **Datasets** откройте raw dataset: там видны файлы и размеры.
6. Нажмите **Нормализовать**. Для PaySim можно включить merchant `M*`; для IBM AML можно передать `selected_file`, иначе выбирается самый маленький совместимый CSV.
7. Проверьте preview, mapping source → canonical, profile, предупреждения качества и compatibility report.
8. Перейдите в **Обучение** — несовместимые задачи и модели будут недоступны.

Прямые downloads никогда не стартуют автоматически при запуске приложения.

## Реестр источников

| ID | Provider | Основное применение | Cash-gap target |
|---|---|---|---|
| `kaggle_paysim` | Kaggle dataset | ingestion, monthly aggregation, balance proxy | нет |
| `kaggle_banksim` | Kaggle dataset | debit-flow forecasting, category analysis | нет |
| `kaggle_ibm_aml` | Kaggle dataset | two-sided flow, forecasting, proxy | нет |
| `kaggle_shell_cashflow` | Kaggle competition | aggregate flow forecasting | нет |
| `hf_paysim_banks` | Hugging Face | ingestion, monthly aggregation, balance proxy | нет |
| `hf_indian_bank_statements` | Hugging Face | JSON statements, flow, balance proxy | нет |
| `hf_us_bank_transactions` | Hugging Face | one-company forecasting demo | нет |
| `hf_transaction_categorization` | Hugging Face | TF-IDF categorization | нет |

Дополнительно зарегистрированы универсальные `http_direct` и `local_file`.

## Mapping по источникам

- **PaySim:** `step` считается часами от `base_date`; обе стороны операции определяются по фактическому изменению баланса, а при неконсистентном delta используется mapping по type с флагом качества. `C*` — клиенты, `M*` выключены по умолчанию.
- **BankSim:** `step` — дни, `amount` остаётся debit; искусственные credits не создаются.
- **IBM AML:** отправитель получает debit в `Payment Currency`, получатель credit в `Receiving Currency`. Валюты агрегируются отдельно. Объединение нескольких валют без явной FX-таблицы запрещено. CSV читается Polars lazy streaming batches и сохраняется в Parquet.
- **Agami:** только structured JSON; failed-операции исключаются из оборота, но сохраняются как quality signal. Баланс ниже threshold — лишь `balance_breach_proxy`.
- **Mindweave:** динамически обнаруживаются companies/accounts/transactions/statements; `client_id` строится на уровне company. Из-за одной компании клиентская классификация запрещена.
- **Shell:** строится только `SHELL_AGGREGATE` с `inflow`, `outflow`, `net_flow`; multi-client classifier запрещён.
- **Categorization:** текст и category хранятся отдельно, не превращаются в месячный cash flow.

## Канонические таблицы

Основная месячная таблица:

```text
client_id, month, transaction_label,
debit_sum, credit_sum,
debit_nonzero_count, credit_nonzero_count,
source_dataset, source_provider, currency, data_quality_flags
```

При наличии остатков создаётся `monthly_liquidity.parquet`:

```text
client_id, month, opening_balance, closing_balance,
minimum_observed_balance, maximum_observed_balance,
balance_observations_count, balance_breach_proxy
```

Валидатор запрещает отрицательные суммы/counts, неверный формат месяца, NaN и infinity.

## Синтетический target и отсутствие leakage

Генератор создаёт дневные поступления/расходы, opening/closing balance и доступный overdraft. Событие возникает из дневной траектории, когда доступной ликвидности не хватает; затем оно сдвигается в `cash_gap_next_month`.

В модель передаются только месячные агрегаты. Дневной баланс, cash-gap amount и будущие значения не становятся признаками. Point-in-time признаки используют текущий и прошлые месяцы: лаги 1/2/3/6, rolling 3/6, тренды, волатильность и серии отрицательного net flow. Split строго временной: ранние месяцы train, следующие validation, последние test.

## Модели и метрики

Реестр `config/models.yaml` подключает:

- competition recipes: `shell_catboost_darts_3rd`, `shell_prophet_8th`;
- pretrained: `chronos_bolt_tiny`, `chronos_bolt_small`, `chronos_2`, `timesfm_2_5`;
- локальный forecasting: `seasonal_naive_local`, `shell_style_catboost_lag`;
- cash-gap classification: `logistic_cashgap`, `catboost_cashgap`, `lightgbm_cashgap`, `random_forest_cashgap`.

Prophet и `timesfm[torch]` остаются optional: страница моделей показывает `NOT_INSTALLED` и точную команду. Kaggle recipes требуют стандартную Kaggle-авторизацию и хранят notebook только как источник/attribution; production adapter — внутренний нормализованный Python-код.

Classification:

- DummyClassifier;
- LogisticRegression;
- RandomForestClassifier;
- CatBoostClassifier;
- LGBMClassifier.

Метрики: PR-AUC, ROC-AUC, Precision, Recall, F1, Brier score, Precision@Top10%, Recall@Top10%, confusion matrix и training time.

Forecasting:

- SeasonalNaive;
- AutoETS;
- AutoARIMA;
- LightGBM с lag-признаками.

Метрики: MAE, RMSE, WAPE, MASE при доступном масштабе и training time. Forecast prediction содержит `actual`, `forecast`, `lower_bound`, `upper_bound`.

Categorization: TF-IDF + LogisticRegression с accuracy и macro-F1. Это отдельная модель, не cash-gap model.

## API

Основные endpoints:

```text
GET    /api/health
GET    /api/models
GET    /api/models/{model_id}
POST   /api/models/{model_id}/check
POST   /api/models/{model_id}/install
DELETE /api/models/{model_id}/install
POST   /api/models/{model_id}/compatibility
POST   /api/models/{model_id}/run
GET    /api/model-jobs
GET    /api/model-jobs/{job_id}
POST   /api/benchmarks/start
GET    /api/benchmarks
GET    /api/benchmarks/{benchmark_id}
GET    /api/benchmarks/{benchmark_id}/comparison
GET    /api/sources
GET    /api/sources/{source_id}
POST   /api/sources/{source_id}/check-access
POST   /api/sources/{source_id}/download
GET    /api/jobs
GET    /api/jobs/{job_id}
POST   /api/jobs/{job_id}/cancel
GET    /api/datasets
GET    /api/datasets/{dataset_id}
GET    /api/datasets/{dataset_id}/preview
GET    /api/datasets/{dataset_id}/profile
GET    /api/datasets/{dataset_id}/compatibility
POST   /api/datasets/{dataset_id}/normalize
DELETE /api/datasets/{dataset_id}
POST   /api/synthetic/generate
POST   /api/experiments/start
GET    /api/experiments
GET    /api/experiments/{run_id}
GET    /api/experiments/{run_id}/metrics
GET    /api/experiments/{run_id}/predictions
GET    /api/experiments/{run_id}/predictions.csv
GET    /api/experiments/{run_id}/feature-importance
POST   /api/experiments/compare
DELETE /api/experiments/{run_id}
```

Пример synthetic generation:

```bash
curl -X POST http://127.0.0.1:8000/api/synthetic/generate \
  -H 'Content-Type: application/json' \
  -d '{"n_clients":3000,"n_months":24,"random_seed":42,"target_gap_rate":0.10,"noise_level":0.15,"overdraft_share":0.55}'
```

Ответ содержит `job_id` и `dataset_id`; прогресс читается через `/api/jobs/{job_id}`.

## Подключение корпоративного target

Минимальный согласованный формат:

```text
client_id, month, cash_gap_next_month
```

Нужно заранее формально определить событие: доступный баланс с учётом подтверждённого overdraft/credit line стал меньше нуля или обязательный платёж не мог быть исполнен из-за ликвидности. Затем добавьте отдельный local adapter, который:

1. сопоставляет корпоративные client/month с canonical-витриной;
2. вызывает `assert_valid_cash_gap_target` для выбранной target-колонки;
3. сохраняет отдельный `target.parquet`;
4. отмечает `cash_gap_target: true` только после согласования определения;
5. не включает target или будущие балансы в признаки.

Точки расширения: `backend/app/adapters/`, `backend/app/adapters/factory.py` и `config/datasets.yaml`.

## Хранение

```text
data/raw/<provider>/<source_id>/<dataset_id>/
data/normalized/<dataset_id>/
artifacts/models/<run_id>.joblib
artifacts/runs/<run_id>/
  metrics.json
  parameters.json
  split.json
  predictions.parquet
  feature_importance.csv        # classification
backend/cashgap.db
cache/
logs/
```

SQLite хранит source registry, imported datasets, background jobs, experiments, metrics, artifacts и errors. Credentials в SQLite не сохраняются.

## Проверка

```bash
make test
```

Команда запускает pytest без integration-маркеров и production build TypeScript/Vite. Обычные тесты используют только маленькие fixture-файлы и не требуют Kaggle/Hugging Face credentials или сети.

Покрываются model registry, безопасная revision-pinned установка/cache, compatibility, Shell flow identity, Chronos и TimesFM adapters, forecasting/classification benchmarks, model API, temporal split, запрет fraud/AML/balance proxy target, каждый data adapter и frontend production build.

Сетевые проверки помечаются `integration` и запускаются отдельно:

```bash
cd backend
../.venv/bin/pytest -m integration
```

## Очистка

```bash
make clean
```

Команда удаляет скачанные raw/normalized/synthetic данные, модели, runs, cache, logs, frontend build и локальную SQLite. Исходники, registry и тестовые fixtures остаются.

## Структура репозитория

```text
backend/app/
  api/ connectors/ adapters/ canonical/ ml/ jobs/ db/ services/
  models_registry/ model_plugins/
backend/tests/fixtures/
frontend/src/
  pages/ components/ api/ types/
config/
data/{raw,normalized,synthetic,uploaded}/
artifacts/{models,runs}/
scripts/{setup,dev,test,demo}.sh
Makefile
```
