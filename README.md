# ML_Avito

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

## Структура кода и пояснения к нему

### БЛОК 1: Подготовка и генерация фич для train

Этот блок выполняет предварительную подготовку данных и создает векторизацию текстовых фичей с использованием HashingVectorizer + TF-IDF + SVD. Он работает в 4 этапа:

- Быстрый подсчет строк, вычисление глобального клиппинга цены (99.9-й перцентиль), создание 5 фолдов по query_id .
- Считает document frequency для двух текстовых комбинаций (query_text + title, query_text + description) через streaming PyArrow, сохраняет HashingVectorizer и IDF-векторы .
- Обучает TruncatedSVD (128 компонент) на сэмпле 300k строк с TF-IDF преобразованием .
- Применяет весь пайплайн к полному train в батчах (30k строк), добавляет engineered фичи (price_log, is_loc_match, conv_missing), SVD-компоненты, сохраняет в частичные parquet .

**Результат**: train_featurized_parts/*.parquet (без raw текста, с 256 SVD-фичами).

### БЛОК 2: Генерация фич для test

Идентичный пайплайн для test данных, используя модели/IDF из train. Загружает предобученные трансформеры и price_clip параметры.

- Обрабатывает test в батчах (30k строк), применяет те же фичи + TF-IDF + SVD.
- Сохраняет в test_featurized_parts/*.parquet с теми же колонками, что и train .

**Ключевое**: Гарантирует одинаковые преобразования train/test для консистентности.

### БЛОК 3: Обучение CatBoost Ranker

Собирает train/validation из фолдов (val=fold_0, train=folds_1+2 = 40% данных), обучает ranking-модель.

- Загружает частичные parquet, фильтрует по query_id фолдам
- Создает CatBoost Pool с group_id=query_id, target=item_contact  
- Обучает CatBoostRanker (YetiRank loss, NDCG@10 метрика, GPU, 3000 итераций)
- Cохраняет модель catboost_ranker_40pct.cbm

### БЛОК 4: Предсказание и submission

Загружает обученную модель, применяет к test частям, генерирует solution.csv.

- Для каждой test части: predict → добавляет score.
- Конкатенирует, сортирует по query_id + score (descending).
- Сохраняет топ-N по query_id (query_id, item_id) в CSV.

## Логика выбора методов

## Текстовые фичи: HashingVectorizer + TF-IDF + SVD

| Метод | Почему выбран | Альтернатива | Проблема альтернативы |
|-------|---------------|--------------|----------------------|
| **HashingVectorizer(2^18)** | Фиксированный размер, без словаря, streaming-safe | TfidfVectorizer | OOM на >10M строк, словарь ~1GB+ |
| **Ручной TF-IDF** | log(1+tf)*smooth_idf*L2, батчи PyArrow | sklearn.TfidfTransformer | Не streaming-safe |
| **TruncatedSVD(128)** | Сжатие 256k→128 dense фич | Полные 256k фич | Overfit, RAM |

**textA=query+title** (точность), **textB=query+desc** (контекст)

## Engineered фичи (top signals)

```
price_clip/log1p      # skewed цены Avito
is_loc_match          # >80% веса в RecSys  
is_cat_match          # точное совпадение категорий
conv_missing/conv_val # обработка -1 в кликах
```

## Модель: CatBoostRanker(YetiRank)

```
Задача: per-query ranking (NDCG@10)
Почему ranking ≠ классификация:
✓ group_id=query_id группирует объявления
✓ YetiRank фокусируется на топ-10
✓ Фолды по query_id → нет data leak
```

**Параметры**: depth=8, lr=0.05, 3000 iter, GPU — баланс скорости/качества

## Streaming архитектура

```
10-50GB parquet → PyArrow.dataset.scanner(batch_size=30k)
→ process → gc.collect() → part_XXX.parquet
```
