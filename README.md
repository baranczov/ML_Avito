# ML_Avito

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Classification-green)
![Status](https://img.shields.io/badge/Status-Completed-success)

# Структура кода и пояснения к нему

## БЛОК 1: Подготовка и генерация фич для train

```python
# Этапы:
# 0-2: Быстрый подсчет строк, price clip (99.9-й перцентиль), 5 фолдов по query_id
# IDF: Streaming подсчет document frequency (PyArrow) для query+title/desc
# SVD: Обучение TruncatedSVD(128) на сэмпле 300k строк  
# FEAT: Полный train → engineered фичи + TF-IDF + SVD → train_featurized_parts/*.parquet
```

**Результат**: 256 SVD-фичи (A=query+title, B=query+desc) + price_log + matches + conv

## БЛОК 2: Генерация фич для test

```python
# Идентичный пайплайн: загружает train-модели (hv, idf, svd, price_clip)
# test → те же фичи → test_featurized_parts/*.parquet
```

**Ключевое**: Гарантия одинаковых преобразований train/test

## БЛОК 3: Обучение CatBoost Ranker

```python
# val=fold_0, train=folds_1+2 (40% данных)
# Pool с group_id=query_id, target=item_contact
# CatBoostRanker(YetiRank, NDCG@10, GPU) → catboost_ranker_40pct.cbm
```

**Особенности**: Стабильная сортировка mergesort, use_best_model=True

## БЛОК 4: Предсказание и submission

```python
# test_parts → predict → sort by (query_id, score desc) → solution.csv
```

**Результат**: Готовый submission для ranking-задачи (query_id, item_id)

***

# Логика выбора методов

## Текстовые фичи: HashingVectorizer + TF-IDF + SVD

| Метод | Почему выбран | Альтернатива | Проблема альтернативы |
|-------|---------------|--------------|----------------------|
| **HashingVectorizer(2^18)** | Фиксированный размер, без словаря, streaming-safe | TfidfVectorizer | OOM на >10M строк, словарь ~1GB+ |
| **Ручной TF-IDF** | log(1+tf)*smooth_idf*L2, батчи PyArrow | sklearn.TfidfTransformer | Не streaming-safe |
| **TruncatedSVD(128)** | Сжатие 256k→128 dense фич | Полные 256k фич | Overfit, RAM |

**textA=query+title** (точность), **textB=query+desc** (контекст)

## Engineered фичи (top signals)

```python
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

**Почему не pandas**: `pd.read_parquet()` упадет с OOM

**Бенчмарк**: Top-4 Kaggle Avito использовали hashing+SVD+GBDT [ссылка на writeup]

**Итог**: Оптимальный пайплайн для Kaggle-scale RecSys (миллионы строк, sparse текст, ranking)
