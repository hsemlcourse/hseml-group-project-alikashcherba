__[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/kOqwghv0)
# ML Project — Предсказание топ-3 лошади на скачках

**Студент:** Щерба Алика Алексеевна

**Группа:** БИВ235


## Оглавление

1. [Описание задачи](#описание-задачи)
2. [Структура репозитория](#структура-репозитория)
3. [Запуски](#быстрый-старт)
4. [Данные](#данные)
5. [Результаты](#результаты)
7. [Отчёт](#отчёт)


## Описание задачи

**Задача:** Бинарная классификация — предсказать, попадёт ли лошадь в топ-3 (целевая переменная `target = 1`) или нет (`target = 0`).

**Датасет:** Исторические данные о скачках с 2017 по 2020 год из Гонконга (ипподромы Sha Tin и Happy Valley).
Скачать датасет с Kaggle: https://www.kaggle.com/datasets/bogdandoicin/horse-racing-results-2017-2020

**Целевая метрика:**пше
- **ROC-AUC** — основная метрика, устойчива к дисбалансу классов (75% / 25%)
- **F1-score** — для баланса precision и recall
- **Precision / Recall** — для анализа ошибок модели


## Структура репозитория
Опишите структуру проекта, сохранив при этом верхнеуровневые папки. Можно добавить новые при необходимости.
```
.
├── .github
│   ├── workflows 
│       ├── lint.yaml 
│       └── ci.yml
├── data
│   ├── processed               # Очищенные и обработанные данные
│   └── raw                     # Исходные файлы
├── models                      # Сохранённые модели 
├── notebooks
│   ├── 01_eda.ipynb            # EDA
│   ├── 02_baseline.ipynb       # Baseline-модель
│   └── 03_experiments.ipynb    # Эксперименты и ablation study
├── presentation                # Презентация для защиты
├── report
│   ├── images                  # Изображения для отчёта
│   └── report.md               # Финальный отчёт
├── src
│   ├── preprocessing.py        # Предобработка данных
│   └── modeling.py             # Обучение и оценка моделей
├── tests
│   └── test.py                 # Тесты пайплайна
├── requirements.txt
└── README.md
```

## Запуск

Этот блок замените способом запуска вашего сервиса.
```bash
# 1. Клонировать репозиторий
git clone <url>
cd <repo-name>

# 2. Создать виртуальное окружение
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# 3. Установить зависимости
pip install -r requirements.txt
```
# 4. Запустить Jupyter Notebook
jupyter notebook

## Данные
- `data/raw/horse_racing_data.csv` — исходные файлы
- `data/processed/` — предобработанные данные


## Результаты

| Модель                   | ROC-AUC | F1    | Примечание                                 |
|--------------------------|--------:|------:|--------------------------------------------|
| Gradient Boosting (best) | 0.784   | 0.413 | Лучший баланс Precision/Recall             |
| Logistic Regression      | 0.768   | 0.513 | Высокий recall (0.835), низкая precision (0.370) |
| Random Forest            | 0.765   | 0.134 |  Плохо находит топ-3                       |
| XGBoost                  | 0.781   | 0.391 | Чуть хуже Gradient Boosting                |
| KNN                      | 0.565   | 0.238 | Слабое качество                            |

## Отчёт

Финальный отчёт: [`report/report.md`](report/report.md)__
