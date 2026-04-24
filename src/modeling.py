from xgboost import XGBClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, classification_report
import joblib
import logging
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RANDOM_STATE = 42


def get_models():
    return {
        "logreg": LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
        "rf": RandomForestClassifier(random_state=42, n_estimators=100),
        "gb": GradientBoostingClassifier(random_state=42, n_estimators=100, learning_rate=0.1),
        "knn": KNeighborsClassifier(),
        "xgb": XGBClassifier(random_state=42, eval_metric='logloss')
    }


def get_param_grids():
    return {
        "logreg": {
            "classifier__C": [0.1, 1, 10]
        },
        "rf": {
            "classifier__n_estimators": [100],
            "classifier__max_depth": [5, 10, None]
        },
        "gb": {
            "classifier__n_estimators": [100],
            "classifier__learning_rate": [0.05, 0.1]
        },
        "knn": {
            "classifier__n_neighbors": [5, 10, 15]
        },
        "xgb": {
            "classifier__n_estimators": [100],
            "classifier__learning_rate": [0.05, 0.1],
            "classifier__max_depth": [3, 5]
        }
    }


def run_experiments(X_train, y_train, X_val, y_val, preprocessor):
    models = get_models()
    param_grids = get_param_grids()

    results = []

    for name, model in models.items():
        print(f"\n=== Model: {name} ===")

        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        grid = list(ParameterGrid(param_grids[name]))

        for params in grid:
            pipeline.set_params(**params)
            pipeline.fit(X_train, y_train)

            metrics = evaluate_model(pipeline, X_val, y_val, dataset_name=f"{name}")

            results.append({
                "model": name,
                "pipeline": pipeline,
                **params,
                **metrics
            })

    return pd.DataFrame(results)


def train_model(X_train: pd.DataFrame, y_train: pd.Series, preprocessor: Pipeline, model_params: dict = None,
                use_smote: bool = False):
    logging.info("Начало обучения модели...")

    if model_params is None:
        model_params = {'solver': 'liblinear', 'random_state': 42, 'class_weight': 'balanced'}

    base_model = LogisticRegression(**model_params)

    if use_smote:
        logging.info("Применение SMOTE для балансировки классов.")
        X_train_transformed = preprocessor.fit_transform(X_train)
        smote = SMOTE(random_state=RANDOM_STATE)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)
        model = base_model.fit(X_train_resampled, y_train_resampled)
    else:
        full_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                        ('classifier', base_model)])
        model = full_pipeline.fit(X_train, y_train)

    logging.info("Обучение модели завершено.")
    return model


def evaluate_model(model: Pipeline, X: pd.DataFrame, y_true: pd.Series, dataset_name: str = "Validation"):
    logging.info(f"Начало оценки модели на {dataset_name} наборе данных...")

    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    roc_auc = roc_auc_score(y_true, y_pred_proba)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    logging.info(f"Метрики на {dataset_name} наборе данных:")
    logging.info(f"  ROC AUC: {roc_auc:.4f}")
    logging.info(f"  F1-Score: {f1:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall: {recall:.4f}")
    logging.info("\n" + classification_report(y_true, y_pred))

    return {'roc_auc': roc_auc, 'f1_score': f1, 'precision': precision, 'recall': recall}


def save_model(model: Pipeline, filepath: str):
    logging.info(f"Сохранение модели в {filepath}...")
    try:
        joblib.dump(model, filepath)
        logging.info("Модель успешно сохранена.")
    except Exception as e:
        logging.error(f"Ошибка при сохранении модели: {e}")
        raise


def load_model(filepath: str):
    logging.info(f"Загрузка модели из {filepath}...")
    try:
        model = joblib.load(filepath)
        logging.info("Модель успешно загружена.")
        return model
    except FileNotFoundError:
        logging.error(f"Файл модели не найден по пути: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке модели: {e}")
        raise
