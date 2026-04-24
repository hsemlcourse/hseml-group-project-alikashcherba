import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(filepath: str) -> pd.DataFrame:
    logging.info(f"Загрузка данных из {filepath}...")
    try:
        df = pd.read_csv(filepath, sep=';')
        logging.info(f"Данные успешно загружены. Размер: {df.shape}")
        return df
    except FileNotFoundError:
        logging.error(f"Файл не найден по пути: {filepath}")
        raise
    except Exception as e:
        logging.error(f"Ошибка при загрузке данных: {e}")
        raise


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Начало очистки данных...")

    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('.', '')
    logging.info(f"Колонки после приведения к snake_case: {df.columns.tolist()}")

    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    if df.shape[0] < initial_rows:
        logging.info(f"Удалено {initial_rows - df.shape[0]} дубликатов.")

    leakage_columns = ['race_time', 'path', 'fgrating']
    df.drop(columns=[col for col in leakage_columns if col in df.columns], inplace=True)
    logging.info(f"Удалены колонки, вызывающие data leakage: {leakage_columns}")

    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y', errors='coerce')

    numeric_cols = [
        'distance', 'prize_money', 'starting_position', 'jockey_weight',
        'odds'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(',', '.', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            logging.debug(f"Приведена колонка '{col}' к числовому типу.")

    if 'horse_age' in df.columns:
        df['horse_age_numeric'] = df['horse_age'].astype(str).str.extract(r'(\d+)').astype(float)
        df.drop(columns=['horse_age'], inplace=True)
        df.rename(columns={'horse_age_numeric': 'horse_age'}, inplace=True)
        logging.info("Извлечен числовой возраст лошади из колонки 'horse_age'.")

    if 'trainer_name' in df.columns:
        df.drop(columns=['trainer_name'], inplace=True)
        logging.info("Удалена колонка 'trainer_name' (используем TrainerID).")

    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            logging.info(f"Пропуски в '{col}' заполнены медианой ({median_val}).")

    for col in df.select_dtypes(include='object').columns:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            logging.info(f"Пропуски в '{col}' заполнены модой ({mode_val}).")

    initial_rows_after_leakage_removal = df.shape[0]
    df.dropna(subset=['date'], inplace=True)
    if df.shape[0] < initial_rows_after_leakage_removal:
        logging.warning(f"Удалено {initial_rows_after_leakage_removal - df.shape[0]} строк из-за отсутствующих дат.")

    logging.info(f"Очистка данных завершена. Размер: {df.shape}")
    return df


def feature_engineer(df: pd.DataFrame) -> pd.DataFrame:
    logging.info("Начало создания новых признаков...")

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['race_season'] = df['date'].apply(lambda x: x.year if x.month >= 9 else x.year - 1)
    logging.info("Добавлены временные признаки: year, month, day_of_week, day_of_year, race_season.")

    df['prize_per_distance'] = df['prize_money'] / df['distance']
    logging.info("Добавлен признак prize_per_distance.")

    df['jockey_weight_per_age'] = df['jockey_weight'] / df['horse_age']
    logging.info("Добавлен признак jockey_weight_per_age.")

    df['target'] = (df['final_place'] <= 3).astype(int)
    logging.info("Создан целевой признак 'target' (топ-3 или нет).")

    df.drop(columns=['final_place'], inplace=True)
    logging.info("Удалена колонка 'final_place'.")

    if 'jockey' in df.columns:
        df.drop(columns=['jockey'], inplace=True)
        logging.info("Удалена колонка 'jockey' (используем jockeyid).")

    logging.info("Создание новых признаков завершено.")
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.15, val_size: float = 0.15, random_state: int = 42) -> tuple:
    logging.info("Начало разделения данных на Train/Val/Test...")

    df = df.sort_values('date').reset_index(drop=True)

    n = len(df)
    test_split_idx = int(n * (1 - test_size))
    val_split_idx = int(n * (1 - test_size - val_size))

    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:test_split_idx]
    test_df = df.iloc[test_split_idx:]

    logging.info(f"Разделение данных завершено:")
    logging.info(
        f"  Train: {train_df.shape[0]} строк (с {train_df['date'].min().date()} по {train_df['date'].max().date()})")
    logging.info(
        f"  Validation: {val_df.shape[0]} строк (с {val_df['date'].min().date()} по {val_df['date'].max().date()})")
    logging.info(
        f"  Test: {test_df.shape[0]} строк (с {test_df['date'].min().date()} по {test_df['date'].max().date()})")

    train_df = train_df.drop(columns=['date'])
    val_df = val_df.drop(columns=['date'])
    test_df = test_df.drop(columns=['date'])
    logging.info("Колонка 'date' удалена из всех выборок.")

    return train_df, val_df, test_df


def preprocess_pipeline(df: pd.DataFrame):
    logging.info("Создание пайплайна предобработки...")

    features = [col for col in df.columns if col not in ['target', 'horseid', 'jockeyid', 'trainerid']]

    numeric_features = df[features].select_dtypes(include=np.number).columns.tolist()
    categorical_features = df[features].select_dtypes(include='object').columns.tolist()

    ohe_categorical_features = [col for col in categorical_features if col not in ['horseid', 'jockeyid', 'trainerid']]

    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, ohe_categorical_features)
        ],
        remainder='passthrough'
    )

    logging.info(f"Числовые признаки для StandardScaler: {numeric_features}")
    logging.info(f"Категориальные признаки для OneHotEncoder: {ohe_categorical_features}")
    logging.info("Пайплайн предобработки создан.")

    return preprocessor
