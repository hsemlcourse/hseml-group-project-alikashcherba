import pandas as pd
import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing import clean_data, feature_engineer
from src.modeling import load_model


class HorseRacingPredictor:

    def __init__(self, model_path: str = "models/best_model.joblib"):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False

    def load_model(self):
        try:
            self.model = load_model(self.model_path)
            self.is_loaded = True
            print(f"✅ Model loaded successfully from {self.model_path}")
            return True
        except Exception as e:
            print(f" Error loading model: {e}")
            self.is_loaded = False
            return False

    def preprocess_for_api(self, df: pd.DataFrame, race_date) -> pd.DataFrame:
        """Предобработка для API запроса"""
        df = df.copy()

        df['date'] = pd.to_datetime(race_date)

        df['race_time'] = 0
        df['path'] = 0
        df['fgrating'] = 0

        df['final_place'] = 1

        df = clean_data(df)

        df = feature_engineer(df)

        cols_to_drop = ['target', 'date', 'race_time', 'path', 'fgrating', 'final_place']
        df.drop(columns=[c for c in cols_to_drop if c in df.columns], inplace=True)

        return df

    def predict(self, features_dict: dict, race_date) -> dict:
        if not self.is_loaded:
            raise ValueError("Model not loaded")

        df = pd.DataFrame([features_dict])
        df_processed = self.preprocess_for_api(df, race_date)

        prediction = int(self.model.predict(df_processed)[0])
        probability = float(self.model.predict_proba(df_processed)[0][1])

        if probability > 0.7:
            confidence = "high"
        elif probability > 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "prediction": prediction,
            "probability": probability,
            "confidence": confidence,
            "message": self._get_message(prediction, probability)
        }

    def _get_message(self, prediction, probability):
        if prediction == 1:
            if probability > 0.7:
                return "Лошадь с высокой вероятностью попадёт в топ-3!"
            elif probability > 0.5:
                return "Лошадь может попасть в топ-3, но есть шанс ошибки"
            else:
                return "Слабая уверенность в попадании в топ-3"
        else:
            if probability < 0.3:
                return "Лошадь с высокой вероятностью НЕ попадёт в топ-3"
            elif probability < 0.5:
                return "Лошадь скорее НЕ попадёт в топ-3"
            else:
                return "Предсказание: вне топ-3, но не уверены"


predictor = HorseRacingPredictor()