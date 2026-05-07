from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from contextlib import asynccontextmanager
from datetime import date
from pydantic import BaseModel, Field, ConfigDict

from app.model import predictor


class HorseFeatures(BaseModel):
    race_date: date = Field(..., description="Дата скачек (ГГГГ-ММ-ДД)")
    track: str
    race_number: int
    distance: int
    surface: str
    prize_money: int
    starting_position: int
    jockey_weight: int
    country: str
    trainername: str = Field(..., alias="trainer_name")  # поддерживаем оба варианта
    odds: float
    racetype: str = Field(..., alias="race_type")
    horseid: int = Field(..., alias="horse_id")
    jockeyid: int = Field(..., alias="jockey_id")
    trainerid: int = Field(..., alias="trainer_id")
    horse_age: float

    model_config = ConfigDict(
        populate_by_name=True,  # позволяет использовать и alias, и оригинальное имя
        json_schema_extra={
            "example": {
                "race_date": "2024-06-15",
                "track": "Sha Tin",
                "race_number": 10,
                "distance": 1400,
                "surface": "Gress",
                "prize_money": 1310000,
                "starting_position": 6,
                "jockey_weight": 52,
                "country": "Sverige",
                "trainer_name": "CH Yip",
                "odds": 22.0,
                "race_type": "Handicap",
                "horse_id": 1736,
                "jockey_id": 8656,
                "trainer_id": 6687,
                "horse_age": 7.0
            }
        }
    )


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    confidence: str
    message: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 Starting Horse Racing Predictor API...")
    predictor.load_model()
    yield
    print("👋 Shutting down...")


app = FastAPI(
    title="Horse Racing Predictor API",
    description="API для предсказания попадания лошади в топ-3 на скачках",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "service": "Horse Racing Predictor",
        "status": "running" if predictor.is_loaded else "no_model",
        "endpoints": {
            "health": "/health",
            "predict": "/predict (POST)",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if predictor.is_loaded else "degraded",
        "model_loaded": predictor.is_loaded
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(features: HorseFeatures):
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Преобразуем в словарь, используя имена полей (не alias)
        features_dict = features.model_dump(by_alias=False)
        race_date = features_dict.pop('race_date')

        result = predictor.predict(features_dict, race_date)
        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)