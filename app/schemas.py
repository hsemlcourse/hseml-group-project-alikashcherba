from datetime import date
from pydantic import BaseModel, Field


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
    trainer_name: str
    odds: float
    race_type: str
    horse_id: int
    jockey_id: int
    trainer_id: int
    horse_age: float

    class Config:
        json_schema_extra = {
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