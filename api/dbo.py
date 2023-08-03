from pydantic import BaseModel

__all__ = ['TextRequest']

class TextPredictionRequest(BaseModel):
    text: str

class TextPrediction(BaseModel):
    label: str
    word: str
    start: int
    end: int
    score: float

class TextPredictionResponse(BaseModel):
    predictions: list[TextPrediction]
