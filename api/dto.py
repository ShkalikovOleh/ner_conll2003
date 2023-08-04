from pydantic import BaseModel

__all__ = ['TextPredictionRequest', 'TextPrediction', 'TextPredictionResponse', 'EvaluationResponse']

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

class EvaluationResponse(BaseModel):
    f1: dict[str, float]
    recall: dict[str, float]
    precision: dict[str, float]
    numbers: dict[str, int]
    accuracy: float
