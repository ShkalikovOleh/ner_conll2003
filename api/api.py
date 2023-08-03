import torch
from fastapi import FastAPI

from .dbo import TextPredictionRequest, TextPredictionResponse
from .pipeline import predict, token_prediction_to_words

app = FastAPI()

@app.get("/predict/", response_model=TextPredictionResponse)
def predict_str(request: TextPredictionRequest):
    """
    Perform prediction for the simple string. Returns words with labels and
    start and end character position in the input by combining tokens prediction.
    The result token's label is equal to the label of the first predicted token.
    """

    predictions = predict(request.text)
    text_predictions = token_prediction_to_words(predictions)
    return TextPredictionResponse(predictions = text_predictions)
