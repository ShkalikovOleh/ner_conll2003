from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from .dbo import TextPrediction

# should be global because have to acts as a static variable (don't perform model creation per func call)
tokenizer = AutoTokenizer.from_pretrained('weights/', local_files_only=True, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained('weights/', local_files_only=True)
pipe = pipeline('ner', model=model, tokenizer=tokenizer)

def predict(text: str):
    return pipe(text)

def token_prediction_to_words(predictions):
    text_predictions = []
    for pred in predictions:
        end = pred["start"] + len(pred["word"].replace("##", ""))

        if pred["word"].startswith("##"):
            text_predictions[-1].end = end
            text_predictions[-1].word += pred["word"].replace("##", "")
        else:
            text_predictions.append(TextPrediction(
                    label = pred["entity"][2:], # first two symbols means beggining or inside
                    word = pred["word"],
                    start = pred["start"],
                    end = end,
                    score = pred["score"]))

    return