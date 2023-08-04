from fastapi import FastAPI, UploadFile, HTTPException
import aiofiles
import time, random, os

from .dto import *
from .pipeline import predict, load_csv_dataset, compute_ner_metrics

app = FastAPI()

# @app.get("/predict/", response_model=TextPredictionResponse)
@app.get("/predict/")
async def predict_str(request: TextPredictionRequest):
    """
    Perform prediction for the simple string. Returns words with labels and
    start and end character position in the input by combining tokens prediction.
    The result token's label is equal to the label of the first predicted token.
    """

    predictions = predict(request.text)

    pred_responses = []
    for pred in predictions:
        text_pred = TextPrediction(label=pred['entity_group'],
                                   score=pred['score'],
                                   word=pred['word'],
                                   start=pred['start'],
                                   end=pred['end'])
        pred_responses.append(text_pred)

    return TextPredictionResponse(predictions = pred_responses)

@app.post("/evaluate/", response_model=EvaluationResponse)
async def evaluate_csv(csv_file: UploadFile,
                       is_word_splitted: bool | None = True,
                       use_nocll_id2label: bool | None = True):
    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid file format. Only CSV Files accepted.')

    first_line = csv_file.file.readline().decode('utf-8')[:-1]
    col_names = first_line.split(',')
    if 'tokens' not in col_names or 'ner_tags' not in col_names:
        raise HTTPException(status_code=400, detail='Wrong structure of the file. CSV File has to contain tokens and ner_tags columns.')

    temp_path = f'/tmp/{csv_file.filename}_{str(time.time())}_{str(random.random())}'
    csv_file.file.seek(0)
    async with aiofiles.open(temp_path, 'wb') as out_file:
        while content := await csv_file.read(1024):
            await out_file.write(content)

    ds = load_csv_dataset(temp_path, is_word_splitted, use_nocll_id2label)
    metrics = compute_ner_metrics(ds)

    f1_dict = {}
    recall_dict = {}
    precision_dict = {}
    number_dict = {}

    for k, v in metrics.items():
        if (k.startswith('overall')):
            metric_name = k.split('_')[1]
            if metric_name == 'precision':
                precision_dict['overall'] = v
            elif metric_name == 'f1':
                f1_dict['overall'] = v
            elif metric_name == 'recall':
                recall_dict['overall'] = v
        else:
            for metric_name, val in v.items():
                if metric_name == 'precision':
                    precision_dict[k] = val
                elif metric_name == 'f1':
                    f1_dict[k] = val
                elif metric_name == 'recall':
                    recall_dict[k] = val
                elif metric_name == 'number':
                    number_dict[k] = val

    os.remove(temp_path)

    return EvaluationResponse(f1=f1_dict, recall=recall_dict,
                              precision=precision_dict, numbers=number_dict,
                              accuracy=metrics['overall_accuracy'])
