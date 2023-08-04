from fastapi import FastAPI, UploadFile, HTTPException
import aiofiles
import time, random, os

from .dto import *
from .pipeline import predict, load_csv_dataset, compute_ner_metrics

app = FastAPI()

@app.get("/predict/", response_model=TextPredictionResponse)
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
                       start_idx: int | None = None, end_idx: int | None = None,
                       seq_separator : str = ' ',
                       use_conll_id2label: bool = True):
    """
    Perform evaluation for the given CSV file. CSV file has to contain tokens and ner_tags columns
    which consists of sequence of word / labels (up to 9, should correspond to nocll dataset labels)
    respectively. Since model has changed id2label mapping user can set additional parameter which
    corresponds to this mapping: whether or not use standard nocll or model specifi one.
    seq_separator parameter corresponds to the separator which used to separate values in tokens and ner_tags
    sequence in the CSV file. With start_idx an end_idx parameter user can specify row range for evaluation.

    :param csv_file: CSV file itself with tokens and ner_tags columns
    :param start_idx: index of the first row (except header) which will be used for evaluation
    :param end_idx: index of the last row (except header) which will be used for evaluation
    :param seq_separator: separator which separates tokens and tags in tokens and ner_tags columns in the CSV file
    :param use_conll_id2label: whether or not id of labels in CSV file corresponds to conll dataset or to model mapping
    """

    if not csv_file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail='Invalid file format. Only CSV Files accepted.')

    # simple check whether we have required columns
    first_line = csv_file.file.readline().decode('utf-8')[:-1]
    col_names = first_line.split(',')
    if 'tokens' not in col_names or 'ner_tags' not in col_names:
        raise HTTPException(status_code=400, detail='Wrong structure of the file. CSV File has to contain tokens and ner_tags columns.')

    if start_idx and end_idx:
        if start_idx > end_idx:
            raise HTTPException(status_code=400, detail='start_idx can not be greater than end_idx')

    # save file locally, because datasets csv reader cannot read from stream
    temp_path = f'/tmp/{csv_file.filename}_{str(time.time())}_{str(random.random())}'
    csv_file.file.seek(0)
    async with aiofiles.open(temp_path, 'wb') as out_file:
        while content := await csv_file.read(1024):
            await out_file.write(content)

    # perform actual computation
    ds = load_csv_dataset(temp_path, use_conll_id2label, seq_separator, start_idx, end_idx)
    if (len(ds) == 0):
        raise HTTPException(status_code=400, detail='Evaluation range is empty. Please check start and end indices')

    metrics = compute_ner_metrics(ds)

    # convert metric result to EvaluationResponse
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
        elif isinstance(v, dict):
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
