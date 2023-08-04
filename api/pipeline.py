from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Sequence, Value, ClassLabel
from evaluate import load, evaluator

# should be global because have to act as a static variable (don't perform model creation per func call)
tokenizer = AutoTokenizer.from_pretrained('weights/', local_files_only=True, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained('weights/', local_files_only=True)
pipe = pipeline('token-classification', model=model, tokenizer=tokenizer)

def predict(text):
    # average is important since we would like to have the same prediction for the whole word
    return pipe(text, aggregation_strategy='average')

def load_csv_dataset(path, use_conll_id2label, seq_separator, start_idx, end_idx):
    ds = load_dataset('csv', data_files={'test': path})['test'] # load_datasets always want to create splits

    # take only the part of all rows if indices are specified
    first_row = min(start_idx, len(ds)) if start_idx else 0
    last_row = min(end_idx, len(ds)) if end_idx else len(ds)
    ds = ds.select(range(first_row, last_row))

    if use_conll_id2label: # Original CoNLL has different id2label mapping from our pipeline !!!
        id_mapping = { 0:0, 1:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:1, 8:2 }
    else:
        id_mapping = {i:i for i in range(9)}

    # csv reader wrongly considers sequences as a string, let's covert them to lists
    def str_to_seq(ex):
        str_tokens = ex['tokens'][1:-1].split(seq_separator)
        str_tags = ex['ner_tags'][1:-1].split(seq_separator)
        tokens = []
        tags = []
        for token, tag in zip(str_tokens, str_tags):
            tokens.append(token.replace('\'', '').rstrip())
            tags.append(id_mapping[int(tag)])
        return {"tokens": tokens, "ner_tags": tags}
    ds = ds.map(str_to_seq)

    # evaluator requires specific types of column, so we have to cast
    ds = ds.cast_column('tokens', Sequence(Value('string')))
    class_names = list(model.config.id2label.values())
    ds = ds.cast_column('ner_tags', Sequence(ClassLabel(names=class_names)))

    return ds

def compute_ner_metrics(dataset):
    seqeval = load('seqeval')
    metric_eval = evaluator('token-classification')

    eval_results = metric_eval.compute(model_or_pipeline=pipe, data=dataset, metric=seqeval)

    return eval_results