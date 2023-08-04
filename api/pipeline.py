from transformers import pipeline, DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification
from datasets import load_dataset, Sequence, ClassLabel
from evaluate import load
from torch.utils.data import DataLoader
import numpy as np

# should be global because have to act as a static variable (don't perform model creation per func call)
tokenizer = AutoTokenizer.from_pretrained('weights/', local_files_only=True, use_fast=True)
model = AutoModelForTokenClassification.from_pretrained('weights/', local_files_only=True)
pipe = pipeline('token-classification', model=model, tokenizer=tokenizer)

def predict(text):
    return pipe(text, aggregation_strategy='average')

def tokenize_and_align_labels(data, is_word_splitted, id_mapping):
    """
    Tokenize inputs and align ner_tags. It's mandatory, since
    tokenizer can split any word into several tokens
    """

    tokenized_inputs = tokenizer(data['tokens'], truncation=True,
                                 is_split_into_words=is_word_splitted)

    labels = []
    for i, label in enumerate(data['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(id_mapping[label[word_idx]])
            else:
                l = label[word_idx]
                if l % 2 == 1:
                    l += 1
                label_ids.append(id_mapping[l])
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs

def load_csv_dataset(path, is_word_splitted, use_nocll_id2label, seq_separator=' '):
    ds = load_dataset('csv', data_files={'test': path})['test']

    def str_to_seq(ex):
        str_tokens = ex['tokens'][1:-1].split(seq_separator)
        str_tags = ex['ner_tags'][1:-1].split(seq_separator)
        tokens = []
        tags = []
        for token, tag in zip(str_tokens, str_tags):
            tokens.append(token.replace('\'', '').rstrip())
            tags.append(int(tag))
        return {"tokens": tokens, "ner_tags": tags}
    ds = ds.map(str_to_seq)

    if use_nocll_id2label: # Original CoNLL has different id2label mapping from our pipeline !!!
        id_mapping = { 0:0, 1:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:1, 8:2 }
    else:
        id_mapping = {i:i for i in range(9)}

    func = lambda x: tokenize_and_align_labels(x, is_word_splitted, id_mapping)
    tokenized_ds = ds.map(func, batched=True, remove_columns=ds.column_names)
    tokenized_ds = tokenized_ds.cast_column('labels', Sequence(ClassLabel(9)))
    tokenized_ds.with_format("torch")

    return tokenized_ds

def compute_ner_metrics(dataset):
    seqeval = load('seqeval')

    collator = DataCollatorForTokenClassification(tokenizer, label_pad_token_id=-100)
    dl = DataLoader(dataset, batch_size=8, collate_fn=collator)

    for data in dl:
        logits = model(input_ids=data['input_ids'], attention_mask=data['attention_mask']).logits
        labels = data['labels'].detach().numpy()
        predictions = np.argmax(logits.detach().numpy(), axis=2)

        true_labels = [[model.config.id2label[l] for l in label if l != -100] for label in labels]
        true_predictions = [
            [model.config.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        seqeval.add_batch(predictions=true_predictions, references=true_labels)

    eval_results = seqeval.compute()

    return eval_results