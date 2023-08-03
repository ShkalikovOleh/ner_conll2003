from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForTokenClassification

tokenizer = AutoTokenizer.from_pretrained('dslim/bert-base-NER')
model = AutoModelForTokenClassification.from_pretrained('dslim/bert-base-NER')

tokenizer.save_pretrained('weights/')
model.save_pretrained('weights/')
