from transformers import AutoTokenizer, AutoModelForTokenClassification

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from config import config

tokenizer = AutoTokenizer.from_pretrained(config.MODEL)
model = AutoModelForTokenClassification.from_pretrained(config.MODEL)

tokenizer.save_pretrained(config.WEIGHT_DIR)
model.save_pretrained(config.WEIGHT_DIR)
