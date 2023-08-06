import torch

class Config:
    MODEL = 'dslim/bert-base-NER'
    WEIGHT_DIR = 'weights/'
    DEVICE = 'cpu'
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu' # I haven't test cuda because my GPU is outdated


config = Config()
