from datasets import load_dataset

ds = load_dataset('conll2003')
ds['test'].to_csv('data/test.csv')
# ds['test'].save_to_disk('data/')
# ds['test'].to_parquet('data/test.parquet')