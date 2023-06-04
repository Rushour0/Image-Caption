import pandas as pd
from numpy.random import RandomState
from tqdm import tqdm
from transformers import AutoTokenizer


def clean():
    """
    _summary_:
    Cleans the data and saves it in happy_moments.csv
    """
    happy_moments = pd.read_csv("data/cleaned_hm.csv")

    happy_moments = happy_moments.dropna(axis=0, subset=['cleaned_hm'])

    happy_moments = happy_moments.drop_duplicates(subset=['cleaned_hm'])

    happy_moments = happy_moments[['cleaned_hm', 'predicted_category']]

    happy_moments = happy_moments.rename(
        columns={'cleaned_hm': 'text', 'predicted_category': 'label'})

    happy_moments.to_csv("data/happy_moments.csv", index=False)

    return happy_moments


def split():
    """
    _summary_:
    Splits the data into train and test data and saves them in train.csv and test.csv
    """
    df = pd.read_csv('data/happy_moments.csv')
    rng = RandomState()

    train = df.sample(frac=0.7, random_state=rng)
    test = df.loc[~df.index.isin(train.index)]

    train.to_csv('data/train/train.csv', index=False)
    test.to_csv('data/test/test.csv', index=False)


def tokenize_save():
    """
    _summary_:
    Tokenizes the train and test data and saves them in train_mod.txt and test_mod.txt
    """

    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    train = pd.read_csv('data/train/train.csv')
    test = pd.read_csv('data/test/test.csv')

    train_m = ""

    for line in tqdm(train['text'].tolist(), desc="Train Data Tokenizing"):
        train_m += (tokenizer.special_tokens_map['bos_token'] +
                    line.rstrip()+tokenizer.special_tokens_map['eos_token'])

    test_m = ""

    for line in tqdm(test['text'].tolist(), desc="Test Data Tokenizing"):
        test_m += (tokenizer.special_tokens_map['bos_token'] +
                   line.rstrip()+tokenizer.special_tokens_map['eos_token'])
        
    with open('data/train/train_mod.txt', "w", encoding='utf-8') as f:
        f.write(train_m)
    
    with open('data/test/test_mod.txt', "w", encoding='utf-8') as f:
        f.write(test_m)

