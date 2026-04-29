from datasets import load_dataset
from transformers import DistilBertTokenizer
from src.config import *

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


def split_dataset(dataset):
    train_test = dataset['train'].train_test_split(test_size=TEST_SIZE, seed=SEED)

    train_valid = train_test['train'].train_test_split(test_size=VALIDATION_SIZE, seed=SEED)

    return (
        train_valid['train'],
        train_valid['test'],
        train_test['test']
    )


def tokenize(batch):
    return tokenizer(
        batch[TEXT_COLUMN],
        truncation=TRUNCATION,
        padding=PADDING,
        max_length=MAX_LENGTH
    )


def prepare_datasets():
    dataset = load_dataset('renshhhh/fake_news_filipino_parquet')

    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    return train_dataset, val_dataset, test_dataset
