from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from src.config import *
from src.preprocess import clean_text
import re


tokenizer = AutoTokenizer.from_pretrained(BACKTRANSLATION_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(BACKTRANSLATION_MODEL_NAME)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)

def split_sentences(text):
    return re.split(r'(?<=[.!?])\s+',text.strip())

def process_text(text, mode=None):
    text = clean_text(text)
    sentences = split_sentences(text)

    processed = []

    for s in sentences:
        if len(s.strip()) < 5:
            processed.append(s)
            continue

        if mode == 'en':
            s = backtranslate_en(s)
        elif mode == 'ko':
            s = backtranslate_ko(s)

        processed.append(s)

    return ' '.join(processed)


def translate(text, src_lang, tgt_lang):
    tokenizer.src_lang = src_lang


    inputs = tokenizer(text, return_tensors='pt').to(device)

    translated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        num_beams=5,
        do_sample=False,
        max_new_tokens=128,
        max_length=None
    )

    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def backtranslate_en(text):
    en = translate(text, 'tgl_Latn', 'eng_Latn')
    return translate(en, 'eng_Latn', 'tgl_Latn')

def backtranslate_ko(text):
    ko = translate(text, 'tgl_Latn', 'kor_Hang')
    return translate(ko, 'kor_Hang', 'tgl_Latn')

def augment_backtranslation(dataset, mode=None, column=TEXT_COLUMN):

    def transform(batch):
        texts = batch[column]
        new_texts = [process_text(t, mode) for t in texts]

        batch[column] = new_texts
        return batch

    return dataset.map(transform, batched=True, batch_size=4)
