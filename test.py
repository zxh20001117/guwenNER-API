import torch
import pandas as pd
from Utils.config import ROOT_PATH, MODEL_PATH
from modules.NER import NER
from modules.model import guwenBERT_LSTM_CRF

data = pd.read_excel('data/1000sample.xlsx')
sentences = data['sentence'].tolist()

ner = NER()

i, batch = 0, 32
res = {
        "tagSentences": [],
        "offices": [],
        "times": [],
        "names": [],
        "origins": [],
        "pers": [],
        "posthumouss": [],
        "titles": [],
        "nicknames": []
    }

while i*batch<len(sentences):
    bios, ans = ner.get_bio(sentences[i*batch:(i+1)*batch]) \
        if (i+1)*batch<len(sentences) \
        else ner.get_bio(sentences[i*batch:])
    i+=1
    for key in res.keys():
        res[key]+=ans[key]
    print(f"batch {i} finished!")
for key in res.keys():
    data[key] = res[key]
data.to_excel('data/1000sample.xlsx')



