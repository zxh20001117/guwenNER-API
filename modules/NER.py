import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from Utils.config import ROOT_PATH, MODEL_PATH, BERT_PATH, DEVICE
from transformers import AutoTokenizer

root_path = ROOT_PATH
bert_path = BERT_PATH


class NER:
    def __init__(self):
        self.model = torch.load(ROOT_PATH + MODEL_PATH + '/model.pth', map_location=torch.device("cpu")).to(DEVICE)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(ROOT_PATH + BERT_PATH)
        self.id2label = np.load(ROOT_PATH + MODEL_PATH + '/id2label.npy', allow_pickle=True).item()

    def get_bio(self, sentences):
        bert_tokens = self.get_tokens(sentences).to(DEVICE)
        seq_lens = self.get_seq_lens(sentences).to(DEVICE)
        _, label_ids = self.model(bert_tokens, seq_lens)
        bios = []
        for (label_id, seq_len) in zip(label_ids, seq_lens):
            bios.append([self.id2label[i] for i in label_id])

        data = self.get_structure_data(bios, sentences)

        return bios, data

    def get_tokens(self, sentences):
        bert_tokens = []
        for index, sentence in enumerate(sentences):
            bert_token = self.tokenizer.convert_tokens_to_ids(list(sentence))
            bert_token = np.array(bert_token)
            bert_tokens.append(torch.LongTensor(bert_token))
        bert_tokens = pad_sequence(bert_tokens, True)
        return bert_tokens

    def get_seq_lens(self, sentences):
        seq_lens = [len(i) for i in sentences]
        seq_lens = torch.LongTensor(seq_lens)
        return seq_lens

    def get_structure_data(self, bios, sentences):
        data = {
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
        for index, bio in enumerate(bios):
            entity = {label.lower().split('-')[1] + 's': [] for label in self.id2label.values() if label != 'O'}
            tag = []
            i = 0
            length = len(bio)
            while i < length:
                if 'B-' in bio[i]:
                    start, label = i, bio[i].split('-')[1]
                    i += 1
                    while i < length and 'I-' in bio[i] and bio[i].split('-')[1] == label: i += 1
                    tag.append((start, i, label))
                    entity[label.lower() + 's'].append(sentences[index][start:i])
                    continue
                i += 1
            for key, value in entity.items():
                data[key].append(value)
            for start, end, label in tag[::-1]:
                sentences[index] = sentences[index][:start] \
                                   + f"{{{sentences[index][start:end]}({label})}}" \
                                   + sentences[index][end:]
        data['tagSentences'] = sentences
        return data
