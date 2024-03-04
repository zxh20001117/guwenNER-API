import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import AutoModel
from fastNLP import seq_len_to_mask


from Utils.config import ROOT_PATH, BERT_PATH, DEVICE

root_path = ROOT_PATH
bert_path = BERT_PATH


class guwenBERT_LSTM_CRF(nn.Module):
    def __init__(self, label_size, lstm_hidden_size):
        super(guwenBERT_LSTM_CRF, self).__init__()
        self.bert = AutoModel.from_pretrained(root_path+bert_path).to(DEVICE)
        for param in self.bert.parameters():
            param.requires_grad_(False)

        self.lstm = nn.LSTM(
            1024,
            lstm_hidden_size,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(2 * lstm_hidden_size, label_size)
        self.crf = CRF(num_tags=label_size, batch_first=True)
        self.embed_drop = nn.Dropout(0.4)
        self.output_drop = nn.Dropout(0.2)

    def _get_lstm_feature(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask, )
        out = out.last_hidden_state
        out = self.embed_drop(out)
        out, _ = self.lstm(out)
        return self.output_drop(self.fc(out))

    def forward(self, input_ids, seq_lens):
        attention_mask = seq_len_to_mask(seq_lens, max_len=input_ids.shape[1]).bool()
        out = self._get_lstm_feature(input_ids, attention_mask)
        return out, self.crf.decode(out, attention_mask.bool())

    def get_loss(self, out, seq_lens, labels):
        attention_mask = seq_len_to_mask(seq_lens, max_len=input_ids.shape[1]).bool()
        return -self.crf.forward(out, labels, attention_mask.bool(), reduction="mean")
