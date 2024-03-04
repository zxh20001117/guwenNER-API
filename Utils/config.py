import torch

APP_NAME = "guwenNER"
HOST = 'localhost'
PORT = '28796'

# DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = "cpu"

ROOT_PATH = '/Users/zhengxuhui/研究生数据/研究试验/guwenNER API'
BERT_PATH = '/data/model_weights/guwenbert-large'
MODEL_PATH = '/data/model_weights/model 2024 01 21'

lstm_hidden_size = 512
