from flask import Flask, request, jsonify
from flask_cors import CORS
from Utils.config import *
from modules.NER import NER
from modules.model import guwenBERT_LSTM_CRF

app = Flask(APP_NAME)
CORS(app, supports_credentials=True)
ner = NER()


@app.route('/guwenNER', methods=['POST'])
def func():
    # sentence = request.form.get('sentence')
    data = request.get_json()
    sentence = data['sentence']
    if not sentence or sentence == '':
        return jsonify({'status': 400})
    bios,  data = ner.get_bio(sentence)
    res = {
        'status': 200,
        'nums': len(sentence),
        'BIOS': bios,
        'data': data
    }
    return jsonify(res)


@app.route('/hello')
def hello():
    return "hello"


if __name__ == "__main__":
    app.run(host=HOST,
            port=PORT,
            threaded=True,
            debug=True
            )
