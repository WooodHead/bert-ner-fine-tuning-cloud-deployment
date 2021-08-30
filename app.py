import pickle

import torch
from flask import Flask, request, jsonify, render_template
from transformers import BertTokenizer

from bert_ner_model import BERTForNerModel, model_save_path, classify_tokens

app = Flask(__name__, static_folder="static")
# Load model from the saved checkpoint
idx2tag = pickle.load(open(model_save_path + "idx2tag_map.pkl", 'rb'))
model = BERTForNerModel(len(idx2tag), pretrained=False)
model.load_state_dict(torch.load(model_save_path + "bert_ner.pth", map_location=torch.device('cpu')))
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


@app.route("/")
def index():
    return render_template('index.html')


@app.route('/classify_tokens', methods=['POST'])
def classify():
    return jsonify(classify_tokens(model, tokenizer, idx2tag, request.json['sentence'], torch.device("cpu")))


if __name__ == "__main__":
    from waitress import serve

    serve(app, host="0.0.0.0", port=8080)
