from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import json

data = pd.read_csv("ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")
words = list(set(data["Word"].values))
words.append("ENDPAD")
words.sort()
n_words = len(words); n_words
tags = list(set(data["Tag"].values))
tags.sort()
max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}


app = Flask(__name__)
global graph
graph = tf.get_default_graph()
model = load_model('LSTM_NER.h5')

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')

@app.route("/predict", methods=["GET","POST"])
def predict():
    args = parser.parse_args()
    user_query = args['query']
    test_sentence = str(user_query).split()
    x_test_sent = pad_sequences(sequences=[[word2idx.get(w, 0) for w in test_sentence]],padding="post", value=0, maxlen=max_len)
    with graph.as_default():
        p = model.predict(np.array([x_test_sent[0]]))
        p = np.argmax(p, axis=-1)
        res = {}
        for w, pred in zip(test_sentence, p[0]):
            if 'per' in tags[pred]:
                res[w] = 'PER'
            elif 'geo' in tags[pred]:
                res[w] = 'LOC'
            elif 'org' in tags[pred]:
                res[w] = 'ORG'
            elif tags[pred] == 'O':
                res[w] = 'O'
            else:
                res[w] = 'MISC'
        return json.dumps(res)

app.run(threaded=True)