import random
import tensorflow as tf
import numpy as np
import pickle
import json
import nltk
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Download nltk data
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()

# Paths to the dataset and models
file_path = "app/dataset/dataset.json"
words_path = "app/models/words.pkl"
classes_path = "app/models/classes.pkl"
model_path = "app/models/model_chatbot.h5"

# Load words and classes
words = pickle.load(open(words_path, 'rb'))
classes = pickle.load(open(classes_path, 'rb'))

# Load the trained model
model = load_model(model_path)

# Load the dataset
with open(file_path, 'r', encoding='utf-8') as file:
    dataset_json = json.load(file)

@app.route('/', methods=['GET'])
def index():
    return jsonify({"msg": "Chatbot status : up"})

@app.route('/chat', methods=['POST'])
def predict():
    data = request.get_json()
    user_input = data['user_input']
    print(f"User input: {user_input}")  # Debugging

    intents = predict_class(user_input)
    print(f"Predicted intents: {intents}")  # Debugging

    if not intents:
        response, tag = get_default_response()
    else:
        response, tag = get_response(intents, dataset_json)

    print(f"Response: {response}, Tag: {tag}")  # Debugging
    return jsonify({"msg": response, "tag": tag})

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.75  # Increase the threshold to 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{'intent': classes[r[0]], 'probability': str(r[1])} for r in results]
    return return_list

def get_response(intents_list, dataset_json):
    if not intents_list:
        return get_default_response()

    tag = intents_list[0]['intent']
    list_of_intents = dataset_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            return result, tag
    return get_default_response()

def get_default_response():
    return "Maaf, saya tidak mengerti pertanyaan Anda. Silakan coba lagi.", "not_found"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5570)))
