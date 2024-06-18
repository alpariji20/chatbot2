import random
import tensorflow as tf
import numpy as np
import pickle
import json
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

import string
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping

lemmatizer = WordNetLemmatizer()

# Initialize variables
words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

file_path = 'app/dataset/dataset.json'
stop_words = 'app/dataset/combined_stop_words.txt'

with open(stop_words, 'r', encoding='utf-8') as file:
    stop_word = file.read().splitlines()

with open(file_path, 'r', encoding='utf-8') as file:
    data_json = json.load(file)

def preprocess_words(sentence):
    lowercase_sentence = sentence.lower()
    lowercase_sentence = lowercase_sentence.translate(str.maketrans("", "", string.punctuation))
    lowercase_sentence = lowercase_sentence.strip()
    tokens = nltk.tokenize.word_tokenize(lowercase_sentence)

    freq_tokens = nltk.FreqDist(tokens)
    list_stopwords = set(stopwords.words('indonesian'))
    tokens_without_stopwords = [word for word in freq_tokens if not word in list_stopwords]

    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    output = [(token + " : " + stemmer.stem(token)) for token in tokens_without_stopwords]

    return tokens_without_stopwords

# Process the data
for intent in data_json['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))

        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words] 
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Prepare training data
training = []
output_empty = [0] * len(classes)

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]

    for w in words:
        bag.append(1) if w in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

X = np.array(list(training[:, 0]), dtype=np.float32)
Y = np.array(list(training[:, 1]), dtype=np.float32)

print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1)

print('Data Training has been created')

# Build the model neural network
model = Sequential()
model.add(Dense(256, input_shape=(len(X_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation='softmax'))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Adding Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=200, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping])
model.save('model_chatbot.h5', history)

print(history.history.keys())

# Plotting loss and accuracy
fig, axs = plt.subplots(1, 2, figsize=(20, 5))

# Plotting loss
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(len(loss_train))
axs[0].plot(epochs, loss_train, 'g', label='Training loss')
axs[0].plot(epochs, loss_val, 'b', label='Validation loss')
axs[0].set_title('Training and Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend()

# Plotting accuracy
acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
axs[1].plot(epochs, acc_train, 'g', label='Training accuracy')
axs[1].plot(epochs, acc_val, 'b', label='Validation accuracy')
axs[1].set_title('Training and Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend()

plt.tight_layout()
plt.show()

model.summary()

print("Model created")