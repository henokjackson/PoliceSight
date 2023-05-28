import pandas as pd
import numpy as np
import spacy
import re
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from tensorflow.keras.models import load_model

def clean_text(text):
  stopwords = nltk.corpus.stopwords.words('english')
  text_cleaning_re = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
  text = re.sub(text_cleaning_re, ' ', str(text).lower()).strip()
  tokens = text.split()
  stop_words = set(stopwords)
  tokens = [w for w in tokens if w not in stop_words]
  
  return " ".join(tokens)

# decode score prediction from the model, to be 0 or 1
def decode_prediction(prediction):
    return 'NOTComplaint' if prediction > 0.5 else 'Complaint'

def predict_class(tweet,model,tokenizer):
  # test model with a new query
  max_length=58

  # clean query text
  input_text = clean_text(tweet)
  # tokenize and pad query test as in training
  input_text = pad_sequences(tokenizer.texts_to_sequences([input_text]),maxlen = max_length)

  # get model prediction
  prediction = model.predict([input_text]).argmax(axis=1)
  # Get the predicted probabilities
  predictions = model.predict([input_text])
  # get decode prediction
  label = decode_prediction(prediction[0])

  # Get the confidence of the prediction
  confidence = np.max(predictions)

  #print("Tweet: \n\n{}\n".format(tweet))
  #print("Score: {} Confidence: {:.2f}% Label: {}".format(prediction, confidence * 100, label))
   
  confidence0 = predictions[0][0]  # Confidence of "complaint" class
  confidence1 = predictions[0][1]  # Confidence of "not complaint" class

  return confidence1,confidence0

def load_lstm():
  tokenizer=Tokenizer(num_words=3000, split=' ')
  model=load_model('complaint_detection/model/best_model.h5')
  return tokenizer,model

def lstm_run(tweet,tokenizer,model):
  confidence1,confidence0=predict_class(tweet,model,tokenizer)
  return confidence1,confidence0