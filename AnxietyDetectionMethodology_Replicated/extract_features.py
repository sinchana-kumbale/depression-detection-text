import fairseq
import torch
import os
import glob
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel
from nltk.corpus import stopwords
import re
import tqdm
import librosa
import math
import numpy as np
import pandas as pd
import pickle


stop_words = set(stopwords.words('english'))
def remove_stopwords(text):
    new_sentence = []
    list_of_words = text.split()
    for word in list_of_words:
        if word not in stop_words:
            new_sentence.append(word)
    return ' '.join(new_sentence)

def clean_text(text):
    """
    Cleans text by removing unnecessary characters, expanding contractions,
  and handling special cases.
  """
    clean_text_regex = r"[^A-Za-z0-9\s\!\?\+\-\=]"
    text = re.sub(clean_text_regex, " ", text)
    
    # Replace contractions with expanded forms
    contractions = {
      "what's": "what is",
      "\'s": " ",
      "\'ve": " have",
      "can't": "cannot",
      "n't": " not",
      "i'm": "i am",
      "\'re": " are",
      "\'d": " would",
      "\'ll": " will",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Fix punctuation spacing and special characters
    text = re.sub(r"([\!\?\+\-\=])", r" \1 ", text)
    text = text.replace(" eg ", " eg ")
    text = text.replace(" b g ", " bg ")
    text = text.replace(" u s ", " american ")
    text = text.replace("\0s", "0")
    text = text.replace(" 9 11 ", "911")
    text = text.replace("e - mail", "email")
    text = text.replace("j k", "jk")
    
    # Remove extra spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    return text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove stopwords
    # text = remove_stopwords(text)
    
    text = clean_text(text)
    
    return text

roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
roberta.eval()

dataset1 = np.array(pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
dataset2 = np.array(pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
dataset3 = np.array(pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))

with torch.no_grad():
    for i in range(0,len(dataset)):
        df = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(int(dataset[i][0])) + "_TRANSCRIPT.csv",delimiter='\t')
        fname_feat = os.path.basename(str(int(dataset[i][0]))) + "_feat.p"
        fname_tok = os.path.basename(str(int(dataset[i][0]))) + "_tokens.p"
        text = ''
        for index, row in df.iterrows():
            if row['speaker'] != 'Ellie':
                text += str(row['value'])
                text = preprocess_text(text)
                tokens = roberta.encode(text)[:512]
                pickle.dump(tokens, open('transcripts_tokens' + fname_tok, 'ab') )
                last_layer_features = roberta.extract_features(tokens)
                pickle.dump(last_layer_features, open('transcripts_features' + fname_feat, 'ab') )
        
