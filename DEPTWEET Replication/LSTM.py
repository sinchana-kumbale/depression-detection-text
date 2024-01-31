#LSTM 
import tensorflow as tf
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import text, sequence
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.models import Model, Sequential
from keras.layers import Dense, Input, LSTM, SimpleRNN, Embedding, Dropout, SpatialDropout1D, Activation, Conv1D,GRU
from keras.layers import Conv1D, Bidirectional, GlobalMaxPool1D, MaxPooling1D, BatchNormalization, Add, Flatten
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, concatenate, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from sklearn import svm
from keras.utils import plot_model
from sklearn.model_selection  import train_test_split
#from sklearn.cross_validation import train_test_split


# For custom metrics
import keras.backend as K
from keras.utils import plot_model
from keras.callbacks import EarlyStopping 


import pandas as pd
import numpy as np
 
import matplotlib.pyplot as plt



import seaborn as sns
from IPython.display import Image

from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer 
import os, re, csv, math, codecs
from nltk.tokenize import word_tokenize
import string
import gensim
import os
import re

np.random.seed(0)

import plotly as py
import plotly.graph_objs as go
import plotly



# Install dependencies
!apt install graphviz
!pip install pydot pydot-ng
!echo "Double check with Python 3"
!python -c "import pydot"

REPLACE_BY_SPACE_RE = re.compile('[/(){}\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
import nltk
nltk.download('stopwords')
stops = set(stopwords.words('english'))


def clean_text(text):
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text. substitute the matched string in REPLACE_BY_SPACE_RE with space.
    text = ' '.join(word for word in text.split() if word not in stops) # remove stopwors from text
    return text
data['text'] = data['text'].astype(str)
data['text'] = data['text'].apply(clean_text)

modified_train_df = pd.concat([train_df,valid_df],ignore_index=True)
modified_train_df = modified_train_df.sample(frac=1)
temp = train_df
train_df = modified_train_df
#Convert text to vectors using keras preprocessing library tools

X_train = train_df["text"].values
X_test  = test_df["text"].values

y_train = train_df[["target"]].values
y_test  = test_df[["target"]].values

from keras.utils import to_categorical   

y_train_categorical_labels = to_categorical(y_train, num_classes=4)
y_test_categorical_labels = to_categorical(y_test, num_classes=4)

test_ck = pd.DataFrame(X_train)
test_ck.columns = ["Response"]

length_of_the_messages = test_ck["Response"].str.split("\s+")


print("Max number of words = ", length_of_the_messages.str.len().max())
print("Index = ", length_of_the_messages.str.len().idxmax())

num_words = 20000 #Max. words to use per comment
max_features = 60000 #Max. number of unique words in embeddinbg vector
max_len = 8192 #Max. number of words per toxic comment to be use
embedding_dims = 64 #embedding vector output dimension 
num_epochs = 10 # (before 5)number of epochs (number of times that the model is exposed to the training dataset)
val_split = 0.1
batch_size2 = 64 

#Tokenization
tokenizer = Tokenizer(num_words)
tokenizer.fit_on_texts(list(X_train))

#Convert tokenized commnent to sequnces
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
 
# padding the sequences
X_train = sequence.pad_sequences(X_train, max_len, padding='post')
X_test  = sequence.pad_sequences(X_test,  max_len, padding='post')

print('X_train shape:', X_train.shape)
print('X_test shape: ', X_test.shape)

X_tra, X_val, y_tra, y_val = train_test_split(X_train, y_train_categorical_labels, train_size =0.9, random_state=233)
early = EarlyStopping(monitor="val_loss", mode="min", patience=4)

LSTM_model = Sequential([
    Embedding(input_dim =num_words, input_length=max_len, output_dim=embedding_dims, trainable=False),
    SpatialDropout1D(0.3),
    #Bidirectional layer will enable our model to predict a missing word in a sequence, 
    #So, using this feature will enable the model to look at the context on both the left and the right.
    LSTM(64, return_sequences=True),
    #**batch normalization layer** normalizes the activations of the previous layer at each batch, 
    #i.e. applies a transformation that maintains the mean activation close to 0 and the activation standard deviation close to 1. 
    BatchNormalization(),
    Dropout(0.3),
    GlobalMaxPool1D(),
    Dense(32, activation = 'relu'),
    Dropout(0.3),
    Dense(4, activation = 'sigmoid')
])

LSTM_model.compile(loss='categorical_crossentropy', optimizer=Adam(0.01), metrics=['accuracy'])

LSTM_model.summary()


plot_model(LSTM_model, to_file='LSTM_model.png')

LSTM_model_fit = LSTM_model.fit(X_tra, y_tra, batch_size=batch_size2, epochs=num_epochs, validation_data=(X_val, y_val), callbacks=[early])

pred_lstm = LSTM_model.predict(X_test)

from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
# roc curve for classes
fpr = {}
tpr = {}
thresh ={}
roc = {}

n_class = 4

for i in range(n_class):    
    fpr[i], tpr[i], thresh[i] = roc_curve(y_test, pred_lstm[:,i], pos_label=i)
    roc[i]   = roc_auc_score(y_test_categorical_labels[:,i], pred_lstm[:,i], multi_class='ovr')

    
# plotting    
plt.plot(fpr[0], tpr[0], linestyle='-',color='orange', label= 'non-depressed '+ str(round(roc[0],6)))
plt.plot(fpr[1], tpr[1], linestyle='-',color='green', label= 'mild '+ str(round(roc[1],6)))
plt.plot(fpr[2], tpr[2], linestyle='-',color='blue', label= 'moderate '+ str(round(roc[2],6)))
plt.plot(fpr[3], tpr[3], linestyle='-',color='red', label= 'severe '+ str(round(roc[3],6)))


plt.title('Multiclass ROC Curve for SVM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive rate')
plt.legend(loc='best')
from sklearn.metrics import roc_curve, auc

def calc_roc_auc(all_labels, all_logits, name=None):
    attributes = ['non-depressed','mild','moderate','severe']
    
    

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(0,len(attributes)):
        fpr[i], tpr[i], _ = roc_curve(all_labels[:, i], all_logits[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    plt.plot(fpr[0], tpr[0], color='blue', label='%s %g' % (attributes[0], roc_auc[0]))
    plt.plot(fpr[1], tpr[1], color='orange', label='%s %g' % (attributes[1], roc_auc[1]))
    plt.plot(fpr[2], tpr[2], color='green', label='%s %g' % (attributes[2], roc_auc[2]))
    plt.plot(fpr[3], tpr[3], color='red', label='%s %g' % (attributes[3], roc_auc[3]))

    plt.xticks(np.arange(0, 1.2, step=0.2))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.title('ROC Curve')
    plt.grid(False)
    
    

    plt.savefig(f"---roc_auc_curve_Bi_LSTM---.pdf")
    plt.clf()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(all_labels.ravel(), all_logits.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    print(f'ROC-AUC Score: {roc_auc["micro"]}')

calc_roc_auc(y_test_categorical_labels,pred_lstm)
