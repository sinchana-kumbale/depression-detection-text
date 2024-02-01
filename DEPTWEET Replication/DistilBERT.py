import pandas as pd
import numpy as np

data = pd.read_csv('/kaggle/input/hela-dep-det/Depression_Severity_Levels_Dataset.csv')
label_target_map = {'minimum':0,'mild':1,'moderate':2,'severe':3}
data["target"] = data["label"].apply(lambda x: label_target_map[x])
data["text"] = data["text"].astype(str)

REPLACE_BY_SPACE_RE = re.compile('[/(){}\|@,;.]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
import nltk
from nltk.corpus import stopwords
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

#Defining the Model
import ktrain
train, val, preproc = ktrain.text.texts_from_array(x_train = X_train, y_train = y_train, 
                                                                             x_test = X_test, y_test = y_test, 
                                                                             class_names = [0,1,2,3], 
                                                                             preprocess_mode = 'distilbert',
                                                                             maxlen = 512,
                                                                             max_features = 35000) 
model = ktrain.text.text_classifier('distilbert', train_data = train , preproc=preproc)


learner = ktrain.get_learner(model,train_data=train, batch_size=8)

learner.fit(2e-5, 2)
predictor = ktrain.get_predictor(learner.model, preproc)
pred_values = predictor.predict(X_test)
pred_values = predictor.predict_proba(X_test)
learner.validate(val_data = val)

#Model Performance
from keras.utils import to_categorical
y_test_cat = to_categorical(y_test, num_classes=4)
from sklearn.metrics import roc_curve,roc_auc_score
#Finding AUC-ROC Score
n_classes = 4
for i in range(n_classes):    
     print(i, ": ",roc_auc_score(y_test_cat[:,i], pred_values[:,i], multi_class='ovr'))
