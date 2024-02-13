# Code replicated from: https://github.com/prabhat1081/Anxiety-Detection-from-free-form-audio-journals
import fairseq
import torch
import os
import glob
from fairseq.models.wav2vec import Wav2VecModel
from fairseq.models.roberta import RobertaModel
import tqdm
import librosa
import math
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix,  classification_report
import matplotlib.pyplot as plt
import IPython
from sklearn import datasets, metrics, model_selection, svm

# Training and predicting the performance of a classifier
def analyze_clf(clf, y_prob=None):
    if y_prob is None:
        clf = clf.fit(X_train, y_train) 
        y_pred = clf.predict(X_valid)
        if not hasattr(clf, 'predict_proba'):
            y_prob = clf.decision_function(X_valid)
        else:
            y_prob = clf.predict_proba(X_valid)[:, 1]
    else:
        y_pred = y_prob > 0.5
    print(y_pred.shape)
    from sklearn.metrics import confusion_matrix,  classification_report
    y_true = y_valid
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))
    print(roc_auc_score(y_true=y_true, y_score=y_prob))


text_feat = {}
with torch.no_grad():
    for text_feat_file in tqdm.tqdm(glob.glob("/kaggle/working/*_feat.p")):
        fname = os.path.basename(text_feat_file)[-10:-7]
        ft = pickle.load( open( text_feat_file, "rb" ) )
        text_feat[fname] = ft[0][0]

df1 = pd.DataFrame(columns=['personId','label','transcripts_features'])
for i in range(0,len(dataset1)):
    if str(int(dataset1[i][0])) in text_feat:
        df1.loc[len(df1.index)] = [str(int(dataset1[i][0])),int(dataset1[i][1]) , text_feat[str(int(dataset1[i][0]))]]

df2 = pd.DataFrame(columns=['personId','label','transcripts_features'])
for i in range(0,len(dataset2)):
    if str(int(dataset2[i][0])) in text_feat:
        df2.loc[len(df2.index)] = [str(int(dataset2[i][0])),int(dataset2[i][1]) , text_feat[str(int(dataset2[i][0]))]]

df3 = pd.DataFrame(columns=['personId','label','transcripts_features'])
for i in range(0,len(dataset3)):
    if str(int(dataset3[i][0])) in text_feat:
        df3.loc[len(df3.index)] = [str(int(dataset3[i][0])),int(dataset3[i][1]) , text_feat[str(int(dataset3[i][0]))]]


df = pd.concat([df1,df2,df3],axis=0)
feats, y = np.stack(df.transcripts_features.values), df['label']
print(feats.shape, y.shape)

X_train,y_train = np.stack(df3.transcripts_features.values), df3['label']
X_valid,y_valid = np.stack(df1.transcripts_features.values), df1['label']
X_test,y_test = np.stack(df2.transcripts_features.values), df2['label']

# Random baseline
from sklearn.dummy import DummyClassifier
dummy_clf = DummyClassifier(strategy="uniform")
analyze_clf(dummy_clf)

# Other classifiers
clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0, max_iter=300, penalty='l1', solver='liblinear'))
analyze_clf(clf)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=1, max_iter=1500))
analyze_clf(clf)

clf = make_pipeline(StandardScaler(), SVC(C=10))
analyze_clf(clf)

# Final best classifier
clf = make_pipeline(StandardScaler(), SVC(C=10))
clf.fit(np.concatenate([X_train, X_valid]), np.concatenate([y_train, y_valid]))
y_true = y_test
y_pred = clf.predict(X_test)
y_prob = clf.decision_function(X_test)
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))
print(roc_auc_score(y_true=y_true, y_score=y_prob))

# Plotting ROC - AUC Curve
metrics.RocCurveDisplay.from_estimator(clf, X_test, y_test, pos_label=None, name='')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.savefig('roc_curve.pdf')
metrics.PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, pos_label=None, name='')  
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('prc_curve.pdf')
