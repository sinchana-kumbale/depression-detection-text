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
        max_len_text = max(max_len_text,len(text))
        tokens = roberta.encode(text)[:512]
        pickle.dump(tokens, open('transcripts_tokens' + fname_tok, 'wb') )
        last_layer_features = roberta.extract_features(tokens)
#         print(last_layer_features.shape)
        pickle.dump(last_layer_features, open('transcripts_features' + fname_feat, 'wb') )
