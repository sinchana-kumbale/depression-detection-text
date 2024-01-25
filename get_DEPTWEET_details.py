#Extracting information about DEPTWEET dataset
import pandas as pd
import numpy as np

df = pd.read_csv('/kaggle/input/deptweet-dataset/deptweet_dataset.csv')
#Displaying number of tweets for each label
print(df['label'].value_counts())
#Finding average confidence_score for each label
print('non-depressed', df.loc[df['label'].eq('non-depressed'),'confidence_score'].mean())
print('mild', df.loc[df['label'].eq('mild'),'confidence_score'].mean())
print('moderate', df.loc[df['label'].eq('moderate'),'confidence_score'].mean())
print('severe', df.loc[df['label'].eq('severe'),'confidence_score'].mean())
