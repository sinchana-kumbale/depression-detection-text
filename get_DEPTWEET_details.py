#Extracting information about DEPTWEET dataset
import pandas as pd
import numpy as np

df = pd.read_csv('deptweet_dataset.csv')
#Displaying number of tweets for each label
print(df['label'].value_counts())
#Finding average confidence_score for each label
labels = set(df['label'])
for label in labels:
    print(label,df.loc[df['label'].eq(label),'confidence_score'].mean())
