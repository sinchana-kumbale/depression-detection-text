#Extracting information about DEPTWEET dataset
import pandas as pd
import numpy as np

def get_dataset_details(filename):
    df = pd.read_csv(filename)
    #Displaying number of tweets for each label
    print(df['label'].value_counts())
    #Finding average confidence_score for each label
    labels = set(df['label'])
    for label in labels:
        print(label,df.loc[df['label'].eq(label),'confidence_score'].mean())
if __name__ == '__main__':
    get_dataset_details('deptweet_dataset.csv')
