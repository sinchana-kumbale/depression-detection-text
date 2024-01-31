#Replicating DEPTWEET work on HelDepDetDataset
import pandas as pd
import numpy as np

data = pd.read_csv('/kaggle/input/hela-dep-det/Depression_Severity_Levels_Dataset.csv')
label_target_map = {'minimum':0,'mild':1,'moderate':2,'severe':3}
data["target"] = data["label"].apply(lambda x: label_target_map[x])

#Data preprocessing
data_0 = data[data['target'] == 0]
data_1 = data[data['target'] == 1]
data_2 = data[data['target'] == 2]
data_3 = data[data['target'] == 3]

print(len(data)==(len(data_0)+len(data_1)+len(data_2)+len(data_3)))

from sklearn.model_selection import train_test_split

def train_validate_test_split(df, seed = 7):
    train, valtest = train_test_split(df, test_size=0.4, random_state = seed)
    val, test = train_test_split(valtest, test_size = 0.5, random_state = seed)
    return train, val, test

train_df_0, valid_df_0, test_df_0 = train_validate_test_split(data_0)
train_df_1, valid_df_1, test_df_1 = train_validate_test_split(data_1)
train_df_2, valid_df_2, test_df_2 = train_validate_test_split(data_2)
train_df_3, valid_df_3, test_df_3 = train_validate_test_split(data_3)


train_df = pd.DataFrame()
train_df = pd.concat([train_df,train_df_0], ignore_index=True)
train_df = pd.concat([train_df,train_df_1], ignore_index=True)
train_df = pd.concat([train_df,train_df_2], ignore_index=True)
train_df = pd.concat([train_df,train_df_3], ignore_index=True)
train_df = train_df.sample(frac=1)

valid_df = pd.DataFrame()
valid_df = pd.concat([valid_df,valid_df_0], ignore_index=True)
valid_df = pd.concat([valid_df,valid_df_1], ignore_index=True)
valid_df = pd.concat([valid_df,valid_df_2], ignore_index=True)
valid_df = pd.concat([valid_df,valid_df_3], ignore_index=True)
valid_df = valid_df.sample(frac=1)

test_df = pd.DataFrame()
test_df = pd.concat([test_df,test_df_0], ignore_index=True)
test_df = pd.concat([test_df,test_df_1], ignore_index=True)
test_df = pd.concat([test_df,test_df_2], ignore_index=True)
test_df = pd.concat([test_df,test_df_3], ignore_index=True)
test_df = test_df.sample(frac=1)


print(len(data)==(len(train_df)+len(valid_df)+len(test_df)))
                  
train_df.to_csv('train.csv')
valid_df.to_csv('valid.csv')
test_df.to_csv('test.csv')
