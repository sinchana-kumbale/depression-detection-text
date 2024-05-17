import random
import pickle
import pandas as pd
import numpy as np

input_df = pd.read_csv('/kaggle/input/phq-similar-responses/phq_based_responses.csv')
train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')

# Combine train and dev dfs
combined_df = pd.concat([train_df, dev_df], ignore_index=True)
input_df = input_df.merge(combined_df, left_on='personId', right_on='Participant_ID' , how='left')
input_df = input_df.dropna()

indices = len(input_df)
labels = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']
few_shot_examples = []
for i in range(8):
    selected_indices = random.choices(range(indices), k=4)
    question_responses = ''
    for index in selected_indices:
        response = str(input_df.iloc[index]['phq_response'+str(i+1)])
        label = str(input_df.iloc[index][labels[i]])
        question_responses += '\nResponse: ' + response + " Label: " + label
    few_shot_examples.append(question_responses)

with open('fewshotexamples', 'wb') as fp:
    pickle.dump(few_shot_examples, fp)
    
