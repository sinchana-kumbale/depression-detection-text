import random
import pickle
import pandas as pd
import numpy as np

def get_response_examples(df, phq_cols, label_cols):
    final_examples = []
    # Iterate through PHQ question columns
    for i, phq_col in enumerate(phq_cols):
        label_col = label_cols[i]
        examples = []
        
        # Select responses with one from each label (same as before)
        for label in range(4):
            filtered_df = df[df[label_col] == label]
            if not filtered_df.empty:
                random_index = random.choice(range(len(filtered_df)))
                response = filtered_df[phq_col].iloc[random_index]
                examples.append(f"\nResponse: {response} Label: {label}")
            else:
                examples.append(f"\nResponse: (no data) Label: {label}")
        final_examples.append("".join(examples))
    return final_examples




input_df = pd.read_csv('/kaggle/input/phq-similar-responses/phq_based_responses.csv')
train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')

# Combine train and dev dfs
combined_df = pd.concat([train_df, dev_df], ignore_index=True)
input_df = input_df.merge(combined_df, left_on='personId', right_on='Participant_ID' , how='left')
input_df = input_df.dropna()

indices = len(input_df)
labels = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']

few_shot_examples = get_response_examples(input_df, [f"phq_response{i+1}" for i in range(8)], labels)
with open('fewshotexamples', 'wb') as fp:
    pickle.dump(few_shot_examples, fp)
    
