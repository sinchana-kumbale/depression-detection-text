import pandas as pd
import numpy as np
from random import choices, shuffle

def read_clean_dataframe(data_path: str) -> pd.DataFrame:
    """Reads and cleans the combined transcripts dataframe from a CSV file."""
    df = pd.read_csv(data_path)
    
    # Remove all trailing and leading spaces
    df["question"] = df["question"].str.strip()
    
    # Filter out rows with missing questions or answers
    df.dropna(subset=["question", "answer"], inplace=True)
    
    return df

def select_person_ids(transcripts_df, question_phrases, m, train_dataset):
    """Selects all participants who have more than m topics"""
    selected_person_ids = []
    person_ids = set(transcripts_df.personId.values)
    train_ids = set(train_dataset[:, 0])

    for person_id in person_ids:
        person_df = transcripts_df[transcripts_df['personId'] == person_id]
        person_topic_count = sum(
            1 for _, row in person_df.iterrows() if any(phrase in row["question"].lower() for phrase in question_phrases)
        )
        if person_topic_count > m and person_id in train_ids:
            selected_person_ids.append(person_id)

    return selected_person_ids

def augment_responses(augmented_responses, selected_person_ids, n, m):
    """Selects n random combination of m topics shuffled from participants who have more than m topics"""
    selected_indices = choices(selected_person_ids, k=n)
    for i in selected_indices:
        person_id = i
        answer = augmented_responses[augmented_responses['personId'] == person_id]['consolidated_response'].values[0]
        topics = answer.split("\n")
        try:
            topics.remove("")
        except:
            pass
        subset = choices(topics, k=m)
        shuffle(subset)
        augmented_responses.loc[len(augmented_responses.index)] = ['\n'.join(subset), person_id]


if __name__ == '__main__':
    m = 5  # Minimum num of unique topics/questions
    n = 200  # Num of augmented samples
    data_path = "/kaggle/input/combined-transcripts/transcriptsall.csv"
    transcripts_df = read_clean_dataframe(data_path)

    question_phrases = [
        "how are you doing today",
        "last time you argued",
        "the last time you felt really happy",
        "controlling your temper",
        "to get a good night's sleep",
        "things that make you really mad",
        "diagnosed with depression",
        "feeling lately",
        "anything you regret",
        "any changes in your behavior",
        "close are you to your family",
        "consider yourself an introvert",
        "therapy"
    ] # Add the topics you have considered for your experiment
    train_dataset = np.array(pd.read_csv('/kaggle/input/test-train-split/train_split_Depression_AVEC2017.csv', delimiter=',', encoding='utf-8'))[:, 0:2]

    selected_person_ids = select_person_ids(transcripts_df, question_phrases, m, train_dataset)

    augmented_responses = pd.read_csv('consolidated_responses.csv') # The dataset consisting of only topic based responses of candidates
    augmented_responses.drop('Unnamed: 0', axis=1, inplace=True)
    augmented_responses.reset_index(drop=True, inplace=True)

    augment_responses(augmented_responses, selected_person_ids, n, m)

    train_df = pd.read_csv('/kaggle/input/test-train-split/train_split_Depression_AVEC2017.csv')
    test_df = pd.read_csv('/kaggle/input/test-train-split/full_test_split.csv')
    dev_df = pd.read_csv('/kaggle/input/test-train-split/dev_split_Depression_AVEC2017.csv')
    test_df['PHQ8_Binary'] = test_df['PHQ_Binary']
    combined_dataset = pd.concat([train_df, test_df, dev_df], ignore_index=True)
    combined_dataset = combined_dataset.sample(frac=1)
    combined_dataset = combined_dataset[['Participant_ID', 'PHQ8_Binary']]

    labels = augmented_responses.merge(combined_dataset, how='left', left_on='personId', right_on='Participant_ID')
    depressed_count = len(labels[labels['PHQ8_Binary'] == 1])
    non_depressed_count = len(labels[labels['PHQ8_Binary'] == 0])
    print("Depressed count:", depressed_count, "Percentage:", (depressed_count / (depressed_count + non_depressed_count)))
    print("Non-depressed count:", non_depressed_count, "Percentage:", non_depressed_count / (non_depressed_count + depressed_count))
