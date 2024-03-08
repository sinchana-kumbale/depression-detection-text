from textblob import TextBlob
import numpy as np
import pandas as pd

def is_emotionally_relevant(text, threshold=0.1):
    """Checks if a sentiment polarity of a given text exceeds a threshold"""
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    return abs(sentiment) > threshold

def extract_emotionally_relevant(text):
    """Extracts sentences with sentiment polarity exceeding a threshold."""
    relevant_sentences = []
    for sentence in text.split('.'):
        if is_emotionally_relevant(sentence):
            relevant_sentences.append(sentence.strip())
    return '. '.join(relevant_sentences)

def load_person_responses(data_path, dataset):
    """
    Loads and processes person responses from individual transcripts.
    """

    person_responses = {}
    for i in range(len(dataset)):
        person_id = str(int(dataset[i][0]))
        text = ""

        # Load transcript
        df = pd.read_csv(data_path + person_id + "_TRANSCRIPT.csv", delimiter='\t')

        # Extract non-interviewer speaker dialogues
        for _, row in df.iterrows():
            if row['speaker'] != 'Ellie':
                text += ". " + str(row['value'])

        # Apply emotional relevance extraction
        processed_text = extract_emotionally_relevant(text)
        person_responses[person_id] = processed_text

    return person_responses

if __name__ == '__main__':
    data_path = '/kaggle/input/daic-woz-transcripts/'
    dataset1 = np.array(pd.read_csv('/kaggle/input/test-train-split/dev_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
    dataset2 = np.array(pd.read_csv('/kaggle/input/test-train-split/full_test_split.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
    dataset3 = np.array(pd.read_csv('/kaggle/input/test-train-split/train_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
    dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))

    person_responses = load_person_responses(data_path, dataset)
    consolidated_responses = pd.DataFrame.from_dict(person_responses, orient='index', columns=['consolidated_response'])
    consolidated_responses['personId'] = consolidated_responses.index

    consolidated_responses.to_csv('emotional_only.csv')
