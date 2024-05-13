import random
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet
import pandas as pd
import numpy as np

def synonym_replacement(words, wordnet):
    """Replaces a random word in a sentence with a synonym using WordNet."""
    synonym_dict = {}
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            synonym_dict[word] = list(synonyms[0].lemmas())
    for i, word in enumerate(words):
        if random.random() < 0.5 and word in synonym_dict:
            synonyms = synonym_dict[word]
            new_word = random.choice(synonyms)
            words[i] = new_word.name()
    return words

def random_insertion(words, wordnet):
    """Inserts a random synonym before a random word in a sentence using WordNet."""
    for i, word in enumerate(words):
        if random.random() < 0.5:
            synonyms = wordnet.synsets(word)
            if synonyms:
                synonym = random.choice(synonyms[0].lemmas()).name()
                words.insert(i, synonym)
    return words

def random_swap(words):
    """Swaps two random words in a sentence."""
    if len(words) > 1:
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return words

def random_deletion(words, p=0.1):
    """Deletes random words in a sentence with probability p."""
    chosen = [random.random() < p for _ in words]
    if not any(chosen):
        chosen[0] = True  # Make sure at least one word remains
    filtered_words = [word for word, chosen in zip(words, chosen) if not chosen]
    return filtered_words

def augment_depressive_samples(dataframe, participant_ids, increase_by_count):
    # Select participant_ids with replacement of the count
    selected_participants = random.choices(participant_ids, k=increase_by_count)
    augmented_data = []
    # For each of the responses to the selected participant from the dataframe do any one of the above actions randomly
    for participant in selected_participants:
        augmentation_method = random.choice([1,2,3,4])
        # print(dataframe[dataframe['participant_id'] == participant]['response'])
        words = dataframe[dataframe['participant_id'] == participant]['response'].values[-1].split()
        if augmentation_method == 1:
            new_words = synonym_replacement(words, wordnet)
        if augmentation_method == 2:
            new_words = random_insertion(words, wordnet)
        if augmentation_method == 3:
            new_words = random_swap(words)
        if augmentation_method == 4:
            new_words = random_deletion(words)
        augmented_data.append({'participant_id': participant , 'response': ' '.join(new_words)})
    return pd.DataFrame(augmented_data)

# Using the functions to add some augmented depressive samples

# Gathering all transcripts
train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv')
dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')
participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)
transcript_df = []
for participant in participant_ids:
    participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
    participant_transcript = participant_transcript[participant_transcript['speaker'] != 'Ellie']
    participant_transcript['value'] = participant_transcript['value'].astype(str)
    transcript_df.append({'participant_id':participant, 'response': ' '.join(list(participant_transcript['value'].values))})

# Selecting only to augment depressive samples
selected_train = train_df[train_df['PHQ8_Binary']==1]
to_be_augmented_df = augment_depressive_samples(pd.DataFrame(transcript_df), list(selected_train.Participant_ID.values), 70)
new_augmented_df = pd.concat((pd.DataFrame(transcript_df), to_be_augmented_df), ignore_index = True)
new_augmented_df.to_csv('augmented_dataset_t1.csv')
