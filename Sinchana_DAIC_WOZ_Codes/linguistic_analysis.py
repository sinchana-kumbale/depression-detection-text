# Analyses the distribution of personal pronoun usage and absolutist words across the depressed and non depressed classes in the shortened and the complete DAIC-WOZ dataset
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

def analyse_personal_pronouns(df: pd.DataFrame) -> pd.DataFrame:
  personal_pronouns = ["i", "me", "my", "mine", "we", "our", "us", "myself", "ourselves"]
  person_pronoun_count = {}
  for _,row in df.iterrows():
    tags = pos_tag(word_tokenize(row['answer']), tagset='universal')
    total_words = len(row['answer'].split())
    person_pronoun_count[row['personId']] = 0
    for word,tag in tags:
      if tag == 'PRON' and word in personal_pronouns:
        person_pronoun_count[row['personId']] += 1
    person_pronoun_count[row['personId']] = person_pronoun_count[row['personId']]*1000/total_words
  df['pronoun count'] = df['personId'].apply(lambda x: person_pronoun_count.get(x))
  return df

def analyse_absolutist_words(df: pd.DataFrame) -> pd.DataFrame:
  absolutist_words = set([
    "absolutely",
    "all",
    "always",
    "complete",
    "completely",
    "constant",
    "constantly",
    "definitely",
    "entire",
    "ever",
    "every",
    "everyone",
    "everything",
    "full",
    "must",
    "never",
    "nothing",
    "totally",
    "whole"])
  absolutist_words_count = {}
  for _,row in df.iterrows():
    words = row['answer'].strip().split()
    total_words = len(words)
    absolutist_words_count[row['personId']] = 0
    for word in words:
      if word in absolutist_words:
        absolutist_words_count[row['personId']] += 1
    absolutist_words_count[row['personId']] = absolutist_words_count[row['personId']]*1000/total_words
  df['absolutist count'] = df['personId'].apply(lambda x: absolutist_words_count.get(x))
  return df

if __name__ == '__main__':
  
  # Extracting the train, test and dev label values
  train_df = pd.read_csv('/content/train_split_Depression_AVEC2017.csv')
  test_df = pd.read_csv('/content/full_test_split.csv')
  dev_df = pd.read_csv('/content/dev_split_Depression_AVEC2017.csv')
  test_df['PHQ8_Binary'] = test_df['PHQ_Binary']
  combined_dataset = pd.concat([train_df,test_df,dev_df],ignore_index=True)
  combined_dataset = combined_dataset.sample(frac=1)


  # Analysing for the complete transcript dataset
  dataset1 = np.array(pd.read_csv('/content/dev_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset2 = np.array(pd.read_csv('/content/full_test_split.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset3 = np.array(pd.read_csv('/content/train_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))
  data_transcripts = pd.DataFrame()
  for i in range(0,len(dataset)):
    df = pd.read_csv('/content/' + str(int(dataset[i][0])) + "_TRANSCRIPT.csv",delimiter='\t')
    text = ''
    for index, row in df.iterrows():
      if row['speaker'] != 'Ellie':
        text = text + ' ' + str(row['value'])
    data_transcripts = data_transcripts.append({'personId':int(dataset[i][0]),'answer':text},ignore_index=True)
  complete_dataset = data_transcripts.merge(combined_dataset,how='left',left_on='personId',right_on='Participant_ID')
  complete_dataset = analyse_personal_pronouns(complete_dataset)
  print("Pronoun Usage by class: \n",complete_dataset.groupby('PHQ8_Binary')['pronoun count'].mean())
  complete_dataset = analyse_absolutist_words(complete_dataset)
  print("Absolutist Word Usage by class: \n",complete_dataset.groupby('PHQ8_Binary')['absolutist count'].mean())
  

  # Analysing for the consolidated dataset
  data_path = '/content/consolidated_responses.csv'
  data_transcripts = pd.read_csv(data_path)
  data_transcripts['answer'] = data_transcripts['consolidated_response']
  data_transcripts.dropna(subset=['answer'], inplace=True)
  data_transcripts['answer'] = data_transcripts['answer'].astype(str)
  complete_dataset = data_transcripts.merge(combined_dataset,how='left',left_on='personId',right_on='Participant_ID')
  complete_dataset = analyse_personal_pronouns(complete_dataset)
  print("Pronoun Usage by class: \n",complete_dataset.groupby('PHQ8_Binary')['pronoun count'].mean())
  complete_dataset = analyse_absolutist_words(complete_dataset)
  print("Absolutist Word Usage by class: \n",complete_dataset.groupby('PHQ8_Binary')['absolutist count'].mean())
  
  
