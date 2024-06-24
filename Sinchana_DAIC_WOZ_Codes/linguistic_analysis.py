# Analyses the distribution of personal pronoun usage and absolutist words across the depressed and non depressed classes in the shortened and the complete DAIC-WOZ dataset
import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
nltk.download('universal_tagset')
nltk.download('averaged_perceptron_tagger')
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from collections import Counter
import matplotlib.pyplot as plt
import umap.umap_ as umap
from textblob import TextBlob
from sklearn.preprocessing import StandardScaler

# Defining positive, negative and depressive word lists
# The text files taken from: https://github.com/Jackustc/Question-Level-Feature-Extraction-on-DAIC-WOZ-dataset
pos_word_list = []
neg_word_list = []
dep_word_list = []
with open('/content/positive-words.txt','r',encoding='utf-8') as f:
  for line in f.readlines():
    pos_word_list.append(str(line).strip())
with open('/content/negative-words.txt','r',encoding='utf-8') as f:
  for line in f.readlines():
    neg_word_list.append(str(line).strip())
with open('/content/depressedword.txt','r',encoding='utf-8') as f:
  for line in f.readlines():
    if len(line.split())==2:
      for ele in line.split()[1].split(','):
        dep_word_list.append(ele)
    else:
      dep_word_list.append(str(line).strip())
          
def weighted_scaler(X, feature_weights):
  """
  Scales features based on weights and then applies standard scaling.
  """
  weighted_features = []
  for i in range(len(X[0])):
    weighted_features.append(X[:, i] * feature_weights[features[i]])
  return StandardScaler().fit_transform(np.array(weighted_features).T)

def analyse_words_used(df: pd.DataFrame) -> pd.DataFrame:
  """
  Analyzes the usage of absolutist words, laughter, sighs, sniffles, positive words,
  negative words, and depressive words in the answer text of each person.
  """

  # Define sets of words for each category
  absolutist_words = {
      "absolutely", "all", "always", "complete", "completely", "constant",
      "constantly", "definitely", "entire", "ever", "every", "everyone",
      "everything", "full", "must", "never", "nothing", "totally", "whole"
  }
  first_personal_pronouns = ["i", "me", "my", "mine", "myself"]
  third_personal_pronouns = ["he", "she", "him", "her", "his", "hers", "they", "them", "theirs"]

  # Create empty dictionaries to store counts for each person
  word_counts = {}

  for _, row in df.iterrows():
    person_id = row["personId"]
    words = row["answer"].strip().split()
    total_words = len(words)

    # Initialize word counts for this person
    word_counts[person_id] = {
        "absolutist": 0,
        "laugh": 0,
        "sigh": 0,
        "sniffle": 0,
        "um": 0,
        "depressive": 0,
        "positive": 0,
        "negative": 0,
        "firstpronoun" : 0,
        "thirdpronoun" : 0,
        "sentimentp" : 0,
        "sentiments" : 0
    }

    # Count pronouns based on nltk tags
    tags = pos_tag(word_tokenize(row['answer']), tagset='universal')
    for word,tag in tags:
      if tag == 'PRON' and word in first_personal_pronouns:
        word_counts[person_id]["firstpronoun"] += 1
      if tag == 'PRON' and word in third_personal_pronouns:
        word_counts[person_id]["thirdpronoun"] += 1
    
    for sentence in row["answer"].split("."):
      sentence_sentiment =  TextBlob(sentence).sentiment
      word_counts[person_id]["sentimentp"] += sentence_sentiment.polarity
      word_counts[person_id]["sentiments"] += sentence_sentiment.subjectivity

    for word in words:
      # Count occurrences based on word categories
      if word.startswith("<laughter>"):
        word_counts[person_id]["laugh"] += 1
      elif word.startswith("<sigh>"):
        word_counts[person_id]["sigh"] += 1
      elif word.startswith("<sniffle>"):
        word_counts[person_id]["sniffle"] += 1
      elif word == "um":
        word_counts[person_id]["um"] += 1
      elif word in dep_word_list:
        word_counts[person_id]["depressive"] += 1
      elif word in pos_word_list:
        word_counts[person_id]["positive"] += 1
      elif word in neg_word_list:
        word_counts[person_id]["negative"] += 1
      elif word in absolutist_words:
        word_counts[person_id]["absolutist"] += 1

    # Calculate word usage percentages
    for category, count in word_counts[person_id].items():
      word_counts[person_id][category] = count * 1000 / total_words

  # Add new columns to the DataFrame with word usage percentages
  for personId, category_counts in word_counts.items():
    for category in category_counts:
      df[f"{category} count"] = df["personId"].apply(lambda x: word_counts[x][category])
  
  return df

if __name__ == '__main__':

  # Extracting the train, test and dev label values
  train_df = pd.read_csv('/content/train_split_Depression_AVEC2017.csv')
  test_df = pd.read_csv('/content/full_test_split.csv')
  dev_df = pd.read_csv('/content/dev_split_Depression_AVEC2017.csv')
  test_df['PHQ8_Binary'] = test_df['PHQ_Binary']
  combined_dataset = pd.concat([train_df,test_df,dev_df],ignore_index=True)
  combined_dataset = combined_dataset.sample(frac=1)

  # Categories for which the dataset is analysed
  categories = ["absolutist","laugh","sigh","sniffle","um","depressive","positive","negative","firstpronoun", "thirdpronoun"]
  
  # Analysing for the complete transcript dataset
  dataset1 = np.array(pd.read_csv('/content/dev_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset2 = np.array(pd.read_csv('/content/full_test_split.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset3 = np.array(pd.read_csv('/content/train_split_Depression_AVEC2017.csv',delimiter=',',encoding='utf-8'))[:, 0:2]
  dataset = np.concatenate((dataset1, np.concatenate((dataset2, dataset3))))
  data_transcripts = []
  for i in range(0,len(dataset)):
    df = pd.read_csv('/content/' + str(int(dataset[i][0])) + "_TRANSCRIPT.csv",delimiter='\t')
    text = ''
    for index, row in df.iterrows():
      if row['speaker'] != 'Ellie':
        text = text + ' ' + str(row['value'])
    data_transcripts.append({'personId':int(dataset[i][0]),'answer':text})
  data_transcripts = pd.DataFrame(data_transcripts)
  complete_dataset = data_transcripts.merge(combined_dataset,how='left',left_on='personId',right_on='Participant_ID')
  complete_dataset = analyse_words_used(complete_dataset)
  for category in categories:
    print("\nStats for " f"{category} count",complete_dataset.groupby('PHQ8_Binary')[f'{category} count'].mean(),complete_dataset.groupby('PHQ8_Binary')[f'{category} count'].std())


  # Analysing for the consolidated dataset
  data_path = '/content/consolidated_responses.csv'
  data_transcripts = pd.read_csv(data_path)
  data_transcripts['answer'] = data_transcripts['consolidated_response']
  data_transcripts.dropna(subset=['answer'], inplace=True)
  data_transcripts['answer'] = data_transcripts['answer'].astype(str)
  complete_dataset = data_transcripts.merge(combined_dataset,how='left',left_on='personId',right_on='Participant_ID')
  complete_dataset = analyse_words_used(complete_dataset)
  for category in categories:
    print("\nStats for " f"{category} count",complete_dataset.groupby('PHQ8_Binary')[f'{category} count'].mean(),complete_dataset.groupby('PHQ8_Binary')[f'{category} count'].std())
  
  # Visualising for PCA
  features = ['absolutist count','laugh count','sigh count','sniffle count','um count','depressive count','positive count', 'negative count', 'firstpronoun count','thirdpronoun count', 'sentimentp count', 'sentiments count']
  feature_weights = {
      'absolutist count': 1,
      'laugh count': 1,
      'sigh count': 1,
      'sniffle count': 1,
      'um count': 1,
      'depressive count': 1,  
      'positive count': 1,  
      'negative count': 1,  
      'firstpronoun count': 1,
      'thirdpronoun count': 1,
      'sentimentp count': 1,
      'sentiments count': 1  
      }
  # Separating out the features
  x = complete_dataset.loc[:, features].values
  
  # Separating out the target
  y = complete_dataset.loc[:,['PHQ8_Binary']].values
  
  # Standardizing the features
  x = weighted_scaler(x, feature_weights)
  
  
  umap_model = umap.UMAP(n_components=3)
  umap_data = umap_model.fit_transform(x)
  targets = [0,1]
  colors = ['b', 'g']
  
  depressed = complete_dataset[complete_dataset['PHQ8_Binary'] == 1]
  not_depressed = complete_dataset[complete_dataset['PHQ8_Binary'] == 0]
  
  plt.scatter(umap_data[depressed.index, 0], umap_data[depressed.index, 1], label='Depressed')
  plt.scatter(umap_data[not_depressed.index, 0], umap_data[not_depressed.index, 1], label='Not Depressed')
  
  plt.xlabel('UMAP Dimension 1')
  plt.ylabel('UMAP Dimension 2')
  plt.title('UMAP Visualization')
  plt.legend()
  plt.show()
