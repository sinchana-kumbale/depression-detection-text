from transformers import AutoTokenizer, AutoModelWithLMHead
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def get_emotion(text):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids,
               max_length=5)

    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label

def get_sentiment(text):
    input_ids = tokenizer.encode(text + '</s>', return_tensors='pt')

    output = model.generate(input_ids=input_ids,
               max_length=5)

    dec = [tokenizer.decode(ids) for ids in output]
    label = dec[0]
    return label

# Extracting Emotion information

tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-emotion")

data_path = '/content/transcriptsall.csv'
t2emotion = {}

# Preparing the dataset
data_transcripts = pd.read_csv(data_path)
data_transcripts.dropna(subset=['answer'], inplace=True) #Retain only the records where the respondent is speaking something
data_transcripts['answer'] = data_transcripts['answer'].astype(str)


data_transcripts['emotion'] = data_transcripts['answer'].apply(lambda x: get_emotion(x))
data_transcripts['emotion'] = data_transcripts['emotion'].apply(lambda x: x[5:-4]) #To remove <pad> <sequence>

print(data_transcripts.emotion.value_counts())

# Extracting Sentiment Information
tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-imdb-sentiment")

model = AutoModelWithLMHead.from_pretrained("mrm8488/t5-base-finetuned-imdb-sentiment")

data_transcripts['sentiment'] = data_transcripts['answer'].apply(lambda x: get_sentiment(x))
data_transcripts['sentiment'] = data_transcripts['sentiment'].apply(lambda x: x[5:-4])

print(data_transcripts.sentiment.value_counts())

# Extracting labels to categorise emotions and sentiments based on depression and non-depression
train_df = pd.read_csv('/content/train_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('/content/full_test_split.csv')
dev_df = pd.read_csv('/content/dev_split_Depression_AVEC2017.csv')
test_df['PHQ8_Binary'] = test_df['PHQ_Binary']

combined_dataset = pd.concat([train_df,test_df,dev_df],ignore_index=True)
combined_dataset = combined_dataset.sample(frac=1)

labels = data_transcripts.merge(combined_dataset,how='left',left_on='personId',right_on='Participant_ID')

# Preparing the dataset for analysis
sen_vals = ['positive', 'negative']
emo_vals = ['joy', 'sadness', 'anger', 'fear','love','surprise']
map = {0:'No Depression',1:'Depression'}
labels['depression_label'] = labels['PHQ8_Binary'].apply(lambda x: map[x])
labels['sentiment'] = labels['sentiment'].apply(lambda x: x.strip())
labels['emotion'] = labels['emotion'].apply(lambda x: x.strip())

# Plotting the sentiment wrt depression level chart
data = labels[labels.sentiment.isin(sen_vals)].groupby(['sentiment', 'depression_label'])['PHQ8_Binary'].count().reset_index()
g = sns.catplot(x="sentiment", y="PHQ8_Binary", hue="depression_label", data=data, kind='bar', legend_out=False)
plt.ylabel('Frequency')
plt.gca().get_legend().set_title('')
plt.savefig('sentiment.pdf')

# Plotting the emotions wrt depression level chart
data = labels[labels.emotion.isin(emo_vals)].groupby(['emotion', 'depression_label'])['PHQ8_Binary'].count().reset_index()
g = sns.catplot(x="emotion", y="PHQ8_Binary", hue="depression_label", data=data, kind='bar', legend_out=False)
plt.ylabel('Frequency')
plt.gca().get_legend().set_title('')
plt.savefig('emotion.pdf')
