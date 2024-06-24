import pandas as pd
from nltk.corpus import stopwords  
from nltk.tokenize import word_tokenize
import smart_open
import gensim.utils
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim 
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel
%matplotlib inline

# Setting up the stopwords with some custom stopwords
stop_words = stopwords.words("english")
stop_words.extend(["um","uh"])

def clean_text(text):
    text = text.lower()
    text = "".join([char for char in text if char.isalnum() or " "])
    tokens = word_tokenize(text)
    filtered_words = [w for w in tokens if w not in stop_words and not(w.startswith("'"))]
    return ' '.join(filtered_words)

def get_sentence_topics(sentences, lda_model, dictionary, threshold, selected_topics):
    sentence_topics = []
    for sentence in sentences:
        sentence = str(sentence)
        bow = dictionary.doc2bow(sentence.split())
        topic_probs = lda_model[bow]
        #print(topic_probs)
        filtered_topics = [t for t, p in topic_probs if p >= threshold and t in selected_topics]
        sentence_topics.append(filtered_topics)
    return sentence_topics

def create_lda_model(cleaned_transcripts, topic_count):
    dictionary = corpora.Dictionary([cleaned_transcripts])
    corpus = [dictionary.doc2bow(text.split()) for text in cleaned_transcripts]
    lda_model = models.LdaMulticore(corpus, id2word=dictionary, num_topics=topic_count, passes=10)
    return lda_model, corpus, dictionary

def visualise_lda_model(lda_model, corpus, dictionary):
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    print(vis)

def get_particpiant_responses(participant_ids, lda_model, dictionary, selected_topics):
    participant_topic_wise = {}
    for participant in participant_ids:
        particpant_sentences = []
        participant_topic_wise[participant] = ''
        participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
        participant_transcript = participant_transcript[participant_transcript['speaker']!='Ellie']
        particpant_sentences.extend(list(participant_transcript['value'].values))
        participant_topics = get_sentence_topics(particpant_sentences,lda_model, dictionary, 0.6, selected_topics)
        for i in range(len(particpant_sentences)):
            if len(participant_topics[i]) != 0:
                participant_topic_wise[participant] += '. ' + particpant_sentences[i]
    return participant_topic_wise

if __name__ == '__main__':

    # Getting all valid participants
    train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
    test_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv')
    dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')
    participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)

    # Creating the cleaned transcript for the LDA Model
    transcript_texts = []
    for participant in participant_ids:
        participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
        transcript_texts.extend(list(participant_transcript['value'].values))
    cleaned_transcripts = [clean_text(str(text)) for text in transcript_texts]


    # Creating a subset of the topic through manual selection
    lda_model, corpus, dictionary = create_lda_model(cleaned_transcripts, 29)
    visualise_lda_model(lda_model, corpus, dictionary)
    
    # Creating a subset of topics through coherence scores
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=[cleaned_transcripts], dictionary=dictionary, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence_per_topic()
    selected_topics = []
    for idx, coherence_score in enumerate(coherence_lda):
      if coherence_score > threshold:
        selected_topics.append(idx)
    participant_topic_wise = get_particpiant_responses(participant_ids, lda_model, dictionary, selected_topics)
    topic_wise_responses = pd.DataFrame.from_dict(participant_topic_wise, orient="index", columns=["topic_responses"])
    topic_wise_responses["personId"] = topic_wise_responses.index
    topic_wise_responses.to_csv('topic_wise_responses_choerence_score.csv')


    
