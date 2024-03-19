import nltk
from nltk.corpus import stopwords
import re
import ktrain
from ktrain import text
nltk.download('stopwords')


def modified_stopwords(stop_words):
    """
    Removes words with negative connotation from the list of stopwords
    """
    # These specific values have been choosen for our use case. Use different values as needed
    negative_words = ['isn\'t','didn','nor','aren\'t','didn\'t','wasn\'t','could\'nt',"shan't","needn't","hasn't","wouldn't","mustn't","doesn't","weren't","not"]
    
    for negative_word in negative_words:
        if negative_word in stop_words:
            stop_words.remove(negative_word)
    return stop_words


stop_words_g = set(stopwords.words('english'))

def remove_stopwords(text):
    stop_words = modified_stopwords(stop_words_g)
    new_sentence = []
    list_of_words = text.split()
    for word in list_of_words:
        if word not in stop_words:
            new_sentence.append(word)
    return ' '.join(new_sentence)


def clean_text(text):
    """
    Cleans text by removing unnecessary characters, expanding contractions,
  and handling special cases.
  """
    clean_text_regex = r"[^A-Za-z0-9\s\!\?\+\-\=]"
    text = re.sub(clean_text_regex, " ", text)
    
    # Replace contractions with expanded forms
    contractions = {
      "what's": "what is",
      "\'s": " ",
      "\'ve": " have",
      "can't": "cannot",
      "n't": " not",
      "i'm": "i am",
      "\'re": " are",
      "\'d": " would",
      "\'ll": " will",
    }
    for contraction, expansion in contractions.items():
        text = text.replace(contraction, expansion)
    
    # Fix punctuation spacing and special characters
    text = re.sub(r"([\!\?\+\-\=])", r" \1 ", text)
    text = text.replace(" eg ", " eg ")
    text = text.replace(" b g ", " bg ")
    text = text.replace(" u s ", " american ")
    text = text.replace("\0s", "0")
    text = text.replace(" 9 11 ", "911")
    text = text.replace("e - mail", "email")
    text = text.replace("j k", "jk")
    
    # Remove extra spaces
    text = re.sub(r"\s{2,}", " ", text)
    
    return text

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    text = clean_text(text)
    
    # Remove stopwords
    text = remove_stopwords(text)
    
    return text

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
            TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
            FP += 1
        if y_actual[i]==y_hat[i]==0:
            TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
            FN += 1

    return(TP, FP, TN, FN)


# Reading the required datasets
df = pd.read_csv('/content/consolidated_responses.csv')
train_split = pd.read_csv('/content/train_split_Depression_AVEC2017.csv')
dev_split = pd.read_csv('/content/dev_split_Depression_AVEC2017.csv')
test_split = pd.read_csv('/content/full_test_split.csv')
test_split['PHQ8_Binary'] = test_split['PHQ_Binary']

# Organising the train, val and test sets per the distribution given
train_df = pd.merge(df,train_split,left_on='personId',right_on='Participant_ID')
test_df = pd.merge(df,test_split,left_on='personId',right_on='Participant_ID')
dev_df = pd.merge(df,dev_split,left_on='personId',right_on='Participant_ID')

# Preprocessing the data
train_df['consolidated_response'] = train_df['consolidated_response'].apply(lambda x: preprocess_text(x))
test_df['consolidated_response'] = test_df['consolidated_response'].apply(lambda x: preprocess_text(x))
dev_df['consolidated_response'] = dev_df['consolidated_response'].apply(lambda x: preprocess_text(x))

# Curating the train and test data
X_train = train_df['consolidated_response'].to_list()
Y_train = train_df['PHQ8_Binary'].to_list()
X_val = dev_df['consolidated_response'].to_list()
Y_val = dev_df['PHQ8_Binary'].to_list()
X_test = test_df['consolidated_response'].to_list()
Y_test = test_df['PHQ8_Binary'].to_list()

# Building Model
BATCH_SIZE = 4
MODEL_NAME = 'roberta-large'

t = text.Transformer(MODEL_NAME, maxlen=512, class_names=[0, 1])
trn = t.preprocess_train(X_train, Y_train)
val = t.preprocess_test(X_val, Y_val)

model = t.get_classifier()
learner = ktrain.get_learner(model, train_data=trn, val_data=val, batch_size=BATCH_SIZE)

# Training the model
learner.fit_onecycle(2e-5, 5)

# Getting predictions on test set
predictor = ktrain.get_predictor(learner.model, preproc = t)
predict = predictor.predict(X_test)
learner.validate(val_data = t.preprocess_test(X_test, Y_test))
TP,FP,TN,FN = perf_measure(Y_test, predict)
print(TP,FP,TN,FN)
print("Precision: ", TP/(TP+FP))
print("Recall: ", TP/(TP+FN))
print("Accuracy: ", (TP+TN)/(TP+FP+TN+FN))
print("F1: ", 2*(TP/(TP+FP))*(TP/(TP+FN))/((TP/(TP+FP))+TP/(TP+FN)))
