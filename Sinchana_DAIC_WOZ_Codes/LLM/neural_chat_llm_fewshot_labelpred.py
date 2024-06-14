# Loading the model details
import transformers
import torch
from csv import writer
import pandas as pd
import numpy as np
import torch.nn as nn
import re
import pickle
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'Intel/neural-chat-7b-v3-3'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model.to(device)


# Preparing the data
phq_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats',
                 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless']
participant_data = pd.read_csv('augmented_dataset_phq_similarity.csv', index_col=[0])

# participant_dictionary = participant_data.set_index('personId').T.to_dict('list')
with open ('fewshotexamples', 'rb') as fp:
    few_shot_examples = pickle.load(fp)

# Using the prompt and data to create output


def predict_phq_score(participant_id, top_n_responses, predictions):
    """
    Predicts PHQ score for a participant using the LLM model.
    """
    participant_predictions = []
    for index in range(len(phq_questions)):
        prompt = "If you were a clinician providing a diagnosis for the PHQ symptom: " + \
            phq_questions[index] + " where some example responses and labels include " + few_shot_examples[index] + "\nWhere 0 represents hardly any symptoms 1 represents mild symptoms 2 represents moderate symptoms and 3 represents severe symptoms"  + "\n For the Participant Response: " + top_n_responses[index] + "\nBased on the level provide only a single word None, Mild, Moderate or Severe as a response and ensure mild diagnosis without overestimation"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=190)
        matched_score = re.search(r'(None|Mild|Moderate|Severe)', tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("\n")[-1])
        if matched_score:  # Check if there's a match
            # Extract the matched group
            predicted_score = matched_score.group()
        else:
            predicted_score = 'None'
        participant_predictions.append(predicted_score)
    prediction_matches = {'None':0, 'Mild':1, 'Moderate':2, 'Severe':3}
    participant_predictions_scores = [prediction_matches[prediction] for prediction in participant_predictions]
    predictions[participant_id] = {'PHQs': participant_predictions, 'Total Score': sum(participant_predictions_scores)}


# Running the Model:
predicted_scores = {}
for index, row in participant_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1,9):
        responses.append(row['phq_response'+str(i)])
    predict_phq_score(participant_id, responses, predicted_scores)
    row = [participant_id] + predicted_scores[participant_id]['PHQs'] + [predicted_scores[participant_id]['Total Score']]
    with open('output_labels.csv', 'a') as f:
        f_write = writer(f)
        f_write.writerow(row)
