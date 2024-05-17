# Loading the model details
import transformers
import torch
from csv import writer
import pandas as pd
import numpy as np
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'Intel/neural-chat-7b-v3-3'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model.to(device)


# Preparing the data
phq_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats',
                 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless']
participant_data = pd.read_csv('phq_based_responses.csv', index_col=[0])

participant_dictionary = participant_data.set_index('personId').T.to_dict('list')



# Using the prompt and data to create output
def predict_phq_score(participant_id, phq_questions, top_n_responses, predictions):
    """
    Predicts PHQ score for a participant using the LLM model.
    """
    participant_predictions = []
    for index in range(len(phq_questions)):
        prompt = "This is a set of selected responses from interviews with a participant. If you were a clinician providing a milder diagnosis for the PHQ symptom: " + \
            phq_questions[index] + "\n Participant Response: " + top_n_responses[index] + \
            "\nBased on the level provide ony a single number 0, 1, 2 or 3 as a response. Just provide the number as a response with no explaination"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_new_tokens=190)
        matched_score = re.search(
            r'\d+', tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("\n")[-1])
        if matched_score:  # Check if there's a match
            # Extract the matched group (the digit) and convert to int
            predicted_score = int(matched_score.group())
        else:
            predicted_score = -1
        participant_predictions.append(predicted_score)
    predictions[participant_id] = {'PHQs': participant_predictions, 'Total Score': sum(participant_predictions)}


# Running the Model:
predicted_scores = {}
for participant_id, responses in participant_dictionary.items():
    predict_phq_score(participant_id, phq_questions,
                      responses, predicted_scores)
    row = [participant_id] + predicted_scores[participant_id]['PHQs'] + \
        [predicted_scores[participant_id]['Total Score']]
    with open('output.csv', 'a') as f:
        f_write = writer(f)
        f_write.writerow(row)
