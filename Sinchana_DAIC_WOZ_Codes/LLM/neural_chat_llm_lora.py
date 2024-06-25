
import transformers
from peft import LoraConfig, LoraModel
import torch
from csv import writer
import pandas as pd
import numpy as np
import torch.nn as nn
import re
import pickle
import torch.nn.functional as F


# Defining the Model and the LoRA configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'Intel/neural-chat-7b-v3-3'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model.generation_config.pad_token_id = tokenizer.pad_token_id
model.to(device)


config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=['k_proj','v_proj','q_proj', 'o_proj']
)
lora_model = LoraModel(model, config, "default")
lora_model.to(device)

# Preparing the data
phq_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats',
                 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless']
label_desc = ['PHQ8_NoInterest', 'PHQ8_Depressed', 'PHQ8_Sleep', 'PHQ8_Tired', 'PHQ8_Appetite', 'PHQ8_Failure', 'PHQ8_Concentrating', 'PHQ8_Moving']
participant_data = pd.read_csv('augmented_dataset_phq_based.csv', index_col=[0])
train_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
dev_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('full_test_split.csv')

participant_data = pd.read_csv('augmented_dataset_phq_based.csv')
train_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
dev_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('full_test_split.csv')

train_data = train_df.merge(participant_data, left_on='Participant_ID', right_on='personId')
val_data = dev_df.merge(participant_data, left_on='Participant_ID', right_on='personId')
test_data = test_df.merge(participant_data, left_on='Participant_ID', right_on='personId')

test_data.drop(test_data.filter(regex="Unname"),axis=1, inplace=True)
train_data.drop(train_data.filter(regex="Unname"),axis=1, inplace=True)
val_data.drop(val_data.filter(regex="Unname"),axis=1, inplace=True)

train_data.dropna(inplace=True)
print(len(train_data),len(val_data),len(test_data))

fine_tuning_data = []
label_list = []

# Train data
for index, row in train_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1, 9):
        responses.append(row['phq_response' + str(i)])
    for i in range(len(phq_questions)):
        prompt = "If a clinician was providing a diagnosis for the PHQ symptom: " + phq_questions[i] + " for the response: " + responses[i] + " based on the level (0,1,2,3) you would provide the label " + str(row[label_desc[i]])  # Access label from corresponding PHQ column
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        fine_tuning_data.append(inputs)
        label_list.append(row[label_desc[i]])  # Access label from corresponding PHQ column


# Validation data 
val_fine_tuning_data = []
val_label_list = []
for index, row in val_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1, 9):
        responses.append(row['phq_response' + str(i)])
    for i in range(len(phq_questions)):
        prompt = "If a clinician was providing a diagnosis for the PHQ symptom: " + phq_questions[i] + " for the response: " + responses[i] + " based on the level (0,1,2,3) you would provide the label " # Access label from corresponding PHQ column
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        val_fine_tuning_data.append(inputs)
        val_label_list.append(row[label_desc[i]])  # Access label from corresponding PHQ column


# Fine-tune the Lora model
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-5)  # Adjust learning rate as needed
loss_fct = nn.CrossEntropyLoss()
for epoch in range(3):
    for i in range(len(fine_tuning_data)):
        data = fine_tuning_data[i]
        optimizer.zero_grad()
        outputs = lora_model.generate(**data, max_new_tokens=190)
        outputs = F.softmax(outputs.float(),dim=-1)
        loss = loss_fct(outputs,torch.tensor(label_list[i]).unsqueeze(0).to(outputs.device).long())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss}")
    if val_fine_tuning_data:
        val_loss = 0
        for i in range(len(val_fine_tuning_data)):
            data = val_fine_tuning_data[i]
            outputs = lora_model.generate(**data, max_new_tokens=190)
            outputs = F.softmax(outputs.float(),dim=-1)
            val_loss += loss_fct(outputs,torch.tensor(val_label_list[i]).unsqueeze(0).to(outputs.device).long()).item()
        val_loss /= len(val_fine_tuning_data)
        print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}")

    

# Using the prompt and data to create output
def predict_phq_score(participant_id, top_n_responses, predictions):
    """
    Predicts PHQ score for a participant using the LLM model.
    """
    participant_predictions = []
    for index in range(len(phq_questions)):
        prompt = "If you were a clinician providing a diagnosis for the PHQ symptom: " + \
            phq_questions[index] + "\n For the Participant Response: " + top_n_responses[index] + "\nBased on the level provide only a single number 0, 1, 2 or 3 as a response and ensure mild diagnosis without overestimation"
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = lora_model.generate(**inputs, max_new_tokens=190)
        matched_score = re.search(r'\d+', tokenizer.decode(outputs[0], skip_special_tokens=True).strip().split("\n")[-1])
        if matched_score:  # Check if there's a match
            # Extract the matched group (the digit) and convert to int
            predicted_score = int(matched_score.group())
        else:
            predicted_score = 0
        participant_predictions.append(predicted_score)
    predictions[participant_id] = {'PHQs': participant_predictions, 'Total Score': sum(participant_predictions)}




# Running the Model:
predicted_scores = {}
for index, row in test_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1,9):
        responses.append(row['phq_response'+str(i)])
    predict_phq_score(participant_id, responses, predicted_scores)
    row = [participant_id] + predicted_scores[participant_id]['PHQs'] + [predicted_scores[participant_id]['Total Score']]
    with open('output_lora.csv', 'a') as f:
        f_write = writer(f)
        f_write.writerow(row)
