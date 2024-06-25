# Loading the model details
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





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'Intel/neural-chat-7b-v3-3'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model.to(device)


class LoraModel(LoraModel):
  def __init__(self, model, config, task_type="default"):
    super().__init__(model, config, task_type)
    self.linear = nn.Linear(model.config.hidden_size, 2)  # Output size 2 for binary classification
    self.sigmoid = nn.Sigmoid()

  def forward(self, **inputs):
    outputs = super().forward(**inputs)
    logits = self.linear(outputs.last_hidden_state[:, -1, :])  # Use last token output
    predictions = self.sigmoid(logits)
    return predictions

config = LoraConfig(
    task_type="CAUSAL_LM",
    r=16,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=['q_proj','k_proj','v_proj','o_proj']
)
lora_model = LoraModel(model, config, "default")
lora_model.to(device)

# Preparing the data
phq_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats',
                 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless']
participant_data = pd.read_csv('augmented_dataset_phq_based.csv', index_col=[0])
train_df = pd.read_csv('train_split_Depression_AVEC2017.csv')
dev_df = pd.read_csv('dev_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('full_test_split.csv')

train_data = train_df.merge(participant_data, left_on='Participant_ID', right_on='personId')
val_data = dev_df.merge(participant_data, left_on='Participant_ID', right_on='personId')
test_data = test_df.merge(participant_data, left_on='Participant_ID', right_on='personId')

print(len(train_data),len(val_data),len(test_data))
fine_tuning_data = []
label_list = []


# Train data
for index, row in train_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1, 9):
        responses.append(row['phq_response' + str(i)])
    all_responses = '\n'.join(responses)
    label = row['PHQ8_Binary']
    inputs = tokenizer(all_responses, return_tensors="pt").to(device)
    fine_tuning_data.append(inputs)
    label_list.append(label)  # Access label from corresponding PHQ column

# Validation data (optional, for monitoring performance during training)
val_fine_tuning_data = []
val_label_list = []
for index, row in val_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1, 9):
        responses.append(row['phq_response' + str(i)])
    all_responses = '\n'.join(responses)
    label = row['PHQ8_Binary']
    inputs = tokenizer(all_responses, return_tensors="pt").to(device)
    val_fine_tuning_data.append(inputs)
    val_label_list.append(label)  # Access label from corresponding PHQ column


# Fine-tune the Lora model (modified for binary classification)
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=1e-5)  # Adjust learning rate as needed
loss_fct = nn.BCEWithLogitsLoss()
for epoch in range(3):
    for i in range(len(fine_tuning_data)):
        data = fine_tuning_data[i]
        optimizer.zero_grad()
        outputs = lora_model.generate(**data, max_new_tokens=190)
        loss = loss_fct(torch.tensor(outputs[0][0]).to(device).float(), torch.tensor(label_list[i]).to(device).float())
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss}")
    # Evaluate on validation set (optional)
    with torch.no_grad():
        val_loss = 0
        for i in range(len(val_fine_tuning_data)):
            data = val_fine_tuning_data[i]
            outputs = lora_model.generate(**data, max_new_tokens=190)
            val_loss += loss_fct(torch.tensor(outputs[0][0]).to(device).float(), torch.tensor(val_label_list[i]).to(device).float())
        val_loss /= len(val_fine_tuning_data)
        print(f"Epoch: {epoch+1}, Validation Loss: {val_loss}")
    


# Using the prompt and data to create output


def predict_binary_score(participant_id, top_n_responses, predictions):
    """
    Predicts PHQ score for a participant using the LLM model.
    """
    participant_prediction = 0
    # prompt = "If a clinician was providing a diagnosis for  depression, for the selected set of responses most similar to the PHQ questions, " + "\nResponses: " + '. '.join(top_n_responses) + " Please provide a single number: 1 for depressed and 0 for non depressed"
    inputs = tokenizer( '\n'.join(top_n_responses), return_tensors="pt").to(device)
    outputs = lora_model.generate(**inputs,max_new_tokens=190)
    logits = outputs[0][0]
    prediction = torch.sigmoid(logits).item()
    threshold = 0.7
    if prediction > threshold:
        participant_prediction = 1  # Depressed
    else:
        participant_prediction = 0  # Non-depressed
    predictions[participant_id] = {'Binary Prediction': participant_prediction, 'Percentage': prediction}

# Running the Model:
predicted_scores = {}
for index, row in test_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1,9):
        responses.append(row['phq_response'+str(i)])
    predict_binary_score(participant_id, responses, predicted_scores)
    row = [participant_id] + [predicted_scores[participant_id]['Binary Prediction']] + [predicted_scores[participant_id]['Percentage']]
    with open('output_softmax.csv', 'a') as f:
        f_write = writer(f)
        f_write.writerow(row)

