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

torch.manual_seed(42)


def get_response_dict(response_string):
	responses = response_string.split("\nResponse: ")
	result_dict = {}
	for response in responses[1:]:
		response_parts = response.split(" Label: ")
		response_text = response_parts[0]
		label = int(response_parts[1])
		# Store the response text in the dictionary with the label as the key
		result_dict[label] = response_text
	return result_dict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
model_name = 'Intel/neural-chat-7b-v3-3'
model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
model.to(device)


config = LoraConfig(
    task_type="CAUSAL_LM",
    r=4,
    lora_alpha=16,
    lora_dropout=0.01,
    target_modules=['q_proj','k_proj','v_proj','o_proj']
)
lora_model = LoraModel(model, config, "default")
lora_model.to(device)

# Preparing the data
phq_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats',
                 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless']
participant_data = pd.read_csv('augmented_dataset_phq_similarity.csv', index_col=[0])

# participant_dictionary = participant_data.set_index('personId').T.to_dict('list')
with open ('fewshotexamples', 'rb') as fp:
    few_shot_examples = pickle.load(fp)

fine_tuning_data = []
label_list = []
# LoRa with fewshotexamples
for i in range(len(few_shot_examples)):
	response_dictionary = get_response_dict(few_shot_examples[i])
	for label, response in response_dictionary.items():
		prompt = "If a clinician was providing a diagnosis for the PHQ symptom: " + phq_questions[i] + " for the response: " + response + " based on the level (0,1,2,3) you would provide the label " + label
		inputs = tokenizer(prompt, return_tensors="pt").to(device)
		fine_tuning_data.append(inputs)
		label_list.append(label)

# Fine-tune the Lora model
optimizer = torch.optim.AdamW(lora_model.parameters(), lr=2e-5)  # Adjust learning rate as needed
loss_fct = nn.CrossEntropyLoss()
for epoch in range(3):
    for i in range(len(fine_tuning_data)):
        data = fine_tuning_data[i]
        print(data)
        optimizer.zero_grad()
        outputs = lora_model.generate(**data, max_new_tokens=190)
        outputs = F.softmax(outputs.float(),dim=-1)
        loss = loss_fct(outputs,torch.tensor(label_list[i]).unsqueeze(0).to(outputs.device))
        loss.requires_grad = True
        loss.backward()
        optimizer.step()
    print(f"Epoch: {epoch+1}, Loss: {loss}")


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
for index, row in participant_data.iterrows():
    participant_id = row['personId']
    responses = []
    for i in range(1,9):
        responses.append(row['phq_response'+str(i)])
    predict_phq_score(participant_id, responses, predicted_scores)
    row = [participant_id] + predicted_scores[participant_id]['PHQs'] + [predicted_scores[participant_id]['Total Score']]
    with open('output.csv', 'a') as f:
        f_write = writer(f)
        f_write.writerow(row)
