import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-mpnet-base-v2')

def get_top_similar_responses(participant_id, participant_responses, diagnostic_questions, n):
    """
    Finds the top n most similar responses to each question for a participant.
    """
    participant_embeddings = model.encode(participant_responses[participant_id])  # Encode participant responses
    question_embeddings = model.encode(diagnostic_questions)  # Encode diagnostic questions
    
    top_n_similar_responses = []
    for i, question_embedding in enumerate(question_embeddings):
        # Calculate cosine similarities between each question embedding and all participant responses
        similarities = util.pytorch_cos_sim(question_embedding.reshape(1, -1), participant_embeddings)
        
        # Sort participant responses by similarity (descending order)
        sorted_similar_indices = torch.argsort(similarities, dim=1, descending=True)
        top_n_indices = sorted_similar_indices[:, :n] 
        # Select top n most similar participant responses
        top_n_similar_responses.append([participant_responses[participant_id][idx] for idx in top_n_indices.flatten().tolist()])
    return top_n_similar_responses

phq_8_questions = ['have interest or pleasure in doing things', 'feel down, depressed or hopeless', 'have trouble staying or falling asleep or sleeping for too long', 'feel tired or have little energy', 'have poor appetite or overeats', 'feel bad about themself, that they are a failure or have let themself or family down', 'trouble concentrating on things like school work, reading or watching television', 'move or speak more slowly than usual or be very fidgety and restless' ]

# Extracting participant details
participant_responses = {}
train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
test_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv')
dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')
participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)
for participant in participant_ids:
    participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
    participant_transcript = participant_transcript[participant_transcript['speaker'] != 'Ellie']
    participant_responses[participant] = list(participant_transcript['value'].values)

# Curating top responses for each question for each participant
top_responses = {}
for participant in participant_ids:
    list_of_responses = get_top_similar_responses(participant, participant_responses, phq_8_questions, 5)
    top_responses[participant] = ['. '.join(each_phq_response) for each_phq_response in list_of_responses]

# Writing out results to file
phq_based_responses = pd.DataFrame.from_dict(top_responses, orient="index", columns=["phq_response1", "phq_response2", "phq_response3", "phq_response4", "phq_response5", "phq_response6", "phq_response7", "phq_response8"])
phq_based_responses["personId"] = phq_based_responses.index
phq_based_responses.to_csv('phq_based_responses.csv')
