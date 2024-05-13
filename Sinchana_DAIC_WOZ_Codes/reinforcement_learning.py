import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import gym
import numpy as np


from transformers import AutoTokenizer, RobertaModel
tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-base")
model = RobertaModel.from_pretrained("FacebookAI/roberta-base")

# Defining Keywords related to positive, negative or depressive emotions
pos_word_list = []
neg_word_list = []
dep_word_list = []
with open('/kaggle/input/word-distribution/positive-words.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        pos_word_list.append(str(line).strip())
    #print(pos_word_list)
with open('/kaggle/input/word-distribution/negative-words.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        neg_word_list.append(str(line).strip())
with open('/kaggle/input/word-distribution/depressedword.txt','r',encoding='utf-8') as f:
    for line in f.readlines():
        if len(line.split())==2:
            for ele in line.split()[1].split(','): dep_word_list.append(ele)
        else:dep_word_list.append(str(line).strip())
keywords = pos_word_list + neg_word_list + dep_word_list

# Defines the classifier using RoBERTa
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.roberta = model
        self.fc = nn.Linear(768, 1)  # Output layer

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        output = torch.sigmoid(self.fc(pooled_output))  # Sigmoid activation for binary classification
        return output

# Defines the environment
class DepressionDetectionEnv(gym.Env):
    def __init__(self, participants, responses, labels):
        super(DepressionDetectionEnv, self).__init__()
        self.participants = participants
        self.responses = responses
        self.labels = labels
        self.action_space = gym.spaces.MultiDiscrete([max(1, len(responses[0]) // 2)] * len(responses[0]))
        self.observation_space = gym.spaces.Discrete(len(participants))  # Observing participants
        self.classifier = Classifier()
        self.optimizer = optim.Adam(self.classifier.parameters(), lr=0.001)
        self.current_participant = 0
        self.previous_accuracy = 0.50
        self.selected_responses = {participant: [] for participant in participants}  # Track selected responses per participant
    
    def reset(self):
        self.current_participant = 0
        return 0  # Start with the first participant
    
    def step(self, action):
        if np.all(action == 0):
            action += 1
        action = np.asarray(action)
        response_pool = list(range(len(self.responses[self.current_participant])))
        selected_indices = np.random.choice(response_pool, size=min(len(response_pool), np.sum(action)), replace=False)
        
        selected_responses = [self.responses[self.current_participant][i] for i in selected_indices]
        self.selected_responses[self.participants[self.current_participant]] += selected_responses
        if len(selected_responses)<=0:
            print("Here")
        input_ids, attention_mask = convert_list_of_text_to_features(selected_responses)
        output = self.classifier(input_ids, attention_mask)
        reward = self.calculate_reward(output)
        self.current_participant += 1
        if self.current_participant < len(self.participants):
            self.action_space = gym.spaces.MultiDiscrete([max(1, len(responses[self.current_participant]) // 2)] * len(responses[self.current_participant]))
        done = self.current_participant >= len(self.participants)
        return self.current_participant, reward, done, {}  # Return next state, reward, done, info
    
    def calculate_reward(self, output):
        current_accuracy = self.evaluate_classifier()
        lexical_diversity = self.evaluate_lexical_diversity()
        keyword_matches = self.evaluate_keyword_matches()
        readability_score = self.evaluate_readability()
        reward = (current_accuracy - self.previous_accuracy) + lexical_diversity + keyword_matches + readability_score
        self.previous_accuracy = current_accuracy
        return reward
    
    def evaluate_classifier(self):
        # Function to evaluate classifier accuracy
        correct = 0
        total = len(self.labels)
        for participant, label in zip(self.participants, self.labels):
            selected_responses = self.selected_responses[participant] 
            input_ids, attention_mask = convert_list_of_text_to_features(selected_responses)
            output = self.classifier(input_ids, attention_mask)
            prediction = (output > 0.5).float()  # Thresholding for binary classification
            correct += torch.sum(prediction == label).item()
        accuracy = correct / total
        return accuracy
        
    def evaluate_lexical_diversity(self):
        total_diversity = 0
        overall_count = len(self.labels)
        for participant, label in zip(self.participants, self.labels):
            selected_responses = self.selected_responses[participant]
            selected_responses = '. '.join(selected_responses)
            if len(selected_responses) == 0:
                continue
            tokens = nltk.word_tokenize(selected_responses)
            vocab_size = len(set(tokens))
            total_diversity += vocab_size / len(tokens)
        return total_diversity/overall_count
    
    def evaluate_keyword_matches(self):
        total_keyword_matches = 0
        overall_count = len(self.labels)
        for participant, label in zip(self.participants, self.labels):
            selected_responses = self.selected_responses[participant] 
            selected_responses = '. '.join(selected_responses)
            if len(selected_responses) == 0:
                continue
            total_keyword_matches += (sum(keyword in selected_responses.lower() for keyword in keywords)/len(selected_responses))
        return total_keyword_matches/overall_count
    
    def evaluate_readability(self):
        total_readability = 0
        overall_count = len(self.labels)
        for participant, label in zip(self.participants, self.labels):
            selected_responses = self.selected_responses[participant] 
            selected_responses = '. '.join(selected_responses)
            if len(selected_responses) > 0:
                total_readability += readability.getmeasures(selected_responses, lang='en')['readability grades']['FleschReadingEase']
        return total_readability/overall_count
# Defines the learning agent
class QLearningAgent:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state, env):
        max_selectable = min(env.action_space.nvec)  # Max number of selectable responses
        if np.random.rand() < self.epsilon:  # Explore: Choose a random number of responses (ensure at least 1)
            num_responses = np.random.randint(1, max_selectable + 1)
            return np.random.choice(env.action_space.nvec, size=num_responses, replace=False)
        else:
        # Exploit: Choose actions iteratively with separate rewards
            q_values = self.q_table[state]
            best_actions = []
            max_q = -np.inf
            for i in range(max_selectable):  # Iterate through possible number of selections
                action_mask = np.zeros_like(q_values)
                action_mask[:i + 1] = 1  # Mask out actions beyond i selections (but keep the first)
                q_with_mask = q_values * action_mask
                best_idx = np.argmax(q_with_mask)
                if q_with_mask[best_idx] > max_q:  # Found a better action with significant Q-value difference
                    max_q = q_with_mask[best_idx]
                    best_actions = [best_idx]
                elif q_with_mask[best_idx] == max_q:  # Multiple actions with same Q-value (avoid unnecessary iterations)
                    best_actions.append(best_idx)
                    break  # Early stop if multiple actions have the same max Q-value
            return np.random.choice(best_actions)  # Choose from actions with highest Q-value
    def update(self, state, action, reward, next_state):
        action = min(np.sum(action), self.action_size - 1)  # Clip action to max index
        next_state = min(next_state, self.state_size - 1)
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.discount_factor * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error

# Function to convert text data into input IDs and attention masks
def convert_text_to_features(text):
    text = str(text)
    text = text[:1500]
    input_ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"]
    attention_mask = input_ids.ne(-100)  # Attention mask for padding tokens
    return input_ids, attention_mask

def convert_list_of_text_to_features(list_of_text):
    text = '. '.join(list_of_text)
    return convert_text_to_features(text)


if __name__ == '__main__':

    # Creating the list of participants and labels
    train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
    test_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv')
    dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')
    participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)
    partcipant_labels = list(train_df.PHQ8_Binary.values) + list(dev_df.PHQ8_Binary.values) + list(test_df.PHQ_Binary.values)
  
    # Creating a list of responses for each participant
    participant_responses = []
    for participant in participant_ids:
        participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
        participant_transcript = participant_transcript[participant_transcript['speaker']!='Ellie']
        participant_transcript = participant_transcript.dropna()
        participant_responses.append(list(participant_transcript['value'].values))
    
    # Initialize participants, responses, and labels
    participants = participant_ids # List of participants
    responses = participant_responses # List of lists of responses corresponding to each participant
    labels = partcipant_labels  # List of labels for each participant (0 or 1)
    
    # Initialize environment
    env = DepressionDetectionEnv(participants, responses, labels)
    max_response_length = max(len(response) for response in responses)
    agent = QLearningAgent(state_size=len(participants), action_size=max_response_length)
    num_episodes = 10
    episode_rewards = []
    best_reward = float('-inf')
    early_stopping_count = 0
    early_stopping_window = 2
    
    # Train RL agent
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.select_action(state, env)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
        episode_rewards.append(episode_reward)
        if episode_rewards[-early_stopping_window:] == [best_reward] * early_stopping_window:
            early_stopping_count += 1
        else:
            early_stopping_count = 0
            best_reward = max(best_reward, episode_reward)
        
        if early_stopping_count >= early_stopping_window:
            print(f"Early stopping triggered after {episode} episodes")
            break
        selected_responses_per_participant = env.selected_responses
  
    # Curating responses for each participant
    participant_selected_responses = {}
    for participant, responses in selected_responses_per_participant.items():
        participant_selected_responses[participant] = '. '.join(responses)
    # Writing the responses to a csv file
    reinforcement_responses = pd.DataFrame.from_dict(participant_selected_responses, orient="index", columns=["reinforcement_responses"])
    reinforcement_responses["personId"] = reinforcement_responses.index
    reinforcement_responses.to_csv('reinforcement_responses.csv')


