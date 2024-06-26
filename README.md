
## Code Organisation and Description

This details the code efforts carried out from January 2024 to July 2024 for depression detection focused on text information and through the README will try to highlight the different folders and files inside and will illustrate what function they serve.


- **AnxietyDetectionMethodology_Replicated**

  Replicates the work by Agarwal et al, 2023 ([Paper](https://arxiv.org/pdf/2312.15272)) but modifies it for depression detection
  * `dataset_analysis.py: ` Used to visualise the sentiment and emotion distribution of the depressed and non depressed classes of the DAIC WOZ dataset with T5 finetuned models.
  * `extract_features.py: ` Used to load the DAIC WOZ transcripts, apply basic preprocessing, use roberta to encode and extract features
  * `train_evaluate_models.py: ` Uses the encoded tokens and features and splits into the default train, test and validation, upsamples to balance the classes, uses SVM classifier and reports metrics


- **DAIC_WOZ_Replication**

  Replicates the work by a student for CS 320 ([Paper](https://cs230.stanford.edu/projects_winter_2019/reports/15762183.pdf))
  * `download_zip_files.py: ` Was a python script created to download all the DAIC WOZ interview zip files through the index page link provided post access
  * `extract_transcript.py: ` Used to extract all DAIC WOZ transcripts from the zip files into another transcripts folder (Useful for all further experimentations) and create a `transcripts_all.csv` as the dataset for this replication
  * `model_transcript.py: ` Used the created transcript file to detect depression with an LSTM model and GloVe embeddings. The code performs basic pre processing before input to the model. It does not use the pre provided train, test and validation split of DAIC WOZ.



- **DEPTWEET Replication**
  
  Replicates the work by Kabir et al, 2023 ([Paper](https://www.sciencedirect.com/science/article/pii/S0747563222003235))
  * `data_preprocess.py: ` Loads a dataset and splits into train, val and test sets
  * `LSTM.py: ` Builds an LSTM Model and calculates AUC Score replicating the work in their paper
  * `BERT.py: ` Uses Transformer's BERT with DEPTWEET and calculated AUC similar to their efforts
  * `DistilBERT.py: ` Uses Transformer's DistilBERT with DEPTWEET and calculated AUC similar to their efforts


- **Sinchana DAIC WOZ Codes**

  Contains all self written codes and experimentations with the DAIC WOZ dataset
  - **LLM**

    Contains all LLM related experimentations (When running on NSCC, ngpus:2 and walltime about 15 hours were selected)
    * `create_fewshot_examples.py: ` Selecting examples from the train set, one of each type as few shot data to feed to the LLM prompts
    * `neural_chat_llm_fewshot.py: ` Predicts PHQ scores for each participant, for every PHQ question using a prompt with the few shot examples and returns an output csv file that contains participant ids, their individual PHQ question scores and overall PHQ Score
    * `neural_chat_llm_fewshot_labelpred.py: ` Predicts a PHQ label (None, Mild, Moderate, Severe) for each participant, for every PHQ question using a prompt with the few shot examples and returns an output csv file that contains participant ids, their individual PHQ question labels and overall PHQ Score
    * `neural_chat_llm_lora.py: ` Uses the DAIC WOZ predefined train and development sets to finetune a LoRA that is further used to provide an output csv file containing predicted PHQ scores for individual paticipants and questions and the overall PHQ Score
    * `neural_chat_llm_lora_binary.py: ` Uses the DAIC WOZ predefined train and development sets to finetune a LoRA that is further used to provide an output csv file containing participant ids and binary depression predictions
    * `neural_chat_llm_zeroshot.py: ` Predicts PHQ scores for each participant, for every PHQ question using a prompt with no prior examples and returns an output csv file that contains participant ids, their individual PHQ question scores and overall PHQ Score
    * `phq_similarity_response_selection.py: ` Uses the all-mpnet based model to create embeddings of the PHQ symptoms and participant responses to curate top n responses most similar to each symptom for each participant.
  <br>

  * `chunck_wise_data.py: ` Used to split the overall DAIC WOZ transcripts into 3 sections - the warm up, core of the interview and the cool down based on the length of the transcripts and the time difference
  * `data_augmentation.py: ` Implementation of the topic modelling based data augmentation ([Ref](https://arxiv.org/pdf/1803.10384)) where n random combination of m topics shuffled from participants who have more than m topics are created
  * `general_data_augmentation.py: ` Implements EDA with synonym replacement, random insertion, random swap and random deletion as a general augmentation method for DAIC WOZ
  * `linguistic_analysis.py: ` Used to analyse word counts for depressive, negative, positive keywords, personal pronoun usage, absolutist words and other linguistic attributes and visualise the distribution across the two classes using a UMAP.
  * `non_neutral_responses.py: ` Used to extract specific subset participant responses that are displaying either positive or negative sentiment as calculated through TextBlob. The idea is to focus on parts of the interview that might be not neural responses but are emotionally charged
  * `pretrained_transformer_baseline.py: ` Using the DAIC WOZ split followed by preprocessing to build and measure performance of a RoBERTa based transformer model with some specific hyper parameters
  * `reinforcement_learning.py: ` Uses a Q Learning agent within the environment with a reward as a combination of increase in RoBERTa accuracy, lexical diversity, keyword matches and readability, trained with early stopping
  * `responses_by_question_phrases.py: ` Selects a set of participant responses based on specifically selected 8 emotionally relevant questions most asked by Ellie. Accounts for follow up questions asking the participant to elaborate by considering the phrases used to follow up.
  * `topic_based_responses.py: ` Creates an LDA topic model with the selected topic number (29) and a subset of selected topics based on the coherence threshold (0.655) observed through experimentation. The participant responses belonging to any of the selected topics are included


## Other Project Related Links
* Drive Containing some relevant documents: [Link](https://learnermanipal-my.sharepoint.com/:f:/g/personal/sinchana_kumbale_learner_manipal_edu/EjpDsmTNaNxDrpyT3HbBnXkBhlFf6aUw4VOzcjccE4abgQ?e=AvbgLD)
* Folder with all presentations: [Link](https://learnermanipal-my.sharepoint.com/:f:/g/personal/sinchana_kumbale_learner_manipal_edu/EhwfX4V-Ne5JlElcyLHwxPEB1fyZyHJdrRJ_y--cNcStYA?e=gfaU4Z?)
* Notion page with excerpts of papers read: [Link](https://thread-drawbridge-64a.notion.site/NTU-Research-Internship-cf4e439cea4f4541a247ef763e8494fc?pvs=4)
* Document with flow of work (kind of disorganised): [Link](https://learnermanipal-my.sharepoint.com/:w:/g/personal/sinchana_kumbale_learner_manipal_edu/EXDdZkBi9NlJu-x73_0M2OwBU7EFIQcxHs_vrX3uJTQUhQ?e=8E3kgt)

