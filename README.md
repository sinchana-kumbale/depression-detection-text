I will try to provide a description of the folders and codes in them to help replicate the results
- DEPTWEET Replication
  
  Replicates the work by Kabir et al, 2023 ([Paper](https://www.sciencedirect.com/science/article/pii/S0747563222003235))
  * `data_preprocess.py: ` Loads a dataset and splits into train, val and test sets
  * `LSTM.py: ` Builds an LSTM Model and calculates AUC Score replicating the work in their paper
  * `BERT.py: ` Uses Transformer's BERT with DEPTWEET and calculated AUC similar to their efforts
  * `DistilBERT.py: ` Uses Transformer's DistilBERT with DEPTWEET and calculated AUC similar to their efforts
