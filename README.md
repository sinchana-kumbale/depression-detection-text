This details the code efforts carried out from January 2024 to July 2024 for depression detection focused on text information and through the README will try to highlight the different folders and files inside and will illustrate what function they serve.

- DEPTWEET Replication
  
  Replicates the work by Kabir et al, 2023 ([Paper](https://www.sciencedirect.com/science/article/pii/S0747563222003235))
  * `data_preprocess.py: ` Loads a dataset and splits into train, val and test sets
  * `LSTM.py: ` Builds an LSTM Model and calculates AUC Score replicating the work in their paper
  * `BERT.py: ` Uses Transformer's BERT with DEPTWEET and calculated AUC similar to their efforts
  * `DistilBERT.py: ` Uses Transformer's DistilBERT with DEPTWEET and calculated AUC similar to their efforts
