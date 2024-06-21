This details the code efforts carried out from January 2024 to July 2024 for depression detection focused on text information and through the README will try to highlight the different folders and files inside and will illustrate what function they serve.

- DEPTWEET Replication
  
  Replicates the work by Kabir et al, 2023 ([Paper](https://www.sciencedirect.com/science/article/pii/S0747563222003235))
  * `data_preprocess.py: ` Loads a dataset and splits into train, val and test sets
  * `LSTM.py: ` Builds an LSTM Model and calculates AUC Score replicating the work in their paper
  * `BERT.py: ` Uses Transformer's BERT with DEPTWEET and calculated AUC similar to their efforts
  * `DistilBERT.py: ` Uses Transformer's DistilBERT with DEPTWEET and calculated AUC similar to their efforts


- DAIC_WOZ_Replication

  Replicates the work by a student for CS 320 ([Paper](https://cs230.stanford.edu/projects_winter_2019/reports/15762183.pdf))
  * `download_zip_files.py: ` Was a python script created to download all the DAIC WOZ interview zip files through the index page link provided post access
  * `extract_transcript.py: ` Used to extract all DAIC WOZ transcripts from the zip files into another transcripts folder (Useful for all further experimentations) and create a `transcripts_all.csv` as the dataset for this replication
  * `model_transcript.py: ` Used the created transcript file to detect depression with an LSTM model and GloVe embeddings. The code performs basic pre processing before input to the model. It does not use the pre provided train, test and validation split of DAIC WOZ.


- AnxietyDetectionMethodology_Replicated

  Replicates the work by Agarwal et al, 2023 ([Paper](https://arxiv.org/pdf/2312.15272)) but modifies it for depression detection
  * `dataset_analysis.py: ` Used to visualise the sentiment and emotion distribution of the depressed and non depressed classes of the DAIC WOZ dataset with T5 finetuned models.
