#Extracting all transcripts from zip
import fnmatch
import os
import zipfile
from pathlib import Path
import pandas as pd
import re

def extract_files(zip_file, out_dir, delete_zip=False):
    """
    A function takes in a zip file and extracts the *TRANSCRIPT.csv 
    files into separate folders in a user
    specified directory.
    """
    # create transcripts directory
    transcripts_dir = os.path.join(out_dir, 'transcripts')
    if not os.path.exists(transcripts_dir):
        os.makedirs(transcripts_dir)

    zip_ref = zipfile.ZipFile(zip_file)
    for f in zip_ref.namelist():  # iterate through files in zip file
        if fnmatch.fnmatch(f, '*TRANSCRIPT.csv'):
            zip_ref.extract(f, transcripts_dir)
    zip_ref.close()

    if delete_zip:
        os.remove(zip_file)

def transcripts_to_dataframe(directory):
    rows_list = []
        
    filenames = os.listdir(directory)
    
    if ".DS_Store" in filenames:
        filenames.remove(".DS_Store")
        
    for filename in filenames:
        transcript_path = os.path.join(directory, filename)
        transcript = pd.read_csv(transcript_path, sep='\t')
        m = re.search("(\d{3})_TRANSCRIPT.csv", filename)
        if m:
            person_id = m.group(1)
            p = {}
            question = ""
            answer = ""
            lines = len(transcript)
            for i in range(0, lines):
                row = transcript.iloc[i]
                if (row["speaker"] == "Ellie") or (i == lines - 1):
                    p["personId"] = person_id
                    if "(" in str(question):
                        question = question[question.index("(") + 1:question.index(")")]
                    p["question"] = question
                    p["answer"] = answer
                    if question != "":
                        rows_list.append(p)
                    p = {}
                    answer = ""
                    question = row["value"]
                else:
                    answer = str(answer) + " " + str(row["value"])

    all_participants = pd.DataFrame(rows_list, columns=['personId', 'question', 'answer'])
    all_participants.to_csv(directory + 'all.csv', sep=',')
    print("File was created")
    return all_participants


if __name__ == '__main__':
    # directory containing DAIC-WOZ zip files
    dir_name = Path('D:/8th Sem/NTU_Singapore/DAIC_WOZ')

    # directory where audio and transcripts folders will be created
    out_dir = Path('D:/8th Sem/NTU_Singapore/DAIC_WOZ')

    # delete zip file after file wav and csv extraction
    delete_zip = False

    # iterate through zip files in dir_name and extracts wav and transcripts
    for file in os.listdir(dir_name):
        if file.endswith('.zip'):
            zip_file = os.path.join(dir_name, file)
            extract_files(zip_file, out_dir, delete_zip=delete_zip)
    transcripts_to_dataframe('D:/8th Sem/NTU_Singapore/DAIC_WOZ/transcripts')
