import pandas as pd
import numpy as np


def read_clean_dataframe(data_path: str) -> pd.DataFrame:
    """Reads and cleans the combined transcripts dataframe from a CSV file."""
    df = pd.read_csv(data_path)
    
    # Remove all trailing and leading spaces
    df["question"] = df["question"].str.strip()
    
    # Filter out rows with missing questions or answers
    df.dropna(subset=["question", "answer"], inplace=True)
    
    return df


def extract_responses_by_question_phrases(transcripts_df: pd.DataFrame, question_phrases: list[str], follow_up_starts: list[str], follow_up_contains: list[str]) -> pd.DataFrame:
    """Extracts consolidated responses for each participant based on given question phrases."""
    person_responses = {}
    person_ids = set(transcripts_df.personId.values)
    
    for person_id in person_ids:
        person_df = transcripts_df[transcripts_df['personId'] == person_id]
        prev_question_matched = False
        consolidated_text = ""

        for _, row in person_df.iterrows():
            question = row["question"].lower()
            answer = row["answer"]
            if prev_question_matched:
                if any(question.startswith(phrase) for phrase in follow_up_starts):
                    consolidated_text += answer
                elif any(phrase in question for phrase in follow_up_contains):
                    consolidated_text += answer
                else:
                    prev_question_matched = False
                    
            elif any(phrase in question for phrase in question_phrases):
                prev_question_matched = True
                consolidated_text = consolidated_text + "\n" + question + " - " + answer
            else:
                prev_question_matched = False
        
        person_responses[person_id] = consolidated_text

        
    
    

    consolidated_responses = pd.DataFrame.from_dict(person_responses, orient="index", columns=["consolidated_response"])
    consolidated_responses["personId"] = consolidated_responses.index
    return consolidated_responses

if __name__ == "__main__":
    data_path = "/kaggle/input/combined-transcripts/transcriptsall.csv"
    question_phrases = [
        "how are you doing today",
        "last time you argued",
        "the last time you felt really happy",
        "controlling your temper",
        "to get a good night's sleep",
        "diagnosed with depression",
        "feeling lately",
        "anything you regret",
        "any changes in your behavior"
        
    ]
    follow_up_starts = ['mhm','yeah','why','really','uh huh','yes','hmm','okay']
    follow_up_contains = ['tell me','give me an example','i\'m sorry','sounds really hard','do you feel that way often','do you feel down','how do you cope','could you have done anything','how did you feel']

    transcripts_df = read_clean_dataframe(data_path)
    responses_df = extract_responses_by_question_phrases(transcripts_df, question_phrases, follow_up_starts, follow_up_contains)
    responses_df.to_csv('consolidated_responses.csv')
