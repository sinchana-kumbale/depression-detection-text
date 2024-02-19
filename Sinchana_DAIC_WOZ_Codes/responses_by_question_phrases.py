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


def extract_responses_by_question_phrases(transcripts_df: pd.DataFrame, question_phrases: list[str]) -> pd.DataFrame:
    """Extracts consolidated responses for each participant based on given question phrases."""
    person_responses = {}
    
    for phrase in question_phrases:
        # Filter rows containing the phrase
        sub_df = transcripts_df[transcripts_df["question"].str.contains(phrase)]

        for _, row in sub_df.iterrows():
            person_id = row["personId"]
            answer = row["answer"]

            if person_id in person_responses:
                person_responses[person_id] += f"\n{answer}"
            else:
                person_responses[person_id] = answer

    consolidated_responses = pd.DataFrame.from_dict(person_responses, orient="index", columns=["consolidated_response"])
    consolidated_responses["personId"] = consolidated_responses.index
    return consolidated_responses

if __name__ == "__main__":
    data_path = "/kaggle/input/combined-transcripts/transcriptsall.csv"
    question_phrases = [
        "where are you from originally",
        "how are you doing today",
        "when was the last time you argued with someone and what was it about",
        "the last time you felt happy",
        "how are you at controlling your temper",
        "most proud of in your life",
        "for you to get a good night's sleep",
        "have you been diagnosed with depression",
        "how would your best friend describe you",
        "what did you study at school",
        "how have you been feeling lately",
        "anything you regret",
    ]

    transcripts_df = read_clean_dataframe(data_path)
    responses_df = extract_responses_by_question_phrases(transcripts_df, question_phrases)
    responses_df.to_csv('consolidated_responses.csv')
