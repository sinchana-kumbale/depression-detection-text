import pandas as pd

def get_participant_responses(dataframe):
    text = ''
    for _,row in dataframe.iterrows():
        if row['speaker'] != 'Ellie':
            text += '. ' + str(row['value'])
    return text

def get_chunckwise_time(dataframe, total_duration):
    chunck1 = ''
    chunck2 = ''
    chunck3 = ''
    first_chunk_threshold = total_duration * 0.2  
    last_chunk_threshold = total_duration * (1 - 0.2) 
    for _, row in dataframe.iterrows():
        time = row['stop_time']
        if time < first_chunk_threshold and row['speaker'] != "Ellie":
            chunck1 += '. ' + str(row['value'])
        elif time >= last_chunk_threshold and row['speaker'] != "Ellie":
            chunck3 += '. ' + str(row['value'])
        elif row['speaker'] != "Ellie":
            chunck2 += '. ' + str(row['value'])
    return chunck1, chunck2, chunck3

if __name__ == '__main__':

    # Extracting list of participants
    train_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/train_split_Depression_AVEC2017.csv')
    test_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/full_test_split.csv')
    dev_df = pd.read_csv('/kaggle/input/test-train-split-daic-woz/dev_split_Depression_AVEC2017.csv')
    participant_ids = list(train_df.Participant_ID.values) + list(dev_df.Participant_ID.values) + list(test_df.Participant_ID.values)
    
    # Fixed length based chuncking
    chunck1 = {}
    chunck2 = {}
    chunck3 = {}
    for participant in participant_ids:
        participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
        total_length = len(participant_transcript)
        chunck1_conv = participant_transcript[:int(0.2*total_length)]
        chunck2_conv = participant_transcript[int(0.2*total_length):int(0.8*total_length)]
        chunck3_conv = participant_transcript[int(0.8*total_length):]
        chunck1[participant] = get_participant_responses(chunck1_conv)
        chunck2[participant] = get_participant_responses(chunck2_conv)
        chunck3[participant] = get_participant_responses(chunck3_conv)
    chuncks = {1:chunck1, 2: chunck2, 3: chunck3}
    for i in range(1,4):
        chunck_responses = pd.DataFrame.from_dict(chuncks[i], orient="index", columns=["chuncked_responses"])
        chunck_responses["personId"] = chunck_responses.index
        chunck_responses.to_csv('chunck' + str(i) +'_length_responses.csv')
    
    # Fixed Time based chuncking
    for participant in participant_ids:
        participant_transcript = pd.read_csv('/kaggle/input/daic-woz-transcripts/' + str(participant) + '_TRANSCRIPT.csv', sep = '\t')
        total_duration = participant_transcript.iloc[len(participant_transcript)-1]['stop_time'] - participant_transcript.iloc[0]['start_time']
        chunck1[participant], chunck2[participant], chunck3[participant] = get_chunckwise_time(participant_transcript, total_duration)
    chuncks = {1:chunck1, 2: chunck2, 3: chunck3}
    for i in range(1,4):
        chunck_responses = pd.DataFrame.from_dict(chuncks[i], orient="index", columns=["chuncked_responses"])
        chunck_responses["personId"] = chunck_responses.index
        chunck_responses.to_csv('chunck' + str(i) +'_time_responses.csv')
