import warnings
import numpy as np
import pandas as pd
from data_utils import drop_meaningless_cols, get_file_names, load_files


def clean_dataset(PATH, nrows):
    """
    The function deos the following
    1. removes the duplicate and NaN rows 
    2. replace inf, with -1
    3. replace negative values with 0
    4. change label (object type) column to numerice binary class 0 normal 1 attack
    Args:
        dataset: pandas dataset
    Return:
        dataset: cleaned dataset
    """
    print("****************************Cleaning Dataset********************************************")
    
    meaning_less_cols = ['Unnamed: 0', 'Flow ID',' Timestamp', ' Source IP', 
                    'SimillarHTTP', ' Source Port', ' Destination IP', ' Destination Port',
                        ' Bwd PSH Flags',' Fwd URG Flags',' Bwd URG Flags', 
                         'Fwd Avg Bytes/Bulk',' Fwd Avg Packets/Bulk',' Fwd Avg Bulk Rate',
                       ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk',  'Bwd Avg Bulk Rate']
    file_names= get_file_names(PATH)    
    
    dataset = load_files(PATH, file_names, nrows)
    
    dataset.drop_duplicates(inplace=True)
    dataset = drop_meaningless_cols(dataset, meaning_less_cols)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        dataset.dropna(axis=0, inplace=True)
        for column in dataset.columns:
            dataset[column] = dataset[column].replace([np.inf, -np.inf], -1)
  
    
    
    dataset = normalize_0_1(dataset)

    print("1. Duplicates and NaN samples Removed")
    print("2. inf values replaced with -1")
    print("3. Normalize Values between 0 and 1")
    
   
    print("****************************Cleaning Dataset Completed********************************************")
    import pdb
    pdb.set_trace()
    print("Dataset Shape:", dataset.shape)
    
    return dataset



def normalize_0_1(dataset):
    from sklearn.preprocessing import MinMaxScaler
    
    dataset1 = dataset.copy()
    scaler = MinMaxScaler()
    dataset1 = scaler.fit_transform(dataset1)
    
    dataset = pd.DataFrame(dataset1, columns=dataset.columns)
    
    
    return dataset



