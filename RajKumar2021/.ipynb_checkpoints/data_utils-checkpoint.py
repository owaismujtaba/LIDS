import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold
import warnings
from imblearn.over_sampling import SMOTE 

from data_utils import load_files, get_file_names, drop_meaningless_cols, drop_constant_features, drop_duplicate_features




"""
Done
    1. Deletion of unnecessary features
    2. Deletion of Null value records
    3. Deletion of Duplicate records
    4. inf with median
    5. -ve with 0
    6. Label encode Protocol

Remarks
    1. Data Balencing cannot be done befor data cleaning, The whole flow of preprocessing is wrong
        it should be done at the end after preprocessing of all steps
    2. Extra Featurs obtained after preprocessing and cleaning
    {'Bwd IAT Total', 'Bwd Packet Length Max', 'Flow Bytes/s', ' SYN Flag Count', ' Total Fwd Packets', 'Fwd PSH Flags', ' Bwd IAT Min', ' Bwd Packets/s', ' Init_Win_bytes_backward', ' Down/Up Ratio', ' Fwd Packet Length Std', 'Active Mean', ' Active Std', ' min_seg_size_forward'}
"""





    
def clean_dataset(PATH, nrows):
    
    print("*******************************************************************************")
    print("*****************************Cleaning Dataset**********************************")
    
    meaning_less_cols = ['Unnamed: 0', 'Flow ID',' Timestamp', ' Source IP', 
                    'SimillarHTTP', ' Source Port', ' Destination IP', ' Destination Port']
    
    
    filenames = get_file_names(PATH)
    
    dataset = load_files(PATH, filenames, nrows)
       
    dataset = drop_meaningless_cols(dataset, meaning_less_cols)
    dataset.dropna(inplace=True)
    dataset.drop_duplicates(inplace=True)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        for column in set(dataset.columns)-set([' Label']):
            dataset[column] = dataset[column].replace([np.inf, -np.inf], dataset[column].median())
    
    dataset[dataset < 0] = 0
    
    encoder = LabelEncoder()
    dataset[' Protocol'] = encoder.fit_transform(dataset[' Protocol'])
    dataset[' Label'] = encoder.fit_transform(dataset[' Label'])
    dataset = scaler(dataset)
    dataset = drop_constant_features(dataset)
    dataset = drop_quasi_constant_features(dataset)
    dataset = drop_duplicate_features(dataset)
    dataset = drop_high_correlated_features(dataset, 0.8)
    
    print("*******************************************************************************")
    print("*******************************Cleaning Dataset Done***************************")

    return dataset

def balance_dataset(dataset):
    """
    The function returns the balanced dataset using SMOTE
    Args:
        dataset: pandas dataset
    return:
        dataset: pandas blanced dataset
    """
    df = dataset.copy()
    y = df[' Label']
    df.drop(' Label', axis=1, inplace=True)
    
    smote = SMOTE(random_state=42)
    X, y = smote.fit_resample(df, y)
    
    
    
    pdb.set_trace()

def scaler(dataset):
    """
    Scales the dataset (x-u)/s
    Args:
        dataset: pandas dataframe
    return:
        datsaet: pandas data frame
    """
  
    print("*******************************Scaling Dataset**************************************")
    scaler = StandardScaler()
    dataset = dataset.reset_index()
    dataset.drop('index', axis=1, inplace=True)
    
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis = 1, inplace = True)
    features = dataset1.columns
    
    
    dataset1 = scaler.fit_transform(dataset1)
    dataset1 = pd.DataFrame(dataset1, columns=features)
    
    dataset1[' Label'] = dataset[' Label']
   
    
    print("**********************************Scaling Dataset Completed*****************************************")
    
    return dataset1



def drop_quasi_constant_features(dataset):
    """
    Removes the Quasi Constant Features
    Args:
        dataset: Pandas dataframe
    Return:
        dataset: withoud quasi constant features
    """
    print("******************************Droping Qasi Constant Features*********************************")
    dataset1 = dataset.copy()
    selector = VarianceThreshold(0.01)
    quasi = selector.fit(dataset1)
    
    features = quasi.get_feature_names_out()
    
    quasi_constant_features = set(dataset.columns)- set(features)
    dataset = dataset[features]
    
    
    i =1
    for item in quasi_constant_features:
        print("{}. {}".format(i, item))
        i += 1
    
    print("Dataset Shape:", dataset.shape)
    
    print("******************************Droped Qasi Constant Features*********************************")
    return dataset






def drop_high_correlated_features(dataset, cofficent):
    """
    RDrops highly correlated features in datafame
    Args:
        dataset: pandas dataframe
    Return:
        Dataframe
    """
    
    print("******************************Drop Highly Coorelated Features***************************************")
    dataset1 = dataset.copy()
    dataset1.drop(' Label', axis=1, inplace=True)

    cor_matrix = dataset1.corr().abs()
    upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

    highly_correlated = [column for column in upper_tri.columns if any(upper_tri[column] > cofficent)]

    dataset.drop(highly_correlated, axis=1, inplace=True)

    print("Highly Correlated Features Dropped")
    for i in range(len(highly_correlated)):
        print("{}. {}".format(i + 1, highly_correlated[i]))

    print("Dataset Shape:", dataset.shape)
    print("******************************Droped Highly Coorelated Features***************************************")

    return dataset




def features_by_authors(dataset):
    """
    Extracts only the features suggested by the authors which arein total8
    Args:
        dataset: pandas dataframe
    return:
        dataset: pandas dataframe
    """

    features = [' Protocol', ' Total Backward Packets', 'Total Length of Fwd Packets', 
                'Total Length of Fwd Packets',' Flow IAT Min', ' URG Flag Count', 
                'Init_Win_bytes_forward', ' Inbound', ' Label', ' Bwd Packet Length Min']
    
    
    dataset = dataset[features]

    return dataset








def load_dataset_by_authors(PATH, nrows):
    
    print("**************************loadind datast by authors*******************************************")
    
    
    dataset = clean_dataset(PATH, nrows)
    
    dataset = features_by_authors(dataset)
    print("Dataset Shape:", dataset.shape)
    
    return dataset
    
    




