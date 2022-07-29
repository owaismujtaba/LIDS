from RajKumar2021.data_utils import load_dataset_by_authors
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier

from data_utils import get_file_names, load_files, drop_meaningless_cols
from RajKumar2021.data_utils import clean_dataset, load_dataset_by_authors
from eval_tools import evaluate_model
import time

def test_model(PATH, nrows):
    
    
    dataset = load_dataset_by_authors(PATH, nrows)
    
    import pdb
    pdb.set_trace()
    
    y = dataset[' Label']
    X = dataset.drop(' Label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)
    
    print(" Class Distribution Before SMOTE")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state = 11)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    
    print(" Class Distribution After SMOTE")
    print(y_train.value_counts())
    
    
    GBModel = GradientBoostingClassifier(learning_rate=0.05, n_estimators=200,
                                         max_depth=10 , max_features='sqrt')
    print("*********************************************************************")
    print("Training")
    print(" X Shape {}, Y shape {}".format(X_train.shape, y_train.shape))
    GBModel.fit(X_train, y_train)
    start_time =time.time()
    train_predictions =  GBModel.predict(X_train)
    execution_time = time.time()-start_time
    print("Execution time: {}, Per sample: {}".format(execution_time, execution_time/len(X_train)))
    print("Training Report")
    evaluate_model(train_predictions, y_train)
    
    print("*********************************************************************")
    print("Testing")
    print(" X Shape {}, Y shape {}".format(X_test.shape, y_test.shape))
    
    test_predictions =  GBModel.predict(X_test)
    print("Testing Report")
    evaluate_model(test_predictions, y_test)
    
