from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

from Abdullah2021.data_utils import clean_dataset
from eval_tools import evaluate_model
import time


def test_model(PATH, EPOCHS, BATCHSIZE, nrows):
    dataset = clean_dataset(PATH, nrows)

    y = dataset[' Label']
    X = dataset.drop(' Label', axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)

    model = Sequential()
    model.add(Dense(71, input_dim=70, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(50, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adamax', metrics=['accuracy'])

    print("******************************Training Started***************************************")
    print(" X Shape {}, Y shape {}".format(X_train.shape, y_train.shape))
    model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCHSIZE)
    start_time = time.time()
    train_predictions = model.predict(X_train)
    execution_time = time.time() - start_time
    print("Execution time: {}, Per sample: {}".format(execution_time, execution_time / len(X_train)))

    train_predictions = train_predictions.reshape(-1)
    train_predictions[train_predictions > 0.5] = 1
    train_predictions[train_predictions < 0.5] = 0
    train_predictions = train_predictions.astype('int')

    print("*******************************Training Report**************************************")
    evaluate_model(train_predictions, y_train)

    print("Testing")
    print(" X Shape {}, Y shape {}".format(X_test.shape, y_test.shape))

    test_predictions = model.predict(X_test)
    test_predictions = test_predictions.reshape(-1)
    test_predictions[test_predictions > 0.5] = 1
    test_predictions[test_predictions < 0.5] = 0
    test_predictions = test_predictions.astype('int')
    print("Testing Report")
    evaluate_model(test_predictions, y_test)
