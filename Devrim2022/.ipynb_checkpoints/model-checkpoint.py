from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, Conv1D,ZeroPadding1D
from keras.layers import MaxPooling2D, Dense, Flatten
from keras.layers import Concatenate

from sklearn.model_selection import train_test_split

from eval_tools import evaluate_model
from Devrim2022.data_utils import clean_dataset


def inceptin_block(layer_in):
    import pdb
    #pdb.set_trace()
    conv1 = Conv1D(64, kernel_size=1, padding='same',activation='relu')(layer_in)
    conv3 = Conv1D(64, kernel_size=3, padding='same',activation='relu')(layer_in)
    conv5 = Conv1D(64, kernel_size=3, padding='same',activation='relu')(layer_in)
    output = Concatenate()([conv1, conv3, conv5])
    return output


def test_model(PATH,EPOCHS, BATCHSIZE, nrows):
    
    dataset = clean_dataset(PATH, nrows)
    import pdb
    pdb.set_trace()
    
    y = dataset[' Label']
    X = dataset.drop(' Label', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=11)
    
   
    inputs = Input(shape=(X_train.shape[1],1))
    block1 = inceptin_block(inputs)
    block2 = inceptin_block(block1)
    
    x = Conv1D(filters=64, kernel_size=7, activation='relu')(block2)
    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    
    outputs = Dense(1, activation='sigmoid')(x)
    
    
  
    model = Model(inputs=inputs, outputs=outputs)
    
    
    lr = 0.0003
    
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=EPOCHS, batch_size=64)
    
    
    
    
    
    
    
    
    
