import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.callbacks import ModelCheckpoint
from DataPreprocess import preprocess_data, preprocess_target

label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
              'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
              'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
              'spy.', 'rootkit.']


if __name__ == '__main__':

    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target

    # 數據預處理
    data = preprocess_data(data)

    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = data.shape[1]
    NEW_CLASS = b'smurf.'
    CODES = 2
    # parameter

    input_data_test_set = list()
    input_data_train_set = list()

    for index, each in enumerate(labels[:DATA_LENGTH]):
        if each == NEW_CLASS:
            input_data_test_set.append(data[index])
        else:
            input_data_train_set.append(data[index])

    input_data_test_set = np.array(input_data_test_set)
    input_data_train_set = np.array(input_data_train_set)

    '''
    print('New class: {}'.format(count))
    print('Old class: {}'.format(count_))
    print('Total: {}'.format(DATA_LENGTH))

    # input_data_train_set = data[0:300000]
    # input_data_test_set = data[300000::]
    '''
    
    input_layer = Input(shape=(INPUT_SHAPE,))
    encode_layer = Dense(units=20, activation='tanh')(input_layer)
    code_layer = Dense(units=CODES)(encode_layer)
    decode_layer = Dense(units=20, activation='tanh')(code_layer)
    output_layer = Dense(units=INPUT_SHAPE)(decode_layer)

    autoencoder = Model(input_layer, output_layer)
    
    '''
    autoencoder = Sequential()
    autoencoder.add(Dense(units=20, input_shape=(INPUT_SHAPE,), activation='tanh'))
    autoencoder.add(Dense(units=2, activation='tanh'))
    autoencoder.add(Dense(units=20, activation='tanh'))
    autoencoder.add(Dense(units=41))
    '''
    autoencoder.summary()

    autoencoder.compile(loss='mse', optimizer='adam')
    autoencoder.fit(input_data_train_set, input_data_train_set, verbose=1, epochs=50)
    autoencoder.save('autoencoder.h5')

    loss = autoencoder.evaluate(input_data_test_set, input_data_test_set)
    print("Loss:{}.".format(loss))
