import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense
from keras.models import Model
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
    labels = preprocess_target(labels)  # 進行OneHot編碼

    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = 41
    ENCODE_UNITS = 20
    NEW_CLASS = label_list.index('guess_passwd.')
    CODES = 2
    # parameter

    count = 0
    count_ = 0
    for each in labels[:DATA_LENGTH]:
        if each[NEW_CLASS] == 1:
            count = count + 1
        elif each[0] == 1:
            count_ = count_ + 1

    input_data_test_set = np.zeros(shape=(count, 41))
    input_data_train_set = np.zeros(shape=(count_, 41))
    count = 0
    count_ = 0
    for index, each in enumerate(labels[:DATA_LENGTH]):
        if each[NEW_CLASS] == 1:
            input_data_test_set[count] = data[index]
            count = count + 1
        elif each[0] == 1:
            input_data_train_set[count_] = data[index]
            count_ = count_ + 1
    '''
    print('New class: {}'.format(count))
    print('Old class: {}'.format(count_))
    print('Total: {}'.format(DATA_LENGTH))

    # input_data_train_set = data[0:300000]
    # input_data_test_set = data[300000::]
    '''
    input_layer = Input(shape=(INPUT_SHAPE,))
    encode_layer = Dense(units=ENCODE_UNITS, activation='tanh')(input_layer)
    code_layer = Dense(units=CODES, activation='tanh')(encode_layer)
    decode_layer = Dense(units=ENCODE_UNITS, activation='tanh')(code_layer)
    output_layer = Dense(units=INPUT_SHAPE, activation='tanh')(decode_layer)

    autoencoder = Model(input_layer, output_layer)

    autoencoder.summary()

    autoencoder.compile(loss='mean_squared_error', optimizer='adam')
    autoencoder.fit(input_data_train_set, input_data_train_set, verbose=1, epochs=50)
    
    autoencoder.save('autoencoder.h5')

    loss = autoencoder.evaluate(input_data_test_set, input_data_test_set)
    print("Loss:{}.".format(loss))
