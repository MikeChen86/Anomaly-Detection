import numpy as np
from sklearn.datasets import kddcup99
from keras.models import load_model
from DataPreprocess import preprocess_target, preprocess_data

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
    CODES = 5
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

    autoencoder = load_model('autoencoder.h5')

    autoencoder.summary()

    max_loss = None
    for index in range(count_):
        try:
            loss = autoencoder.evaluate(input_data_train_set[index:index + 1], input_data_train_set[index:index + 1],
                                           verbose=0)
            if not max_loss or max_loss < loss:
                max_loss = loss
        except ValueError:
            print(index)

    print("Max Loss of Training Data:{}".format(max_loss))

    min_loss = None
    for index in range(count):
        try:
            loss = autoencoder.evaluate(input_data_test_set[index:index + 1], input_data_test_set[index:index + 1],
                                           verbose=0)
            if not min_loss or min_loss > loss:
                min_loss = loss
        except ValueError:
            print(index)

    print("Min Loss of Testing Data:{}".format(min_loss))

    loss = autoencoder.evaluate(input_data_train_set, input_data_train_set, verbose=0)
    print("Average Loss of Training Data = {}".format(loss))
    loss = autoencoder.evaluate(input_data_test_set, input_data_test_set, verbose=0)
    print("Average Loss of Testing Data = {}".format(loss))
