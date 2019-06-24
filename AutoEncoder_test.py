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

    # parameter
    DATA_LENGTH = data.shape[0]
    NEW_CLASS = b'neptune.'
    CODES = 5
    THRESHOLD = 26
    # parameter

    input_data_agnostic_set = list()
    input_data_aware_set = list()
    input_data_test_set = list()
    input_data_train_set = list()
    
    for index, each in enumerate(labels[:DATA_LENGTH]):
        if each == NEW_CLASS:
            input_data_agnostic_set.append(data[index])
        else:
            input_data_aware_set.append(data[index])

    input_data_agnostic_set = np.array(input_data_agnostic_set)
    input_data_test_set = np.array(input_data_aware_set[int(len(input_data_aware_set)*4/5):])
    input_data_train_set = np.array(input_data_aware_set[:int(len(input_data_aware_set)*4/5)])
    input_data_aware_set = np.array(input_data_aware_set)

    autoencoder = load_model('autoencoder.h5')

    autoencoder.summary()
    '''
    max_loss = None
    count = 0
    fn_label = [0 for each in label_list]
    for index in range(input_data_test_set.shape[0]):
        try:
            loss = autoencoder.evaluate(input_data_test_set[index:index + 1], input_data_test_set[index:index + 1],
                                           verbose=0)
            if not max_loss or max_loss < loss:
                max_loss = loss
            if loss > THRESHOLD:
                fn_label[label_list.index(str(each, encoding = "utf-8"))] += 1
                count += 1
        except ValueError:
            print(index)
    
    print("False Postive = {}".format(count))
    for index, each in enumerate(fn_label):
        print('{Class}: {number}'.format(Class=label_list[index], number=each))
    
    print("Max Loss of Aware Data = {}".format(max_loss))

    min_loss = None
    for index in range(input_data_agnostic_set.shape[0]):
        try:
            loss = autoencoder.evaluate(input_data_agnostic_set[index:index + 1], input_data_agnostic_set[index:index + 1],
                                           verbose=0)
            if not min_loss or min_loss > loss:
                min_loss = loss
        except ValueError:
            print(index)
    
    print("Min Loss of Agnostic Data = {}".format(min_loss))
    '''
    print("    # Agnostic class is \"{}\" : {}".format(str(NEW_CLASS, encoding = "utf-8"),
                                                   input_data_agnostic_set.shape[0]))
    print("    # Agnostic class is all : {}".format(input_data_aware_set.shape[0]))
    print("    ## Training : {} (80%)".format(input_data_train_set.shape[0]))
    print("    ## Testing : {} (20%)".format(input_data_test_set.shape[0]))

    loss = autoencoder.evaluate(input_data_train_set, input_data_train_set, verbose=0)
    print("    Average Loss of Training Aware Data = {}".format(loss))

    loss = autoencoder.evaluate(input_data_test_set, input_data_test_set, verbose=0)
    print("    Average Loss of Testing Aware Data = {}".format(loss))

    loss = autoencoder.evaluate(input_data_aware_set, input_data_aware_set, verbose=0)
    print("    Average Loss of Aware Data = {}".format(loss))

    loss = autoencoder.evaluate(input_data_agnostic_set, input_data_agnostic_set, verbose=0)
    print("    Average Loss of Agnostic Data = {}".format(loss))
