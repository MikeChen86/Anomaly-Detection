import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras.losses import kullback_leibler_divergence
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target, separate_label
import csv
import datetime

if __name__ == '__main__':
    
    header = ['Agnostic Class',
              'Number of Agnostic Class',
              'Number of Aware Class',
              'Number of Training Data',
              'Number of Testing Data',
              'Training Data Average Loss',
              'Testing Data Average Loss',
              'Aware Class Average Loss',
              'Agnostic Class Average Loss']
    writer = csv.writer(csv_file)
    writer.writerow(header)

    label_list = [b'normal.', b'buffer_overflow.', b'loadmodule.', b'perl.', b'neptune.', b'smurf.',
                      b'guess_passwd.', b'pod.', b'teardrop.', b'portsweep.', b'ipsweep.', b'land.', b'ftp_write.',
                      b'back.', b'imap.', b'satan.', b'phf.', b'nmap.', b'multihop.', b'warezmaster.', b'warezclient.',
                      b'spy.', b'rootkit.']

    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target

    # 數據預處理
    data = preprocess_data(data)


    # data = separate_label(data, labels, label_list)

    agnostic_set = dict()
    aware_set = dict()
    test_set = dict()

    for new_class in label_list:
        agnostic_set[new_class] = list()
        aware_set[new_class] = list()
        test_set[new_class] = list()

        for index, each in enumerate(labels):
            if each != new_class:
                aware_set[new_class].append(data[index])
            else:
                agnostic_set[new_class].append(data[index])

        test_set[new_class] = np.array(aware_set[new_class][int(len(aware_set[new_class])*0.75):].extend(agnostic_set[new_class]))

    for new_class in label_list:
        
        new_class = str(new_class, encoding = "utf-8")[:-1]
        autoencoder = load_model('model/autoencoder_{}.h5'.format(new_class))
