import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense
from keras.models import Model, Sequential
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target
import csv
import datetime

if __name__ == '__main__':
    with open('Autoencoder_{}_{}_{}_{}.csv'.format(datetime.datetime.now().date(), 
                                                   datetime.datetime.now().hour, 
                                                   datetime.datetime.now().minute, 
                                                   datetime.datetime.now().second), 'w', newline='') as csv_file:
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

        # parameter
        DATA_LENGTH = data.shape[0]
        INPUT_SHAPE = data.shape[1]
        CODES = 2
        # parameter

        # model configure
        input_layer = Input(shape=(INPUT_SHAPE,))
        encode_layer = Dense(units=20, activation='tanh')(input_layer)
        code_layer = Dense(units=CODES)(encode_layer)
        decode_layer = Dense(units=20, activation='tanh')(code_layer)
        output_layer = Dense(units=INPUT_SHAPE)(decode_layer)

        autoencoder = Model(input_layer, output_layer)

        autoencoder.summary()

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        autoencoder.compile(loss='mse', optimizer='adam')
        # model configure

        for new_class in label_list:
            input_data_agnostic_set = list()
            input_data_aware_set = list()
            input_data_test_set = list()
            input_data_train_set = list()
            
            for index, each in enumerate(labels[:DATA_LENGTH]):
                if each == new_class:
                    input_data_agnostic_set.append(data[index])
                else:
                    input_data_aware_set.append(data[index])

            input_data_agnostic_set = np.array(input_data_agnostic_set)
            input_data_test_set = np.array(input_data_aware_set[int(len(input_data_aware_set)*4/5):])
            input_data_train_set = np.array(input_data_aware_set[:int(len(input_data_aware_set)*4/5)])
            input_data_aware_set = np.array(input_data_aware_set)

            autoencoder.fit(input_data_train_set, input_data_train_set, 
                            validation_data=(input_data_test_set, input_data_test_set), 
                            verbose=0, epochs=50, callbacks=[monitor,])

            new_class = str(new_class, encoding = "utf-8")[:-1]

            autoencoder.save('autoencoder_{}.h5'.format(new_class))

            print("    # Agnostic class is \"{}\" : {}".format(new_class, input_data_agnostic_set.shape[0]))
            print("    # Agnostic class is all : {}".format(input_data_aware_set.shape[0]))
            print("    ## Training : {} (80%)".format(input_data_train_set.shape[0]))
            print("    ## Testing : {} (20%)".format(input_data_test_set.shape[0]))

            avg_train_aware_loss = autoencoder.evaluate(input_data_train_set, input_data_train_set, verbose=0)
            print("    Average Loss of Training Aware Data = {}".format(avg_train_aware_loss))

            avg_test_aware_loss = autoencoder.evaluate(input_data_test_set, input_data_test_set, verbose=0)
            print("    Average Loss of Testing Aware Data = {}".format(avg_test_aware_loss))

            avg_aware_loss = autoencoder.evaluate(input_data_aware_set, input_data_aware_set, verbose=0)
            print("    Average Loss of Aware Data = {}".format(avg_aware_loss))

            avg_agnostic_loss = autoencoder.evaluate(input_data_agnostic_set, input_data_agnostic_set, verbose=0)
            print("    Average Loss of Agnostic Data = {}".format(avg_agnostic_loss))
            print("---------------------------------------------------------")

            writer.writerow([new_class, 
                             input_data_agnostic_set.shape[0],
                             input_data_aware_set.shape[0],
                             input_data_train_set.shape[0],
                             input_data_test_set.shape[0],
                             avg_train_aware_loss,
                             avg_test_aware_loss,
                             avg_aware_loss,
                             avg_agnostic_loss])
