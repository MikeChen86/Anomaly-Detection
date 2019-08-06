import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense
from keras.models import Model, Sequential, load_model
from keras.losses import kullback_leibler_divergence
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target, separate_label
import csv
import datetime
import os
import errno

if __name__ == '__main__':
    dir_name = '{}_{:0>2}_{:0>2}_{:0>2}'.format(datetime.datetime.now().date(), 
                                                   datetime.datetime.now().hour, 
                                                   datetime.datetime.now().minute, 
                                                   datetime.datetime.now().second)
    try:
        os.makedirs(dir_name)
    except OSError as exc: # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise

    with open('{}/Autoencoder.csv'.format(dir_name), 'w', newline='') as csv_file:
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

        normal_list = ['normal.']
        dos_list = ['back.', 'land.', 'neptune.', 'pod.', 'smurf.', 'teardrop.']
        u2r_list = ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.']
        r2l_list = ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.', 'phf.', 'spy.', 'warezmaster.', 'warezclient.']
        probe_list = ['portsweep.', 'ipsweep.', 'satan.', 'nmap.',]

        label_dict = {'normal': normal_list, 
                      'DOS': dos_list, 
                      'U2R': u2r_list,
                      'R2L': r2l_list,
                      'PROBE': probe_list}

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

        # data = separate_label(data, labels, label_list)

        # model configure
        input_layer = Input(shape=(INPUT_SHAPE,))

        encode_layer = Dense(units=25, activation='tanh')(input_layer)
        encode_layer = Dense(units=10, activation='tanh')(encode_layer)

        middle_layer = Dense(units=CODES)(encode_layer)

        decode_layer = Dense(units=10, activation='tanh')(middle_layer)
        decode_layer = Dense(units=25, activation='tanh')(decode_layer)

        output_layer = Dense(units=INPUT_SHAPE)(decode_layer)

        autoencoder = Model(input_layer, output_layer)

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        autoencoder.compile(loss='mse', optimizer='adam')
        # model configure

        for new_class, class_list in label_dict.items():
            agnostic_set = list()
            aware_set = list()
            validation_set = list()
            development_set = list()
            testing_set = list()
            training_set = list()

            for index, each in enumerate(labels):
                if bytes.decode(each) in class_list:
                    agnostic_set.append(data[index])
                else:
                    aware_set.append(data[index])

            training_set = aware_set[:int(len(aware_set)*0.75)]
            validation_set = aware_set[int(len(aware_set)*0.75):]
            development_set = validation_set[:int(len(validation_set)/2)] + agnostic_set[:int(len(validation_set)/2)]
            development_aware_len = int(len(validation_set)/2)
            testing_set = validation_set[int(len(validation_set)/2):] + agnostic_set[int(len(validation_set)/2):]
            
            agnostic_set = np.array(agnostic_set)
            aware_set = np.array(aware_set) 
            validation_set = np.array(validation_set) # For preventing overfitting
            development_set = np.array(development_set) # For tuning the hyperparameters(Threshold)
            testing_set = np.array(testing_set)
            training_set = np.array(training_set)

            '''
            aware_set = list()
            test_set = list()
            train_set = list()
            for label in label_list:
                if label != new_class:
                    train_set.extend(data[label][:int(len(data[label])*0.75)])
                    test_set.extend(data[label][int(len(data[label])*0.75):])
                    aware_set.extend(data[label])
                else:
                    agnostic_set = np.array(data[label])
            train_set = np.array(train_set)
            test_set = np.array(test_set)
            aware_set = np.array(aware_set)
            '''

            autoencoder.fit(training_set, training_set, 
                            validation_data=(validation_set, validation_set), 
                            verbose=0, epochs=50, callbacks=[monitor,])

            autoencoder.save('model/autoencoder_{}.h5'.format(new_class))

            print("    # Agnostic class is \"{}\" : {}".format(new_class, agnostic_set.shape[0]))
            print("    # Aware class is all : {}".format(aware_set.shape[0]))
            print("    ## Training : {} (75%)".format(training_set.shape[0]))
            print("    ## Testing : {} (25%)".format(validation_set.shape[0]))

            avg_train_aware_loss = autoencoder.evaluate(training_set, training_set, verbose=0)
            print("    Average Loss of Training Aware Data = {}".format(avg_train_aware_loss))

            avg_test_aware_loss = autoencoder.evaluate(validation_set, validation_set, verbose=0)
            print("    Average Loss of Testing Aware Data = {}".format(avg_test_aware_loss))

            avg_aware_loss = autoencoder.evaluate(aware_set, aware_set, verbose=0)
            print("    Average Loss of Aware Data = {}".format(avg_aware_loss))

            avg_agnostic_loss = autoencoder.evaluate(agnostic_set, agnostic_set, verbose=0)
            print("    Average Loss of Agnostic Data = {}".format(avg_agnostic_loss))
            print("---------------------------------------------------------")

            writer.writerow([new_class, 
                             agnostic_set.shape[0],
                             aware_set.shape[0],
                             training_set.shape[0],
                             validation_set.shape[0],
                             avg_train_aware_loss,
                             avg_test_aware_loss,
                             avg_aware_loss,
                             avg_agnostic_loss])
