import numpy as np
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential, load_model
from keras.losses import kullback_leibler_divergence
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target, train_test_split, confidence_test_split
import csv
import datetime
import os
import errno

if __name__ == '__main__':

    label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
          'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
          'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
          'spy.', 'rootkit.']
    
    # label_list = [normal_list, dos_list, u2r_list, r2l_list, probe_list]
    # label_list = [normal_list, dos_list, u2r_list, r2l_list]

    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target

    # 數據預處理
    data = preprocess_data(data)
    with open('outlier_detection_output.csv', 'w', newline='') as csv_file:
        header = ['Outlier class',
                  'Accuracy of Training data ',
                  'Accuracy of Testing data',
                  'Aware Average',
                  'Outlier Average',
                  ]
        writer = csv.writer(csv_file)
        writer.writerow(header)

        # parameter
        DATA_LENGTH = data.shape[0]
        INPUT_SHAPE = data.shape[1]
        OUTPUT_SHAPE = len(label_list) - 1
        # parameter

        # model configure
        input_layer = Input(shape=(INPUT_SHAPE,))
        x = Dense(units=100, kernel_initializer='random_normal', activation='relu')(input_layer)
        x = Dense(units=50, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(units=OUTPUT_SHAPE, kernel_initializer='random_normal')(x)
        output_layer = Activation('softmax')(x)

        model = Model(input_layer, output_layer)
        weights = model.get_weights()
        confident_machine = Model(input_layer, x)

        model.summary()

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model configure
        for label in label_list:
            copy_list = label_list.copy()
            copy_list.remove(label)
            x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, label_list.copy(), label)
            # x_train, y_train, x_test, y_test = train_test_split(data, labels, label_list)
            model.set_weights(weights)
            model.fit(x_train, y_train, 
                      validation_data=(x_test, y_test), 
                      verbose=1, epochs=50, callbacks=[monitor,])

            _, train_accuracy = model.evaluate(x_train, y_train)
            _, test_accuracy = model.evaluate(x_test, y_test)

            predict_distribution = confident_machine.predict(outlier_set)
            predict_test = confident_machine.predict(x_test)

            outlier_data_length = outlier_set.shape[0]
            outlier_sum_output = 0

            for each in predict_distribution:
                result_index = np.argmax(each)
                outlier_sum_output += each[result_index]

            print("Outlier Average: {}".format(outlier_sum_output / outlier_data_length))

            aware_data_length = x_test.shape[0]
            aware_sum_output = 0

            for each in predict_test:
                result_index = np.argmax(each)
                aware_sum_output += each[result_index]

            print("Aware Average: {}".format(aware_sum_output / aware_data_length))

            writer.writerow([label, 
                             train_accuracy,
                             test_accuracy,
                             aware_sum_output / aware_data_length, 
                             outlier_sum_output / outlier_data_length])
