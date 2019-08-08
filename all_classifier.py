import numpy as np
import pandas as pd
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense, Dropout, Activation
from keras.models import Model, Sequential, load_model
from keras.losses import kullback_leibler_divergence
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target, train_test_split, confidence_test_split
import csv
from classifier import RobustSoftmax

if __name__ == '__main__':

    label_list = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
                  'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write',
                  'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient',
                  'spy', 'rootkit']

    # label_list = [normal_list, dos_list, u2r_list, r2l_list, probe_list]
    # label_list = [normal_list, dos_list, u2r_list, r2l_list]

    df_train = pd.read_csv('dataset/KDDTrain+.txt', header=None)
    df_test = pd.read_csv('dataset/KDDTest+.txt', header=None)
    train_data = df_train.as_matrix()
    test_data = df_test.as_matrix()
    data = np.concatenate([train_data[:, :41], test_data[:, :41]], axis=0)
    labels = np.concatenate([train_data[:, 41:42], test_data[:, 41:42]], axis=0)
    print(data.shape)
    '''
    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target
    '''
    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = data.shape[1]
    OUTPUT_SHAPE = len(label_list) - 1
    threshold = 0.35
    # parameter
    # 數據預處理
    data = preprocess_data(data)
    with open('outlier_detection.csv', 'w', newline='') as csv_file:
        header = ['Outlier class',
                  'Accuracy of Training data ',
                  'Accuracy of Testing data',
                  'Number of outlier sample',
                  'Number of aware sample',
                  'Number of sample',
                  'Count',
                  'AUC',
                  ]
        writer = csv.writer(csv_file)
        writer.writerow(header)

        monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, verbose=1, mode='auto')

        # Build Autoencoder
        ae_input_layer = Input(shape=(INPUT_SHAPE,))
        encode_layer = Dense(units=100, kernel_initializer='random_normal', activation='tanh')(ae_input_layer)
        encode_layer = Dense(units=50, kernel_initializer='random_normal', activation='tanh')(encode_layer)
        encode_layer = Dense(units=25, kernel_initializer='random_normal', activation='tanh')(encode_layer)
        encode_layer = Dense(units=10, kernel_initializer='random_normal', activation='tanh')(encode_layer)
        ae_code = Dense(units=2, kernel_initializer='random_normal')(encode_layer)
        decode_layer = Dense(units=10, kernel_initializer='random_normal', activation='tanh')(ae_code)
        decode_layer = Dense(units=25, kernel_initializer='random_normal', activation='tanh')(decode_layer)
        decode_layer = Dense(units=50, kernel_initializer='random_normal', activation='tanh')(decode_layer)
        decode_layer = Dense(units=100, kernel_initializer='random_normal', activation='tanh')(decode_layer)
        ae_output_layer = Dense(units=41, kernel_initializer='random_normal')(decode_layer)

        autoencoder = Model(ae_input_layer, ae_output_layer)

        ae_weights = autoencoder.get_weights()

        input_layer = Dense(units=200, kernel_initializer='random_normal', activation='relu')(ae_code)
        x = Dense(units=100, kernel_initializer='random_normal', activation='relu')(input_layer)
        x = Dense(units=50, kernel_initializer='random_normal', activation='relu')(x)
        x = Dense(units=OUTPUT_SHAPE, kernel_initializer='random_normal')(x)
        output_layer = RobustSoftmax()(x)

        model = Model(ae_input_layer, output_layer)
        model_weights = model.get_weights()
        # Train Autoencoder

        for label in label_list:
            copy_list = label_list.copy()
            x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, copy_list, label)
            # x_train, y_train, x_test, y_test = train_test_split(data, labels, label_list)

            autoencoder.set_weights(ae_weights)
            for layer_index in range(3):
                autoencoder.get_layer(index=layer_index).trainable = True

            autoencoder.compile(loss='mse', optimizer='adam')
            autoencoder.fit(x_train, x_train,
                            validation_data=(x_test, x_test),
                            verbose=1, epochs=50, callbacks=[monitor, ])
            sigmoid = Activation('sigmoid')(ae_code)
            encoder = Model(ae_input_layer, sigmoid)
            # model configure
            model.set_weights(model_weights)
            for layer_index in range(3):
                model.get_layer(index=layer_index).trainable = False
            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            # model configure

            model.fit(x_train, y_train,
                      validation_data=(x_test, y_test),
                      verbose=1, epochs=50, callbacks=[monitor, ])

            _, train_accuracy = model.evaluate(x_train, y_train)
            _, test_accuracy = model.evaluate(x_test, y_test)

            class_count = [0 for _ in copy_list]
            class_avg = [[0, 0] for _ in copy_list]
            coordinate_list = encoder.predict(x_train)
            for coordinate, data_label in zip(coordinate_list, y_train):
                result_index = np.argmax(data_label)
                class_count[result_index] += 1
                class_avg[result_index][0] += coordinate[0]
                class_avg[result_index][1] += coordinate[1]
            for index in range(len(copy_list)):
                class_avg[index][0] /= class_count[index]
                class_avg[index][1] /= class_count[index]

            count = 0
            
            outlier_sample = outlier_set.shape[0]
            aware_sample = x_test.shape[0]
            total_sample = outlier_sample * aware_sample
            auc = count / total_sample
            print("Label: {}".format(label))
            print("outlier: {}".format(outlier_sample))
            print("aware: {}".format(aware_sample))

            outlier_predict = model.predict(outlier_set)
            aware_predict = model.predict(x_test)
            outlier_code_list = encoder.predict(outlier_set)
            aware_code_list = encoder.predict(x_test)
            for outlier_code, outlier_distribution in zip(outlier_code_list, outlier_predict):
                for aware_code, aware_distribution in zip(aware_code_list, aware_predict):

                    result_index = np.argmax(outlier_distribution)
                    x, y = class_avg[result_index]
                    outlier_distance = abs(x - outlier_code[0]) + abs(y - outlier_code[1])

                    result_index = np.argmax(aware_distribution)
                    x, y = class_avg[result_index]
                    aware_distance = abs(x - aware_code[0])**2 + abs(y - aware_code[1])

                    if outlier_distance > aware_distance:
                        count += 1
            
            print("Count: {}".format(count))
            print("AUC: {}".format(auc))
            writer.writerow([label,
                             train_accuracy,
                             test_accuracy,
                             outlier_sample, aware_sample,
                             total_sample, count, auc])
            '''
            threshold = 0.5
            best_threshold = 0.5
            best_tp = 0
            best_fp = 0
            best_tn = 0
            best_fn = 0
            best_f1 = 0
            flag = True
            while flag:
                TP = 0 # Outlier -> Outlier
                FP = 0 # Normal -> Outlier
                TN = 0 # Normal -> Normal
                FN = 0 # Outlier -> Normal
                error_classifier_count = [0 for _ in range(len(copy_list))]

                predict_distribution = model.predict(outlier_set)
                code_list = encoder.predict(outlier_set)
                for code, each in zip(code_list, predict_distribution):
                    result_index = np.argmax(each)
                    x, y = class_avg[result_index]
                    distance = ((x - code[0])**2 + (y - code[1])**2)**0.5

                    if distance > threshold:
                        TP += 1
                    else:
                        FN += 1
                        error_classifier_count[result_index] += 1

                predict_distribution = model.predict(x_test)
                code_list = encoder.predict(x_test)
                for code, each in zip(code_list, predict_distribution):
                    result_index = np.argmax(each)
                    x, y = class_avg[result_index]
                    distance = ((x - code[0])**2 + (y - code[1])**2)**0.5

                    if distance < threshold:
                        TN += 1
                    else:
                        FP += 1

                TPR = TP / (TP + FN)
                FPR = FP / (FP + TN)
                F1 = (2*TPR*(1-FPR)) / (TPR + (1-FPR))
                if F1 >= best_f1:
                    best_threshold = threshold
                    threshold -= 0.05
                    best_tp = TP
                    best_fn = FN
                    best_tn = TN
                    best_fp = FP
                    best_f1 = F1
                else:
                    flag = False

            best_tpr = best_tp / (best_tp + best_fn)
            best_fpr = best_fp / (best_fp + best_tn)

            print("Label: {}".format(label))
            print("Training: {:.2f}%, Testing:{:.2f}%.".format(train_accuracy * 100, test_accuracy * 100))
            print("TP: {}, FN: {}, FP: {}, TN:{}.".format(best_tp, best_fn, best_fp, best_tn))
            print("TPR = {:.2f}%, FPR = {:.2f}%.".format(best_tpr * 100, best_fpr * 100))

            for label_name, number in zip(copy_list, error_classifier_count):
                print("{}: {}".format(label_name, number))
            print("=================================")
            writer.writerow([label,
                             train_accuracy,
                             test_accuracy,
                             best_tp, best_fp, best_tn, best_fn,
                             best_tpr, best_fpr, best_threshold])
            '''
