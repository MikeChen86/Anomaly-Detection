import numpy as np
from sklearn.datasets import kddcup99
from keras.models import load_model
from DataPreprocess import preprocess_data, preprocess_target, train_test_split, confidence_test_split
import csv
from scipy.stats import rankdata

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
    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = data.shape[1]
    OUTPUT_SHAPE = len(label_list) - 1
    threshold = 0.35
    # parameter
    # 數據預處理
    data = preprocess_data(data)
    with open('auc.csv', 'w', newline='') as csv_file:
        header = ['Outlier class',
                  'Accuracy of Training data ',
                  'Accuracy of Testing data',
                  'Number of outlier sample',
                  'Number of aware sample',
                  'AUC',
                  ]
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for label in label_list:
            copy_list = label_list.copy()
            x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, copy_list, label)
            
            model = load_model('model/classifier_{}.h5'.format(label[:-1]))
            encoder = load_model('model/encoder_{}.h5'.format(label[:-1]))
            
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

            print("Label: {}".format(label))
            print("outlier: {}".format(outlier_sample))
            print("aware: {}".format(aware_sample))

            outlier_predict = model.predict(outlier_set)
            aware_predict = model.predict(x_test)
            outlier_code_list = encoder.predict(outlier_set)
            aware_code_list = encoder.predict(x_test)
            outlier_l1_norm = list()
            aware_l1_norm = list()

            for outlier_code, outlier_distribution in zip(outlier_code_list, outlier_predict):
                result_index = np.argmax(outlier_distribution)
                x, y = class_avg[result_index]
                outlier_distance = abs(x - outlier_code[0]) + abs(y - outlier_code[1])
                outlier_l1_norm.append(outlier_distance)

            for aware_code, aware_distribution in zip(aware_code_list, aware_predict):
                result_index = np.argmax(aware_distribution)
                x, y = class_avg[result_index]
                aware_distance = abs(x - aware_code[0]) + abs(y - aware_code[1])
                aware_l1_norm.append(aware_distance)

            total_l1_norm = aware_l1_norm + outlier_l1_norm
            total_l1_norm = rankdata([-1 * i for i in total_l1_norm])
            
            rank_sum = np.sum(total_l1_norm[:aware_sample])
            auc = (rank_sum - (aware_sample*(aware_sample+1)/2)) / (aware_sample*outlier_sample)

            print("AUC: {}".format(auc))
            writer.writerow([label,
                             train_accuracy,
                             test_accuracy,
                             outlier_sample, aware_sample, auc])