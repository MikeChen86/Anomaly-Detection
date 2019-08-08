import numpy as np
import pandas as pd
from sklearn.datasets import kddcup99
from keras.models import load_model
from DataPreprocess import preprocess_data, preprocess_target, confidence_test_split
import csv
from scipy.stats import rankdata
from classifier import RobustSoftmax

if __name__ == '__main__':

    label_list = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
                  'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write',
                  'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient',
                  'spy', 'rootkit']
    
    df_train = pd.read_csv('dataset/KDDTrain+.txt', header=None)
    df_test = pd.read_csv('dataset/KDDTest+.txt', header=None)
    train_data = df_train.as_matrix()
    test_data = df_test.as_matrix()
    data = np.concatenate([train_data[:, :41], test_data[:, :41]], axis=0)
    labels = np.concatenate([train_data[:, 41:42], test_data[:, 41:42]], axis=0)
    label = 'buffer_overflow'
    # 數據預處理
    data = preprocess_data(data)

    x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, label_list, label)

    encoder = load_model('encoder.h5')

    '''
    class_count = [0 for _ in label_list]
    class_avg = [[0, 0] for _ in label_list]
    coordinate_list = encoder.predict(x_train)
    for coordinate, data_label in zip(coordinate_list, y_train):
        result_index = np.argmax(data_label)
        class_count[result_index] += 1
        class_avg[result_index][0] += coordinate[0]
        class_avg[result_index][1] += coordinate[1]
    for index in range(len(label_list)):
        class_avg[index][0] /= class_count[index]
        class_avg[index][1] /= class_count[index]
    '''

    model = load_model('classifier.h5', custom_objects={'RobustSoftmax': RobustSoftmax})
    
    outlier_sample = outlier_set.shape[0]
    aware_sample = x_test.shape[0]
    outlier_predict = model.predict(outlier_set)
    aware_predict = model.predict(x_test)
    label_count = [0 for _ in label_list]
    aware_count = [0 for _ in label_list]
    TP = FP = TN = FN = 0
    for outlier_distribution in outlier_predict:
        result_index = np.argmax(outlier_distribution)
        confidence = outlier_distribution[result_index]
        if confidence < 0.9:
            TP += 1
        else:
            FN += 1
            label_count[result_index] += 1

    for aware_distribution in aware_predict:
        result_index = np.argmax(aware_distribution)
        confidence = aware_distribution[result_index]
        if confidence >= 0.9:
            TN += 1
        else:
            FP += 1
            aware_count[result_index] += 1

    print("TPR:", TP/(TP+FN), "FPR:", FP/(TN+FP))

    '''
    outlier_sample = outlier_set.shape[0]
    aware_sample = x_test.shape[0]
    outlier_predict = model.predict(outlier_set)
    aware_predict = model.predict(x_test)
    outlier_code_list = encoder.predict(outlier_set)
    aware_code_list = encoder.predict(x_test)
    outlier_l1_norm = list()
    aware_l1_norm = list()
    label_count = [0 for _ in label_list]
    for outlier_code, outlier_distribution in zip(outlier_code_list, outlier_predict):
        result_index = np.argmax(outlier_distribution)
        x, y = class_avg[result_index]
        label_count[result_index] += 1
        outlier_distance = abs(x - outlier_code[0]) + abs(y - outlier_code[1])
        outlier_l1_norm.append(outlier_distance)

    for aware_code, aware_distribution in zip(aware_code_list, aware_predict):
        result_index = np.argmax(aware_distribution)
        x, y = class_avg[result_index]
        aware_distance = abs(x - aware_code[0]) + abs(y - aware_code[1])
        aware_l1_norm.append(aware_distance)
    
    outlier_sample = outlier_set.shape[0]
    aware_sample = x_test.shape[0]
    outlier_predict = model.predict(outlier_set)
    aware_predict = model.predict(x_test)
    outlier_confidences = list()
    aware_confidences = list()
    label_count = [0 for _ in label_list]
    aware_count = [0 for _ in label_list]
    for outlier_distribution in outlier_predict:
        result_index = np.argmax(outlier_distribution)
        confidence = outlier_distribution[result_index]
        if confidence > 0.9:
            label_count[result_index] += 1
        outlier_confidences.append(confidence)

    for aware_distribution in aware_predict:
        result_index = np.argmax(aware_distribution)
        confidence = aware_distribution[result_index]
        if confidence < 0.9:
            aware_count[result_index] += 1
        aware_confidences.append(confidence)

    total_l1_norm = aware_confidences + outlier_confidences
    # total_l1_norm = rankdata([-1 * i for i in total_l1_norm])
    total_l1_norm = rankdata([i for i in total_l1_norm])
    rank_sum = np.sum(total_l1_norm[:aware_sample])
    auc = (rank_sum - (aware_sample*(aware_sample+1)/2)) / (aware_sample*outlier_sample)
    print("Aware sample:", aware_sample)
    print("Outlier sample:", outlier_sample)
    print("AUC:", auc)
    for each, count_outlier, count_aware in zip(label_list, label_count, aware_count):
        print(each, count_outlier, count_aware)
    '''