import numpy as np
from sklearn.datasets import kddcup99
from keras.models import  load_model
from DataPreprocess import preprocess_data, preprocess_target, confidence_test_split
import csv

if __name__ == '__main__':

    label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
          'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
          'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
          'spy.', 'rootkit.']
    
    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target
    label = 'portsweep.'
    # 數據預處理
    data = preprocess_data(data)

    x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, label_list, label)

    model = load_model('classifier.h5')
    print(x_test[:1])
    predict = model.predict(x_test[:1])
    result_index = np.argmax(predict)
    print(label_list[result_index])

    '''
    encoder = load_model('encoder.h5')
    

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

    model = load_model('classifier.h5')
    
    predict_distribution = model.predict(outlier_set)

    predict_test = model.predict(x_test)

    data_length = outlier_set.shape[0]
    sum_output = 0

    for each in predict_distribution:
        result_index = np.argmax(each)
        sum_output += each[result_index]

    print("Outlier Average: {}".format(sum_output / data_length))

    data_length = x_test.shape[0]
    sum_output = 0

    for each in predict_test:
        result_index = np.argmax(each)
        sum_output += each[result_index]

    print("Aware Average: {}".format(sum_output / data_length))

    '''
    '''
    with open('threshold.csv', 'w', newline='') as csv_file:
        header = ['Threshold',
                  'TP', 'FP', 'TN', 'FN',
                  'TPR', 'FPR', 'F1-score'
                 ]
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for threshold in np.arange(0, 0.55, 0.05):
            TP = 0 # Outlier -> Outlier
            FP = 0 # Normal -> Outlier
            TN = 0 # Normal -> Normal
            FN = 0 # Outlier -> Normal

            error_classifier_count = [0 for _ in range(len(label_list))]
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
            writer.writerow([threshold,
                             TP, FP, TN, FN,
                             TPR, FPR, F1])
    

    threshold = 0.5
    best_threshold = 0.5
    learning_rate = 0.05
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
        error_classifier_count = [0 for _ in range(len(label_list))]

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
            threshold -= learning_rate
            best_tp = TP
            best_fn = FN
            best_tn = TN
            best_fp = FP
            best_f1 = F1
        else:       
            flag = False
        print("F1:{}, best_f1:{}, Threshold:{}->{}".format(F1, best_f1, best_threshold, threshold))
    
    
    best_tpr = best_tp / (best_tp + best_fn)
    best_fpr = best_fp / (best_fp + best_tn)
    best_f1 = (2*best_tpr*(1-best_fpr)) / (best_tpr + (1-best_fpr))
    print("Label: {}".format(label))
    print("Threshold: {}".format(best_threshold))
    print("TP: {}, FN: {}, FP: {}, TN:{}.".format(best_tp, best_fn, best_fp, best_tn))
    print("TPR = {:.2f}%, FPR = {:.2f}%.".format(best_tpr*100, best_fpr*100))
    print("F1-score = {:.2f}".format(best_f1))
    for label_name, number in zip(label_list, error_classifier_count):
        print("{}: {}".format(label_name, number))
    
    count = 0
    outlier_predict = model.predict(outlier_set)
    aware_predict = model.predict(x_test)
    outlier_code_list = encoder.predict(outlier_set)
    aware_code_list = encoder.predict(x_test)
    for outlier_code, outlier_distribution in zip(outlier_code_list, outlier_predict):
        for aware_code, aware_distribution in zip(aware_code_list, aware_predict):

            result_index = np.argmax(outlier_distribution)
            x, y = class_avg[result_index]
            outlier_distance = ((x - outlier_code[0])**2 + (y - outlier_code[1])**2)**0.5

            result_index = np.argmax(aware_distribution)
            x, y = class_avg[result_index]
            aware_distance = ((x - aware_code[0])**2 + (y - aware_code[1])**2)**0.5

            if outlier_distance > aware_distance:
                count += 1
    outlier_sample = outlier_set.shape[0]
    aware_sample = x_test.shape[0]
    total_sample = outlier_sample * aware_sample
    print("Label: {}".format(label))
    print("outlier: {}".format(outlier_sample))
    print("aware: {}".format(aware_sample))
    print("Count: {}".format(count))
    print("AUC: {}".format(count / total_sample))
    '''