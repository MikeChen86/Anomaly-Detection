from keras.utils import to_categorical
from sklearn.datasets import kddcup99
from sklearn import preprocessing
from simhash import Simhash
import numpy as np
import pandas as pd


def z_score(array):
    data_list = list()
    array = array.astype('float64')
    for each in range(array.shape[1]):
        if each not in [1, 2, 3, 6, 11, 20, 21]:
            data_list.append(preprocessing.scale(array[:, each:each+1]))
        else:
            data_list.append(array[:, each:each+1])
    return np.hstack(data_list)

# 定義kdd99數據預處理函數
def preprocess_data(data_source):
    data = data_source
    for i in range(data_source.shape[0]):
        row = data_source[i]  # 獲取數據
        data[i][1] = handle_protocol(row)  # 將源文件行中3種協議類型轉換成數字標識
        data[i][2] = handle_service(row)  # 將源文件行中70種網絡服務類型轉換成數字標識
        data[i][3] = handle_flag(row)  # 將源文件行中11種網絡連接狀態轉換成數字標識
    data = z_score(data)
    return data


def preprocess_target(target_data_source):
    target = target_data_source
    for i in range(target_data_source.shape[0]):
        row = target_data_source[i]
        # target[i] = handle_label(row)  # 將源文件行中23種攻擊類型轉換成數字標識
        target[i] = detail_handle_label(row)
    return to_categorical(target)


# 定義將源文件行中3種協議類型轉換成數字標識的函數
def handle_protocol(input_data):
    protocol_list = ['tcp', 'udp', 'icmp']
    # tmp = bytes.decode(input_data[1])
    if input_data[1] in protocol_list:
        return protocol_list.index(input_data[1])


# 定義將源文件行中70種網絡服務類型轉換成數字標識的函數
def handle_service(input_data):
    service_list = ['aol', 'auth', 'bgp', 'courier', 'csnet_ns', 'ctf', 'daytime', 'discard', 'domain', 'domain_u',
                    'echo', 'eco_i', 'ecr_i', 'efs', 'exec', 'finger', 'ftp', 'ftp_data', 'gopher', 'harvest',
                    'hostnames',
                    'http', 'http_2784', 'http_443', 'http_8001', 'imap4', 'IRC', 'iso_tsap', 'klogin', 'kshell',
                    'ldap',
                    'link', 'login', 'mtp', 'name', 'netbios_dgm', 'netbios_ns', 'netbios_ssn', 'netstat', 'nnsp',
                    'nntp',
                    'ntp_u', 'other', 'pm_dump', 'pop_2', 'pop_3', 'printer', 'private', 'red_i', 'remote_job', 'rje',
                    'shell',
                    'smtp', 'sql_net', 'ssh', 'sunrpc', 'supdup', 'systat', 'telnet', 'tftp_u', 'tim_i', 'time',
                    'urh_i', 'urp_i',
                    'uucp', 'uucp_path', 'vmnet', 'whois', 'X11', 'Z39_50']
    # tmp = bytes.decode(input_data[2])
    if input_data[2] in service_list:
        return service_list.index(input_data[2])


# 定義將源文件行中11種網絡連接狀態轉換成數字標識的函數
def handle_flag(input_data):
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    # tmp = bytes.decode(input_data[3])

    if input_data[3] in flag_list:
        return flag_list.index(input_data[3])


# 定義將源文件行中攻擊類型轉換成數字標識的函數(訓練集中共出現了22個攻擊類型，而剩下的17種只在測試集中出現)
def handle_label(label):

    normal_list = ['normal']
    dos_list = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']
    u2r_list = ['buffer_overflow', 'loadmodule', 'perl', 'rootkit']
    r2l_list = ['ftp_write', 'guess_passwd', 'imap', 'multihop', 'phf', 'spy', 'warezmaster', 'warezclient']
    probe_list = ['portsweep', 'ipsweep', 'satan', 'nmap',]

    label_list = [normal_list, dos_list, u2r_list, r2l_list, probe_list]

    # tmp = bytes.decode(label)

    for each in label_list:
        if label in each:
            return label_list.index(each)


def detail_handle_label(label):
    label_list = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
              'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write',
              'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient',
              'spy', 'rootkit']
    # tmp = bytes.decode(label)
    for each in label_list:
        if label == each:
            return label_list.index(label)


def separate_label(input_data, labels, label_list):
    data_label_separate = dict()
    for each in label_list:
        data_label_separate[each] = list()

    for index, each in enumerate(labels):
        data_label_separate[each].append(input_data[index])

    return data_label_separate


def train_test_split(input_data, labels, label_list, split=0.75):
    data_label_separate = list()
    for each in range(len(label_list)):
        data_label_separate.append(list())

    for index, label in enumerate(labels):
        # label = bytes.decode(label)
        label_index = label_list.index(label)
        data_label_separate[label_index].append(input_data[index])
        '''
        for label_index, detail_label in enumerate(label_list):
            if label in detail_label:
                data_label_separate[label_index].append(input_data[index])
        '''

    train_x_set = list()
    test_x_set = list()
    train_y_set = list()
    test_y_set = list()
    for index, value in enumerate(data_label_separate):
        temp = np.zeros(len(label_list))
        temp[index] = 1
        value.reverse()
        train_data_length = int(len(value)*0.75)
        train_x_set.extend(value[:train_data_length])
        train_y_set.extend([ temp for _ in range(train_data_length)])
        test_x_set.extend(value[train_data_length:])
        test_y_set.extend([ temp for _ in range(len(value) - train_data_length)])

    train_x_set = np.array(train_x_set)
    test_x_set = np.array(test_x_set)
    train_y_set = np.array(train_y_set)
    test_y_set = np.array(test_y_set)

    return train_x_set, train_y_set, test_x_set, test_y_set


def confidence_test_split(input_data, labels, label_list, outlier):

    data_label_separate = list()

    label_list.remove(outlier)

    for each in range(len(label_list)):
        data_label_separate.append(list())

    outlier_set = list()
    
    for index, label in enumerate(labels):
        # label = bytes.decode(label)
        try:
            label_index = label_list.index(label)
            data_label_separate[label_index].append(input_data[index])
        except ValueError:
            outlier_set.append(input_data[index])

    train_x_set = list() # 75% label-aware data
    test_x_set = list() # 25% label-aware data
    train_y_set = list()
    test_y_set = list()

    for index, value in enumerate(data_label_separate):
        temp = np.zeros(len(label_list))
        temp[index] = 1
        value.reverse()
        train_data_length = int(len(value)*0.75)
        train_x_set.extend(value[:train_data_length])
        train_y_set.extend([ temp for _ in range(train_data_length)])
        test_x_set.extend(value[train_data_length:])
        test_y_set.extend([ temp for _ in range(len(value) - train_data_length)])

    train_x_set = np.array(train_x_set)
    test_x_set = np.array(test_x_set)
    train_y_set = np.array(train_y_set)
    test_y_set = np.array(test_y_set)
    outlier_set = np.array(outlier_set)

    return train_x_set, train_y_set, test_x_set, test_y_set, outlier_set


def convert_to_simhash(input_data):
    list_ = list()
    for each in input_data:
        simhash_value = "{0:b}".format(Simhash([str(feature) for feature in each]).value).zfill(64)
        nparr = np.array([bit for bit in simhash_value]).reshape(8, 8, 1)
        list_.append(nparr)

    return np.array(list_)


if __name__ == '__main__':
    label_list = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
              'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write',
              'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient',
              'spy', 'rootkit']

    df_train = pd.read_csv('dataset/KDDTrain+.txt', header=None)
    df_test = pd.read_csv('dataset/KDDTest+.txt', header=None)
    train_data = df_train.as_matrix()
    test_data = df_test.as_matrix()
    print(len(set(test_data[:, 41])))
    data = np.concatenate([train_data[:, :41], test_data[:, :41]], axis=0)
    labels = np.concatenate([train_data[:, 41:42], test_data[:, 41:42]], axis=0)


    # 數據預處理
    data = preprocess_data(data)
    set_ = set()
    '''
    print(data[0])
    '''
    
    label_count = [0 for each in label_list]

    for label in labels:
        try:
            index = label_list.index(label)
        except:
            set_.add(label[0])
        label_count[index] += 1

    for index, each in enumerate(label_list):
        print('{} : {}'.format(each, label_count[index]))

    print('---------------\n{}'.format(set_))
    '''
    for label in label_list:
        copy_list = label_list.copy()
        copy_list.remove(label)
        x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, label_list.copy(), label)
        print(label, x_test.shape[0], outlier_set.shape[0])
    '''