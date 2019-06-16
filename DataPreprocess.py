from keras.utils import to_categorical
from sklearn.datasets import kddcup99


# 定義kdd99數據預處理函數
def preprocess_data(data_source):
    data = data_source
    for i in range(data_source.shape[0]):
        row = data_source[i]  # 獲取數據
        data[i][1] = handle_protocol(row)  # 將源文件行中3種協議類型轉換成數字標識
        data[i][2] = handle_service(row)  # 將源文件行中70種網絡服務類型轉換成數字標識
        data[i][3] = handle_flag(row)  # 將源文件行中11種網絡連接狀態轉換成數字標識
    return data


def preprocess_target(target_data_source):
    target = target_data_source
    for i in range(target_data_source.shape[0]):
        row = target_data_source[i]
        target[i] = handle_label(row)  # 將源文件行中23種攻擊類型轉換成數字標識
    return to_categorical(target)


# 定義將源文件行中3種協議類型轉換成數字標識的函數
def handle_protocol(input_data):
    protocol_list = ['tcp', 'udp', 'icmp']
    tmp = bytes.decode(input_data[1])
    if tmp in protocol_list:
        return protocol_list.index(tmp)


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
    tmp = bytes.decode(input_data[2])
    if tmp in service_list:
        return service_list.index(tmp)


# 定義將源文件行中11種網絡連接狀態轉換成數字標識的函數
def handle_flag(input_data):
    flag_list = ['OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH']
    tmp = bytes.decode(input_data[3])
    if tmp in flag_list:
        return flag_list.index(tmp)


# 定義將源文件行中攻擊類型轉換成數字標識的函數(訓練集中共出現了22個攻擊類型，而剩下的17種只在測試集中出現)
def handle_label(label):
    label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
                  'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
                  'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
                  'spy.', 'rootkit.']
    tmp = bytes.decode(label)
    if tmp in label_list:
        return label_list.index(tmp)


if __name__ == '__main__':
    label_list = ['normal.', 'buffer_overflow.', 'loadmodule.', 'perl.', 'neptune.', 'smurf.',
              'guess_passwd.', 'pod.', 'teardrop.', 'portsweep.', 'ipsweep.', 'land.', 'ftp_write.',
              'back.', 'imap.', 'satan.', 'phf.', 'nmap.', 'multihop.', 'warezmaster.', 'warezclient.',
              'spy.', 'rootkit.']

    dataset = kddcup99.fetch_kddcup99()
    labels = dataset.target
    # 數據預處理

    label_count = [0 for each in label_list]

    for each in labels:
        index = label_list.index(str(each, encoding = "utf-8"))
        label_count[index] = label_count[index] + 1

    for index, each in enumerate(label_list):
        print('{} : {}'.format(each, label_count[index]))

    '''
    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = 41
    ENCODE_UNITS = 20
    NEW_CLASS = label_list.index('guess_passwd.')
    CODES = 5
    # parameter

    count = 0
    for each in labels[:DATA_LENGTH]:
        if each[NEW_CLASS] == 1:
            count = count + 1
    '''