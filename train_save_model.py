import numpy as np
import pandas as pd
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense, Activation
from keras.models import Model
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
    

    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = data.shape[1]
    OUTPUT_SHAPE = len(label_list) - 1
    # parameter

    # 數據預處理
    data = preprocess_data(data)
    
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
    # output_layer = Activation('softmax')(x)
    output_layer = RobustSoftmax()(x)
    
    model = Model(ae_input_layer, output_layer)
    model_weights = model.get_weights()
    # Train Autoencoder

    for label in label_list:
        copy_list = label_list.copy()
        x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, copy_list, label)
        # x_train, y_train, x_test, y_test = train_test_split(data, labels, label_list)
        print(x_train.shape, y_train.shape, x_test.shape, y_train.shape, outlier_set.shape)
        autoencoder.set_weights(ae_weights)
        for layer_index in range(3):
            autoencoder.get_layer(index=layer_index).trainable = True

        autoencoder.compile(loss='mse', optimizer='adam')
        autoencoder.fit(x_train, x_train,
                        validation_data=(x_test, x_test),
                        verbose=1, epochs=50, callbacks=[monitor, ])

        sigmoid = Activation('sigmoid')(ae_code)
        encoder = Model(ae_input_layer, sigmoid)
        encoder.save('model/encoder_{}.h5'.format(label))

        # model configure
        model.set_weights(model_weights)
        for layer_index in range(3):
            model.get_layer(index=layer_index).trainable = False
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # model configure

        model.fit(x_train, y_train,
                  validation_data=(x_test, y_test),
                  verbose=1, epochs=50, callbacks=[monitor, ])
        model.save('model/classifier_{}.h5'.format(label))
        
