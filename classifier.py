import numpy as np
import pandas as pd
from sklearn.datasets import kddcup99
from keras.layers import Input, Dense, Activation, Softmax
from keras import activations
from keras.models import Model, Sequential, load_model
from keras.losses import kullback_leibler_divergence
from keras.callbacks import EarlyStopping
from DataPreprocess import preprocess_data, preprocess_target, train_test_split, confidence_test_split
import csv


class RobustSoftmax(Softmax):
    def call(self, inputs):
        np.true_divide(inputs, 200)
        return activations.softmax(inputs, axis=self.axis)


if __name__ == '__main__':

    label_list = ['normal', 'buffer_overflow', 'loadmodule', 'perl', 'neptune', 'smurf',
          'guess_passwd', 'pod', 'teardrop', 'portsweep', 'ipsweep', 'land', 'ftp_write',
          'back', 'imap', 'satan', 'phf', 'nmap', 'multihop', 'warezmaster', 'warezclient',
          'spy', 'rootkit']
    
    # label_list = [normal_list, dos_list, u2r_list, r2l_list, probe_list]
    # label_list = [normal_list, dos_list, u2r_list, r2l_list]

    df = pd.read_csv('dataset/KDDTrain+.txt', header=None)
    dataset = df.as_matrix()
    data = dataset[:, :41]
    labels = dataset[:, 41:42]
    label = 'buffer_overflow'

    '''
    dataset = kddcup99.fetch_kddcup99()
    data = dataset.data
    labels = dataset.target
    '''

    # 數據預處理
    data = preprocess_data(data)

    x_train, y_train, x_test, y_test, outlier_set = confidence_test_split(data, labels, label_list, label)
    # x_train, y_train, x_test, y_test = train_test_split(data, labels, label_list)
    
    # parameter
    DATA_LENGTH = data.shape[0]
    INPUT_SHAPE = data.shape[1]
    OUTPUT_SHAPE = y_train.shape[1]
    # parameter

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

    # Train Autoencoder
    autoencoder = Model(ae_input_layer, ae_output_layer)
    autoencoder.compile(loss='mse', optimizer='adam')
    autoencoder.fit(x_train, x_train, 
                    validation_data=(x_test, x_test), 
                    verbose=1, epochs=50, callbacks=[monitor,])
    sigmoid = Activation('sigmoid')(ae_code)
    encoder = Model(ae_input_layer, sigmoid)
    encoder.save('encoder.h5')

    # Fix parameters
    # ae_first_en_layer.trainable = False
    # ae_second_en_layer.trainable = False
    # ae_code.trainable = False

    # input_layer = Input(shape=(INPUT_SHAPE,))
    # x = Dense(units=100, kernel_initializer='random_normal', activation='relu')(input_layer)
    input_layer = Dense(units=200, kernel_initializer='random_normal', activation='relu')(ae_code)
    x = Dense(units=100, kernel_initializer='random_normal', activation='relu')(input_layer)
    x = Dense(units=50, kernel_initializer='random_normal', activation='relu')(x)
    x = Dense(units=OUTPUT_SHAPE, kernel_initializer='random_normal')(x)
    output_layer = RobustSoftmax()(x)

    model = Model(ae_input_layer, output_layer)
    # confident_machine = Model(input_layer, x)
    for index in range(3):
        model.get_layer(index=index).trainable = False

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.summary()

    model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), 
              verbose=1, epochs=50, callbacks=[monitor,])

    model.save('classifier.h5')

    # model = load_model('classifier.h5')
    # confident_machine.save('classifier.h5')

    _, accuracy = model.evaluate(x_test, y_test)
    print(accuracy)

    # print(model.predict(x_test[:1]))
    # print(confident_machine.predict(x_test[:1]))
