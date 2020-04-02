from keras.layers import LSTM, Input, Dense, Lambda, Flatten
from keras.optimizers import RMSprop, Adam
from keras import Model, utils
import pandas as pd
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


class IonSwitchingLSTM(object):

    def __init__(self):
        self.time_steps = 100
        self.feature_dim = 1
        self.label_dim = 4
        self.train_data, self.test_data = self.load_data('./data/')

    def load_data(self, path):
        train_data = pd.read_csv(path + 'train.csv')
        test_data = pd.read_csv(path + 'test.csv')
        return train_data, test_data

    def preprocess(self, choice='train'):
        if choice == 'train':
            features, label_in = self.train_data['signal'], self.train_data['open_channels']

            pre_channels = []
            signals = features[:][100:]
            labels = pd.DataFrame(utils.to_categorical(label_in[100:], dtype='int32'))

            for i in range(0, len(features)-self.time_steps):
                pre_channel = labels[i:i + self.time_steps][:]
                pre_channels.append(pre_channel.values)
                # signal = features[:][i+self.time_steps]
                # signals.append(signal)

            return labels, signals, pre_channels

    def train(self):
        labels, signals, pre_channels = self.preprocess(choice='train')

        # X = np.random.random((100, self.time_steps, self.feature_dim))
        # y = np.random.random((100, self.label_dim))

        X = signals
        y = labels

        input_data = Input(shape=(self.time_steps, self.feature_dim,))

        lstm = LSTM(
            units=150,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            input_shape=(self.time_steps, self.feature_dim)
        )(input_data)

        output = Dense(
            units=self.label_dim,
            activation='softmax'
        )(lstm)

        model = Model(input_data, output)

        optimizer = Adam(learning_rate=0.01)

        model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy']
        )

        model.summary()

        callbacks = [
            ModelCheckpoint(
                'lstm_best.hdf5',
                monitor='loss',
                verbose=1,
                save_best_only=True,
                mode='auto',
                period=1
            ),
            EarlyStopping(
                monitor='loss',
                patience=5,
                mode='auto',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='loss',
                patience=3,
                verbose=1,
                mode='auto',
                factor=0.1
            )
        ]

        model.fit(
            x=X,
            y=y,
            shuffle=True,
            batch_size=32,
            epochs=100,
            callbacks=callbacks
        )

        model.save('lstm.hdf5')

    def test(self):
        # X, y = self.preprocess()

        input_data = Input(shape=(self.time_steps, self.feature_dim,))

        lstm = LSTM(
            units=150,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True,
            input_shape=(self.time_steps, self.feature_dim)
        )(input_data)

        output = Dense(
            units=self.label_dim,
            activation='softmax'
        )(lstm)

        model = Model(input_data, output)
        model.load_weights('lstm_best_model_weights.hdf5')


if __name__ == '__main__':
    ion_switching_lstm = IonSwitchingLSTM()
    ion_switching_lstm.train()
    # ion_switching_lstm.test('data/test.csv')
