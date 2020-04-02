from keras.layers import LSTM, Input, Dense, Concatenate
from keras.optimizers import Adam
from keras import Model
from keras import utils
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import pandas as pd


class IonSwitchingLSTM(object):

    def __init__(self):
        self.time_steps = 100
        self.feature_dim = 1
        self.label_dim = 4
        self.train_data, self.test_data = self.load_data('./data/')

    @staticmethod
    def load_data(path):
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

        # X = np.random.random((100, self.time_steps, self.label_dim))
        # s = np.random.random((100, 1))
        # y = np.random.random((100, self.label_dim))

        X = signals
        y = labels

        input_data = Input(shape=(self.time_steps, self.label_dim,))
        signal = Input(shape=(1,))

        lstm = LSTM(
            units=150,
            activation='tanh',
            recurrent_activation='sigmoid',
            use_bias=True
        )(input_data)

        lstm = Dense(
            units=self.label_dim,
            activation='softmax'
        )(lstm)

        mix = Concatenate()([lstm, signal])

        output = Dense(
            units=self.label_dim,
            activation='softmax'
        )(mix)

        model = Model([input_data, signal], output)

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
            x=[pre_channels, X],
            y=y,
            shuffle=True,
            batch_size=1,
            epochs=1,
            callbacks=callbacks
        )

        model.save('lstm.hdf5')

    def test(self):
        seed = np.random.random((1, self.time_steps, self.label_dim))

        model = load_model('lstm.hdf5')

        result = []

        for i in range(len(self.test_data)):
            signal = np.reshape(self.test_data['signal'][i], (1, 1))
            curr = model.predict([seed, signal])

            seed = np.reshape(np.concatenate((seed[0][1:][:], curr)), (1, self.time_steps, self.label_dim))
            result.append(curr)

        print(result)


if __name__ == '__main__':
    ion_switching_lstm = IonSwitchingLSTM()
    ion_switching_lstm.train()
    ion_switching_lstm.test()
