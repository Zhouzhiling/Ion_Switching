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
        self.label_dim = -1
        self.train_data, self.test_data = self.load_data('./data/')

    @staticmethod
    def load_data(path):
        train_data = pd.read_csv(path + 'train.csv')
        test_data = pd.read_csv(path + 'test.csv')
        return train_data, test_data

    def preprocess(self, choice='train'):
        if choice == 'train':
            features, label_in = self.train_data['signal'][:-1:100], self.train_data['open_channels'][:-1:100]

            signals = np.reshape(features[100:].to_numpy(), (len(features[100:]), 1))

            labels = pd.DataFrame(utils.to_categorical(label_in, dtype='int32')).to_numpy()
            self.label_dim = np.size(labels, axis=1)

            pre_channels = np.zeros((len(features) - self.time_steps, self.time_steps, self.label_dim))

            for i in range(0, len(features) - self.time_steps):
                pre_channel = labels[i:i + self.time_steps][:]
                pre_channels[i][:][:] = pre_channel
                # signal = features[:][i+self.time_steps]
                # signals.append(signal)

            return labels[100:], signals, np.reshape(pre_channels, (len(pre_channels), self.time_steps, self.label_dim))

    def train(self):
        labels, signals, pre_channels = self.preprocess(choice='train')

        X = signals
        y = labels

        input_data = Input(shape=(self.time_steps, self.label_dim,))
        signal = Input(shape=(1,))

        lstm = LSTM(
            units=256,
            activation='tanh',
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

        optimizer = Adam(learning_rate=0.1)

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
            batch_size=32,
            epochs=10,
            callbacks=callbacks,
            validation_split=0.2
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

        result = pd.DataFrame(result)
        result.to_csv('result.csv')

        print(result)


if __name__ == '__main__':
    ion_switching_lstm = IonSwitchingLSTM()
    ion_switching_lstm.train()
    # ion_switching_lstm.test()
