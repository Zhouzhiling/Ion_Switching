from keras.layers import LSTM, Input, Dense, Lambda, Flatten
from keras.optimizers import RMSprop, Adam
from keras import Model
import pandas as pd
from keras.models import load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np


class IonSwitchingLSTM(object):

    def __init__(self):
        self.time_steps = 100
        self.feature_dim = 1
        self.label_dim = 4
        pass

    def load_data(self, path):
        return pd.read_csv(path)

    def preprocess(self, path):
        X, y = [], []
        data = self.load_data(path)
        data['open_channels']


        return X, y

    def train(self, path):

        X, y = self.preprocess(path)

        # X = np.random.random((100, self.time_steps, self.feature_dim))
        # y = np.random.random((100, self.label_dim))

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

    def test(self, path):
        X, y = self.preprocess(path)

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
    ion_switching_lstm.train('data/train.csv')
    # ion_switching_lstm.test('data/test.csv')
