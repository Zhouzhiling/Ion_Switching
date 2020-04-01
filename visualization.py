import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class Visualization(object):

    def __init__(self, path):
        self.data = pd.read_csv(path)

    def print_statistics(self):
        pass

    def draw_statistics(self):
        type1 = self.data[:][self.data['open_channels']==1]
        g = sns.relplot(x='time', y='signal', size=20, data=type1[:][:-1:100])
        # g = sns.relplot(x='time', y='signal', hue='open_channels', size=20, data=self.data[:][:-1:5000])
        plt.show()


if __name__ == '__main__':
    vi = Visualization('data/train.csv')
    vi.draw_statistics()
