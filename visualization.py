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
        g = sns.relplot(x='time', y='signal', hue='open_channels', size='open_channels', data=self.data[:][:-1:10000])
        plt.show()


if __name__ == '__main__':
    vi = Visualization('data/train.csv')
    vi.draw_statistics()
