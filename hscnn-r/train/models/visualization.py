import os
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation

f, axes = plt.subplots(1, 2)
f.set_size_inches((14, 6))
f.suptitle('Loss', fontsize=15)

recent = sorted(list(map(lambda fname: (fname, os.path.getctime(fname)),
                         list(filter(lambda fname: os.path.isdir(fname), os.listdir('.'))))
                     ), key=lambda x: x[1], reverse=True)[0][0]

path = recent + '/'


def animate(i):
    try:
        data = pd.read_csv(path + 'loss.csv')
    except pd.errors.EmptyDataError:
        axes[0].cla()
        axes[1].cla()
        print('Empty csv file!')
        return

    plt.cla()
    x = data['epoch']
    y1 = data['train_loss']
    y2 = data['test_loss']

    axes[0].set_title('Full Range')
    axes[0].plot(x, y1, 'r', label='train')
    axes[0].plot(x, y2, 'b', label='test')

    axes[1].set_title('Range < 0.1')
    axes[1].set_ylim((0, 0.1))
    axes[1].plot(x, y1, 'r', label='train')
    axes[1].plot(x, y2, 'b', label='test')

    plt.legend()
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=1000)

plt.tight_layout()
plt.show()
