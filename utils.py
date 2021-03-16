from scipy.signal import hilbert
import matplotlib.pyplot as plt
import numpy as np


def plot_hilbert_transform():
    dt = 0.0001
    t = np.arange(0, 0.1, dt)
    x = (1 + np.cos(2 * np.pi * 50 * t)) * np.cos(2 * np.pi * 1000 * t)
    plt.plot(x)
    plt.show()

    h = abs(hilbert(x))
    plt.plot(h)
    plt.show()

    x = x + 1
    h = abs(hilbert(x))
    plt.plot(h)
    plt.show()


def plot_one(data):
    data = data.view(28, 28)
    plt.imshow(data, cmap='gray')
    plt.title("Figure")
    plt.show()
