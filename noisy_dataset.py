import random

import torch
import torch.backends.cudnn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset


class NoisyDataset(Dataset):
    def __init__(self, noise_data, normal_data, label_data):
        self.x = noise_data
        self.y = normal_data
        self.label = label_data
        self.n_samples = noise_data.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.label[index]

    def __len__(self):
        return self.n_samples


# MNIST handwritten digit database
def load_mnist():
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    return train_data, test_data


# Read MNIST handwritten digit database, add noise given probability and save to disk
def create_noisy_dataset(noise_prob):
    # Load dataset
    train_data, test_data = load_mnist()
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=train_data, batch_size=60000, shuffle=True, pin_memory=False)
    # Add noise to the train data
    for x, y in train_loader:
        var_x = Variable(x)
        var_y = Variable(torch.clone(x))
        var_l = Variable(y)
        # Add noise to the data
        for i in range(60000):
            print(i)
            for j in range(28):
                for k in range(28):
                    r = random.random()
                    if r <= noise_prob / 3:
                        var_x[i][0][j][k] = 0.0
                    elif r <= noise_prob / 2:
                        var_x[i][0][j][k] = 1.0
                    elif r <= noise_prob:
                        var_x[i][0][j][k] = random.random()
        torch.save(var_x, f'noisy-mnist/{noise_prob}/train/noisy.pt')
        torch.save(var_y, f'noisy-mnist/{noise_prob}/train/normal.pt')
        torch.save(var_l, f'noisy-mnist/{noise_prob}/train/label.pt')

    # Prepare test data
    test_loader = Data.DataLoader(dataset=test_data, batch_size=10000, shuffle=True, pin_memory=False)
    # Add noise to the test data
    for x, y in test_loader:
        var_x = Variable(x)
        var_y = Variable(torch.clone(x))
        var_l = Variable(y)
        # Add noise to the data
        for i in range(10000):
            print(i)
            for j in range(28):
                for k in range(28):
                    r = random.random()
                    if r <= noise_prob / 3:
                        var_x[i][0][j][k] = 0.0
                    elif r <= noise_prob / 2:
                        var_x[i][0][j][k] = 1.0
                    elif r <= noise_prob:
                        var_x[i][0][j][k] = random.random()
        torch.save(var_x, f'noisy-mnist/{noise_prob}/test/noisy.pt')
        torch.save(var_y, f'noisy-mnist/{noise_prob}/test/normal.pt')
        torch.save(var_l, f'noisy-mnist/{noise_prob}/test/label.pt')