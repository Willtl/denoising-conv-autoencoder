import os
import random

import torch
import torch.backends.cudnn
import torch.utils.data as Data
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST


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
def load_mnist(img_size):
    transform = transforms.Compose([transforms.Resize(img_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])

    train_data = MNIST(root='./mnist/', train=True, download=True, transform=transform)
    test_data = MNIST(root='./mnist/', train=False, download=True, transform=transform)

    return train_data, test_data


# Read MNIST handwritten digit database, add noise given probability and save to disk
def create_noisy_dataset(img_size, noise_prob):
    # Load dataset
    train_data, test_data = load_mnist(img_size)
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=train_data, batch_size=60000, shuffle=True, pin_memory=False)

    # Create folders
    if not os.path.exists(f'noisy-mnist/{noise_prob}/train/'):
        os.makedirs(f'noisy-mnist/{noise_prob}/train/')
        os.makedirs(f'noisy-mnist/{noise_prob}/test/')

        # Add noise to the train data
        for x, y in train_loader:
            var_x = Variable(x)
            var_y = Variable(torch.clone(x))
            var_l = Variable(y)
            # Add noise to the data
            if noise_prob > 0.0:
                for i in range(60000):
                    print(f'Adding noise to the {i}th sample of the training set')
                    for j in range(img_size):
                        for k in range(img_size):
                            r = random.random()
                            if r <= noise_prob:
                                var_x[i][0][j][k] = torch.normal(mean=0, std=1.0, size=(1, 1)).item()
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
            if noise_prob > 0.0:
                for i in range(10000):
                    print(f'Adding noise to the {i}th sample of the test set')
                    for j in range(img_size):
                        for k in range(img_size):
                            r = random.random()
                            if r <= noise_prob:
                                var_x[i][0][j][k] = torch.normal(mean=0, std=1.0, size=(1, 1)).item()
            torch.save(var_x, f'noisy-mnist/{noise_prob}/test/noisy.pt')
            torch.save(var_y, f'noisy-mnist/{noise_prob}/test/normal.pt')
            torch.save(var_l, f'noisy-mnist/{noise_prob}/test/label.pt')
