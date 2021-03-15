import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Dataset, DataLoader

import model1
import model2

# Reproducibility
torch.manual_seed(0)
torch.cuda.manual_seed(0)
random.seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 64
LR = 0.0005
DOWNLOAD_MNIST = True
ADD_NOISE_MNIST = False
NOISE_PROB = 0.25
N_TEST_IMG = 10


def test_classifier(autoencoder):
    # Define GPU and CPU devices
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    # Load test data (noise, normal, and labels)
    noise_data = torch.load('noisy-mnist/0.25/test/noisy.pt')
    # index = random.randint(0, 60000)
    # plot_one(noise_data[index])
    normal_data = torch.load('noisy-mnist/0.25/test/normal.pt')
    # plot_one(normal_data[index])
    label_data = torch.load('noisy-mnist/0.25/test/label.pt')

    # Create noise dataset for easy management
    dataset = NoisyDataset(noise_data, normal_data, label_data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    # Move data to GPU
    batch_x = []
    batch_y = []
    batch_z = []
    for noise_data, normal_data, label_data in dataloader:
        batch_x.append(Variable(noise_data).to(gpu))
        batch_y.append(Variable(normal_data).to(gpu))
        batch_z.append(Variable(label_data).to(gpu))

    # Evaluate model
    autoencoder.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for b_x, b_y, b_z in zip(batch_x, batch_y, batch_z):
            encoded, decoded, cls = autoencoder(b_x)
            # Loop through outputs and check if it is correct or not
            for idx, i in enumerate(cls):
                if torch.argmax(i) == b_z[idx]:
                    correct += 1
                total += 1

        print("Accuracy: ", round(correct / total, 6))


def test_denoising(autoencoder, batch_x, batch_y):
    # Define GPU and CPU devices
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")
    # # First N_TEST_IMG images for visualization
    batch_numb = random.randint(0, len(batch_x))
    view_data = Variable(batch_x[batch_numb][:N_TEST_IMG]).to(cpu)
    view_norm = Variable(batch_y[batch_numb][:N_TEST_IMG]).to(cpu)

    # Plotting decoded image
    with torch.no_grad():
        # Set the model to evaluation mode
        autoencoder = autoencoder.eval()

        # Encode and decode view_data to visualize the outcome
        encoded_data, decoded_data, classf_data = autoencoder(view_data.to(gpu))
        decoded_data = decoded_data.to(cpu)

        # initialize figure
        f, a = plt.subplots(3, N_TEST_IMG, figsize=(5, 2))

        for i in range(N_TEST_IMG):
            a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(view_norm.data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())

        for i in range(N_TEST_IMG):
            a[2][i].clear()
            a[2][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[2][i].set_xticks(())
            a[2][i].set_yticks(())
        plt.show()


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
def load_dataset():
    train_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    test_data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=False,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    return train_data, test_data


# Add noise to the MNIST handwritten digit database and save to disk
def create_noisy_dataset(train_loader):
    # Load dataset
    train_data, test_data = load_dataset()
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
                    if r <= NOISE_PROB / 3:
                        var_x[i][0][j][k] = 0.0
                    elif r <= NOISE_PROB / 2:
                        var_x[i][0][j][k] = 1.0
                    elif r <= NOISE_PROB:
                        var_x[i][0][j][k] = random.random()
        torch.save(var_x, 'train/noisy.pt')
        torch.save(var_y, 'train/normal.pt')
        torch.save(var_l, 'train/label.pt')

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
                    if r <= NOISE_PROB / 3:
                        var_x[i][0][j][k] = 0.0
                    elif r <= NOISE_PROB / 2:
                        var_x[i][0][j][k] = 1.0
                    elif r <= NOISE_PROB:
                        var_x[i][0][j][k] = random.random()
        torch.save(var_x, 'train/noisy.pt')
        torch.save(var_y, 'train/normal.pt')
        torch.save(var_l, 'train/label.pt')


def plot_one(data):
    data = data.view(28, 28)
    plt.imshow(data, cmap='gray')
    plt.title("Figure")
    plt.show()


def main():
    # Define GPU and CPU devices
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    cpu = torch.device("cpu")

    # Read MNIST handwritten digit database, add noise given probability and save to disk
    if ADD_NOISE_MNIST:
        train_data = load_dataset()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)
        create_noisy_dataset(train_loader)

    # Load noise, normal, and labels
    noise_data = torch.load('noisy-mnist/0.5/train/noisy.pt')
    # index = random.randint(0, 60000)
    # plot_one(noise_data[index + 3])
    normal_data = torch.load('noisy-mnist/0.5/train/normal.pt')
    # plot_one(normal_data[index + 3])
    label_data = torch.load('noisy-mnist/0.5/train/label.pt')

    # Create noise dataset for easy management
    dataset = NoisyDataset(noise_data, normal_data, label_data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=False)

    # Move data to GPU
    batch_x = []
    batch_y = []
    batch_z = []
    for noise_data, normal_data, label_data in dataloader:
        batch_x.append(Variable(noise_data).to(device))
        batch_y.append(Variable(normal_data).to(device))
        batch_z.append(Variable(label_data).to(device))

    # Load model and move it to the GPU
    autoencoder = model2.ConvAutoEncoder()
    autoencoder.to(device)
    # Define optimizer (must be done after moving the model to the GPU)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    # Criterion for the conv. autoencoder
    criterion1 = nn.MSELoss().to(device)
    criterion2 = nn.NLLLoss().to(device)

    scheduler = StepLR(optimizer, step_size=1, gamma=0.7)

    # Training loop
    for epoch in range(EPOCH):
        # To calculate mean loss and calculate time per epoch
        autoencoder_loss = []
        classifier_loss = []
        start = time.time()
        # Loop through batches
        for b_x, b_y, b_z in zip(batch_x, batch_y, batch_z):
            encoded, decoded, cls = autoencoder(b_x)    # Feed data
            optimizer.zero_grad()                       # clear gradients for this training step
            loss1 = criterion1(decoded, b_y)            # reconstruction loss (mean square error)
            loss2 = criterion2(cls, b_z)                # classifier loss (negative log likelihood)
            loss3 = (0.1 * loss1) + (0.9 * loss2)       # combined loss (reconstruction + classifier)
            loss3.backward()                            # backpropagation, compute gradients
            optimizer.step()                            # apply gradients

            autoencoder_loss.append(loss1.to(cpu).item())   # used to calculate the autoencoder epoch mean loss
            classifier_loss.append(loss2.to(cpu).item())    # used to calculate the classifier epoch mean loss

        print(f'Autoencoder {epoch}, mean loss: {np.mean(np.array(autoencoder_loss))}, time: {time.time() - start}')
        print(f' Classifier {epoch}, mean loss: {np.mean(np.array(classifier_loss))}, time: {time.time() - start}')
    # Testing reconstructed images
    test_denoising(autoencoder, batch_x, batch_y)
    # Testing classification
    test_classifier(autoencoder)


if __name__ == '__main__':
    main()
