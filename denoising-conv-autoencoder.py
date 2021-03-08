import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import random
from torch.utils.data import Dataset, DataLoader

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 10
BATCH_SIZE = 128
LR = 0.0005
DOWNLOAD_MNIST = True
ADD_NOISE_MNIST = False
NOISE_PROB = 0.25
N_TEST_IMG = 10


def main():
    # Define GPU and CPU devices
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    cpu = torch.device("cpu")
    gpu = torch.device("cuda:0")

    # Read MNIST handwritten digit database, add noise given probability and save to disk
    if ADD_NOISE_MNIST:
        train_data = load_dataset()
        train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
        create_noisy_dataset(train_loader)

    # Load noise, normal, and labels
    noise_data = torch.load('noise_data.pt')
    normal_data = torch.load('normal_data.pt')
    label_data = torch.load('label_data.pt')

    # Create noise dataset for easy management
    dataset = NoisyDataset(noise_data, normal_data, label_data)
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

    # Move data to GPU
    batch_x = []
    batch_y = []
    batch_z = []
    for noise_data, normal_data, label_data in dataloader:
        batch_x.append(Variable(noise_data).to(device))
        batch_y.append(Variable(normal_data).to(device))
        batch_z.append(Variable(label_data).to(device))

    # Load model and move it to the GPU
    autoencoder = ConvAutoEncoder()
    autoencoder.to(device)
    # Define optimizer (must be done after moving the model to the GPU)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
    # Criterion for the conv. autoencoder
    criterion = nn.MSELoss().to(device)

    # Training loop
    for epoch in range(EPOCH):
        # To calculate mean loss and calculate time per epoch
        epoch_loss = []
        start = time.time()
        # Loop through batches
        for b_x, b_y in zip(batch_x, batch_y):
            encoded, decoded = autoencoder(b_x)

            loss = criterion(decoded, b_y)  # mean square error
            optimizer.zero_grad()           # clear gradients for this training step
            loss.backward()                 # backpropagation, compute gradients
            optimizer.step()                # apply gradients

            epoch_loss.append(loss.to(cpu).item())  # used to calculate the epoch mean loss
        print(f'Epoch {epoch}, mean loss: {np.mean(np.array(epoch_loss))}, time: {time.time() - start}')

    # # First N_TEST_IMG images for visualization
    view_data = Variable(batch_x[0][:N_TEST_IMG]).to(cpu)

    # Testing - Plotting decoded image
    with torch.no_grad():
        # Set the model to evaluation mode
        autoencoder = autoencoder.eval()

        # Encode and decode view_data to visualize the outcome
        _, decoded_data = autoencoder(view_data.to(device))
        decoded_data = decoded_data.to(cpu)

        # initialize figure
        f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))

        for i in range(N_TEST_IMG):
            a[0][i].imshow(np.reshape(view_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[0][i].set_xticks(())
            a[0][i].set_yticks(())

        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data.data.numpy()[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(())
            a[1][i].set_yticks(())
        plt.show()


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        super(ConvAutoEncoder, self).__init__()

        n_features = 6
        # Conv network
        # Output size of each convolutional layer = [(in_channel + 2 * padding - kernel_size) / stride] + 1
        self.convEncoder = nn.Sequential(
            # In this case output = [(28 + 2 * 1 - 5) / 1] + 1 = 26
            # Input [128, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=n_features, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e, 16, 16]

            nn.Conv2d(in_channels=n_features, out_channels=n_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e * 2, 8, 8]

            nn.Conv2d(in_channels=n_features * 2, out_channels=n_features * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e * 4, 4, 4]
        )

        self.convDecoder = nn.Sequential(
            # Input [128, features_e * 4, 4, 4]
            nn.ConvTranspose2d(n_features * 4, n_features * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(),
            # Output [128, features_e * 2, 8, 8]

            nn.ConvTranspose2d(n_features * 2, n_features, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),
            # Output [128, 10, 16, 16]

            nn.ConvTranspose2d(n_features, 1, kernel_size=4, stride=2, padding=3),
            nn.Sigmoid()
            # Output [128, 1, 28, 28]
        )

    def forward(self, x):
        conv_encoded = self.convEncoder(x)
        conv_decoded = self.convDecoder(conv_encoded)

        # decoded = self.decoder(encoded)
        return conv_encoded, conv_decoded


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
    data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    return data


def plot_one(data):
    data = data.view(28, 28)
    plt.imshow(data, cmap='gray')
    plt.title("Figure")
    plt.show()


# Add noise to the MNIST handwritten digit database and save to disk
def create_noisy_dataset(train_loader):
    # Load dataset
    train_data = load_dataset()
    # Data Loader for easy mini-batch return in training
    train_loader = Data.DataLoader(dataset=train_data, batch_size=60000, shuffle=True, pin_memory=True)
    # Loop through to add noise
    for x, y in train_loader:
        var_x = Variable(x)
        var_y = Variable(x)
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

        # plot_one(var_x[0], var_x[0])

        torch.save(var_x, 'noise_data.pt')
        torch.save(var_y, 'normal_data.pt')
        torch.save(var_l, 'label_data.pt')
        break


main()