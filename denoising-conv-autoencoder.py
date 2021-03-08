import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import time
import kmeans

torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 128
LR = 0.0002
DOWNLOAD_MNIST = False
N_TEST_IMG = 10


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
            # Output [128, features_e * 3, 4, 4]

            # In this case output = [(13 + 2 * 1 - 5) / 1] + 1 = 11
            # nn.Conv2d(in_channels=10, out_channels=24, kernel_size=4, padding=1, stride=1),
            # nn.Sigmoid(),
            # nn.MaxPool2d(kernel_size=2),  # End up with 24 channels of size 5 x 5

            # Dense layers
            # nn.Linear(in_features=24 * 5 * 5, out_features=64),
            # nn.Relu(),
            # nn.Dropout(p=0.2),  # Dropout with probability of 0.2 to avoid overfitting
            # nn.Linear(in_features=64, out_features=10)  # 10 equals the number of classes
        )

        self.convDecoder = nn.Sequential(
            # Input [128, features_e * 3, 4, 4]
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
        # print("forwarding")
        # print(x.shape)
        conv_encoded = self.convEncoder(x)
        # print("encoded shape", conv_encoded.shape)
        # quit()
        conv_decoded = self.convDecoder(conv_encoded)
        # print("decoded shape", conv_decoded.shape)
        # quit()

        # decoded = self.decoder(encoded)
        return conv_encoded, conv_decoded


def load_dataset():
    # Mnist digits dataset
    data = torchvision.datasets.MNIST(
        root='./mnist/',
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST
    )
    return data


def plot_one(data, target):
    # plot one example
    # print(data.data.size())     # (60000, 28, 28)
    # print(data.targets.size())   # (60000)
    # plt.imshow(data.numpy(), cmap='gray')
    plt.imshow(data, cmap='gray')
    plt.title('%i' % target)
    plt.show()


# Define GPU and CPU devices
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
cpu = torch.device("cpu")
gpu = torch.device("cuda:0")

# Load dataset
train_data = load_dataset()

# Define model
autoencoder = ConvAutoEncoder()

# Move it to the GPU
autoencoder.to(device)
# Define optimizer after moving to the GPU
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=LR)
criterion = nn.MSELoss().to(device)

# Data Loader for easy mini-batch return in training
train_loader = Data.DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# Move data to GPU
batch_x = []
batch_y = []
for x, y in train_loader:
    batch_x.append(Variable(x).to(device))
    batch_y.append(Variable(x).to(device))

gpu_data = zip(batch_x, batch_y)

# Training
for epoch in range(EPOCH):
    # To calculate mean loss over this epoch
    epoch_loss = []
    start = time.time()

    # Loop through batches
    for b_x, b_y in zip(batch_x, batch_y):
        encoded, decoded = autoencoder(b_x)

        loss = criterion(decoded, b_y)      # mean square error
        optimizer.zero_grad()               # clear gradients for this training step
        loss.backward()                     # backpropagation, compute gradients
        optimizer.step()                    # apply gradients

        epoch_loss.append(loss.to(cpu).item())      # used to calculate the epoch mean loss
    print(f'Epoch {epoch}, mean loss: {np.mean(np.array(epoch_loss))}, time: {time.time() - start}')

# First N_TEST_IMG images for visualization
view_data = Variable(train_data.data[:N_TEST_IMG].view(-1, 1, 28, 28).type(torch.FloatTensor) / 255.)

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