import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        super(ConvAutoEncoder, self).__init__()

        n_features = 3
        # Conv network
        # Output size of each convolutional layer = [(in_channel + 2 * padding - kernel_size) / stride] + 1
        self.convEncoder = nn.Sequential(
            # In this case output = [(28 + 2 * 1 - 5) / 1] + 1 = 26
            # Input [128, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e, 16, 16]

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e * 2, 8, 8]

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2),
            # Output [128, features_e * 4, 4, 4]
        )

        self.convDecoder = nn.Sequential(
            # Input [128, features_e * 4, 4, 4]
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # Output [128, features_e * 2, 8, 8]

            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # Output [128, 10, 16, 16]

            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=3),
            nn.Sigmoid()
            # Output [128, 1, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),

            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Autoencoder part
        conv_encoded = self.convEncoder(x)
        conv_decoded = self.convDecoder(conv_encoded)
        # Classifier part
        # print(f'# input dimensions for the linear layer {conv_encoded.view(conv_encoded.shape[0], -1).shape}')
        cls = self.classifier(conv_encoded.view(conv_encoded.shape[0], -1))

        return conv_encoded, conv_decoded, cls