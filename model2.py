import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        super(ConvAutoEncoder, self).__init__()

        n_features = 11
        # Conv network
        # Output size of each convolutional layer = [(in_channel + 2 * padding - kernel_size) / stride] + 1
        self.convEncoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_features, kernel_size=4, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),

            nn.Conv2d(in_channels=n_features, out_channels=n_features * 2, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(),

            nn.Conv2d(in_channels=n_features * 2, out_channels=n_features * 3, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_features * 3),
            nn.ReLU(),

            nn.Conv2d(in_channels=n_features * 3, out_channels=n_features * 4, kernel_size=4, stride=2, padding=1,
                      bias=False),
            nn.BatchNorm2d(n_features * 4),
            nn.ReLU()
        )

        self.convDecoder = nn.Sequential(
            nn.ConvTranspose2d(n_features * 4, n_features * 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 3),
            nn.ReLU(),

            nn.ConvTranspose2d(n_features * 3, n_features * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features * 2),
            nn.ReLU(),

            nn.ConvTranspose2d(n_features * 2, n_features, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(n_features),
            nn.ReLU(),

            nn.ConvTranspose2d(n_features, 1, kernel_size=4, stride=2, padding=3),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(176, 1024),
            nn.Dropout(0.5),
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
