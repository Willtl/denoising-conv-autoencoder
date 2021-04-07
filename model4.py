import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        # Initialize superclass
        super(ConvAutoEncoder, self).__init__()

        # Conv networkS
        # Output size of each convolutional layer = [(in_channel + 2 * padding - kernel_size) / stride] + 1
        self.convEncoder = nn.Sequential(
            # In this case output = [(28 + 2 * 1 - 5) / 1] + 1 = 26
            # Input [128, 1, 28, 28]
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 28, 28]
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 13, 13]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 13, 13]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 6, 6]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
            # [batch_size, 128, 6, 6]
        )

        self.convDecoder = nn.Sequential(
            # [batch_size, 128, 6, 6]
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 4, 4]
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 13, 13]
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 13, 13]
            nn.ConvTranspose2d(32, 32, kernel_size=3, stride=2, padding=0, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 28, 28]
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            # [batch_size, 32, 28, 28]
        )

        self.classifier = nn.Sequential(
            nn.Linear(4608, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Auto-encoder part
        conv_encoded = self.convEncoder(x)
        # print(conv_encoded.shape)
        conv_decoded = self.convDecoder(conv_encoded)
        # print(conv_decoded.shape)
        # quit()

        # Classifier part
        # print(f'# input dimensions for the linear layer {conv_encoded.view(conv_encoded.shape[0], -1).shape}')
        cls = self.classifier(conv_encoded.view(conv_encoded.shape[0], -1))
        return conv_encoded, conv_decoded, cls