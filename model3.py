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
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 28, 28]
            nn.MaxPool2d(kernel_size=2),
            # [batch_size, 32, 14, 14]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 14, 14]
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 14, 14]
            nn.MaxPool2d(kernel_size=2),
            # [batch_size, 64, 7, 7]
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
            # [batch_size, 128, 7, 7]
        )

        self.convDecoder = nn.Sequential(
            # [batch_size, 128, 7, 7]
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 7, 7]
            nn.Upsample(scale_factor=2),
            # nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            # [batch_size, 64, 14, 14]
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # [batch_size, 64, 7, 7]
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 14, 14]
            nn.Upsample(scale_factor=2),
            # nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1),
            # [batch_size, 32, 28, 28]
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # [batch_size, 32, 28, 28]
            nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.classifier = nn.Sequential(
            nn.Linear(6272, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Auto-encoder part
        conv_encoded = self.convEncoder(x)
        conv_decoded = self.convDecoder(conv_encoded)
        # Classifier part
        # print(f'# input dimensions for the linear layer {conv_encoded.view(conv_encoded.shape[0], -1).shape}')
        cls = self.classifier(conv_encoded.view(conv_encoded.shape[0], -1))
        return conv_encoded, conv_decoded, cls