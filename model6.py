import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    """ ResNet based """
    def __init__(self):
        # Initialize superclass
        super(ConvAutoEncoder, self).__init__()
        # Conv encoding layers
        self.e1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.e2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.e3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.e4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=0)
        self.e5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        # Transpose conv. decoding layers
        self.d1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.d2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=0)
        self.d3 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.d4 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0,
                                     output_padding=1)
        self.d5 = nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1)
        # Autoencoder activation functions
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Batch norm for convolutional layers
        self.norm32_1 = nn.BatchNorm2d(32)
        self.norm32_2 = nn.BatchNorm2d(32)
        self.norm64_1 = nn.BatchNorm2d(64)
        self.norm64_2 = nn.BatchNorm2d(64)
        self.norm64_3 = nn.BatchNorm2d(64)
        self.norm64_4 = nn.BatchNorm2d(64)
        self.norm128_1 = nn.BatchNorm2d(128)
        self.norm128_2 = nn.BatchNorm2d(128)
        self.norm256 = nn.BatchNorm2d(256)
        # Array of autoencoder's layers
        self.autoencoder = [self.e1, self.e2, self.e3, self.e4, self.e5, self.e2,
                            self.d1, self.d2, self.d3, self.d4, self.d5]
        # Classifier
        self.l1 = nn.Linear(9216, 1024)
        self.l2 = nn.Linear(1024, 10)
        self.drop = nn.Dropout(0.25)
        self.lsoft = nn.LogSoftmax(dim=1)
        """
        self.classifier = nn.Sequential(
            nn.Linear(9216, 1024),
            nn.Dropout(0.25),
            nn.ReLU(),
            nn.Linear(1024, 10),
            nn.LogSoftmax(dim=1)
        )"""

    def forward(self, x):
        # Encoding
        # print('input', x.shape)
        e1 = self.e1(x)
        out_e1 = self.relu(self.norm32_1(e1))
        e2 = self.e2(out_e1)
        out_e2 = self.relu(self.norm64_1(e2))
        e3 = self.e3(out_e2)
        out_e3 = self.relu(self.norm64_2(e3))
        e4 = self.e4(out_e3)
        out_e4 = self.relu(self.norm128_1(e4))
        e5 = self.e5(out_e4)
        out_e5 = self.relu(self.norm256(e5))

        # Decoding
        d1 = self.d1(out_e5)
        out_d1 = torch.add(d1, e4)
        out_d1 = self.relu(self.norm128_2(out_d1))
        d2 = self.d2(out_d1)
        out_d2 = torch.add(d2, e3)
        out_d2 = self.relu(self.norm64_3(out_d2))
        d3 = self.d3(out_d2)
        out_d3 = torch.add(d3, e2)
        out_d3 = self.relu(self.norm64_4(out_d3))
        d4 = self.d4(out_d3)
        out_d4 = torch.add(d4, e1)
        out_d4 = self.relu(self.norm32_2(out_d4))
        d5 = self.d5(out_d4)
        out_d5 = torch.add(d5, x)
        out_d5 = self.sigmoid(out_d5)

        # Classifier
        # print(f'# input dimensions for the linear layer {conv_encoded.view(conv_encoded.shape[0], -1).shape}')
        """cls = self.classifier(out_e5.view(out_e5.shape[0], -1))"""
        c1 = self.l1(out_e5.view(out_e5.shape[0], -1))
        d1 = self.relu(self.drop(c1))
        cls = self.lsoft(self.l2(d1))
        return out_e5, out_d5, cls

    def init_xavier(self, verbose=False):
        with torch.no_grad():
            # Init encoder
            nn.init.xavier_normal_(self.e1.weight)
            nn.init.zeros_(self.e1.bias)
            nn.init.xavier_normal_(self.e2.weight)
            nn.init.zeros_(self.e2.bias)
            nn.init.xavier_normal_(self.e3.weight)
            nn.init.zeros_(self.e3.bias)
            nn.init.xavier_normal_(self.e4.weight)
            nn.init.zeros_(self.e4.bias)
            nn.init.xavier_normal_(self.e5.weight)
            nn.init.zeros_(self.e5.bias)
            # Init decoder
            nn.init.xavier_normal_(self.d1.weight)
            nn.init.zeros_(self.d1.bias)
            nn.init.xavier_normal_(self.d2.weight)
            nn.init.zeros_(self.d2.bias)
            nn.init.xavier_normal_(self.d3.weight)
            nn.init.zeros_(self.d3.bias)
            nn.init.xavier_normal_(self.d4.weight)
            nn.init.zeros_(self.d4.bias)
            nn.init.xavier_normal_(self.d5.weight)
            nn.init.zeros_(self.d5.bias)

            # Init classifier
            # nn.init.kaiming_normal_(self.l1.weight, nonlinearity='relu')
            nn.init.zeros_(self.l1.bias)
            # nn.init.kaiming_normal_(self.l2.weight, nonlinearity='relu')
            nn.init.zeros_(self.l2.bias)

            if verbose:
                print("Parameters initialized using xavier initialization")

