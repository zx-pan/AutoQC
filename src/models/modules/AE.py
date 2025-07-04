import torch
from torch import nn

class CNN_Encoder(nn.Module):
    def __init__(self, output_size, input_channels=3):
        super(CNN_Encoder, self).__init__()

        self.channel_mult = 128  # Increase the number of filters to enhance capacity

        # Convolutions
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=self.channel_mult * 1,
                      kernel_size=4, stride=2, padding=1),  # Halves input size
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 1, self.channel_mult * 2, 4, 2, 1),  # Halves again
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 2, self.channel_mult * 4, 4, 2, 1),  # Halves again
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 4, self.channel_mult * 8, 4, 2, 1),  # Halves again
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(self.channel_mult * 8, self.channel_mult * 16, 4, 2, 1),  # Halves again
            nn.BatchNorm2d(self.channel_mult * 16),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Fully connected layers
        self.output_size = output_size
        self.flat_fts = None
        self.fc = nn.Linear(1, output_size)  # Placeholder, will adjust dynamically

    def get_flat_fts(self, x):
        x = self.conv(x)
        return int(x.view(x.size(0), -1).size(1))

    def forward(self, x):
        if self.flat_fts is None:
            self.flat_fts = self.get_flat_fts(x)
            self.fc = nn.Linear(self.flat_fts, self.output_size).to(x.device)

        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten the output from conv layers
        x = self.fc(x)  # Fully connected layer
        return x


class CNN_Decoder(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(CNN_Decoder, self).__init__()
        self.input_dim = embedding_size
        self.channel_mult = 128  # Match the increased filters in the encoder

        # Fully connected layer to expand latent representation
        self.fc_output_dim = None  # To be calculated dynamically
        self.fc = nn.Linear(embedding_size, 1)  # Placeholder, will adjust dynamically

        # Deconvolutions
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(self.channel_mult * 16, self.channel_mult * 8, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult * 8, self.channel_mult * 4, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult * 4, self.channel_mult * 2, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult * 2, self.channel_mult * 1, 4, 2, 1),
            nn.BatchNorm2d(self.channel_mult * 1),
            nn.ReLU(True),
            nn.ConvTranspose2d(self.channel_mult * 1, input_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x, encoder_feature_shape):
        if self.fc_output_dim is None:
            self.fc_output_dim = encoder_feature_shape[1] * encoder_feature_shape[2] * encoder_feature_shape[3]
            self.fc = nn.Linear(self.input_dim, self.fc_output_dim).to(x.device)

        x = self.fc(x)
        x = x.view(-1, encoder_feature_shape[1], encoder_feature_shape[2], encoder_feature_shape[3])
        x = self.deconv(x)
        return x


class AE(nn.Module):
    def __init__(self, embedding_size, input_channels=3):
        super(AE, self).__init__()
        self.encoder = CNN_Encoder(embedding_size, input_channels)
        self.decoder = CNN_Decoder(embedding_size, input_channels)

    def forward(self, x, t, cond):
        encoded = self.encoder(x)
        encoder_feature_shape = self.encoder.conv(x).shape  # Get feature map size after encoder
        return self.decoder(encoded, encoder_feature_shape)


if __name__ == '__main__':
    input_size = (3, 256, 256)  # Increased input size (e.g., larger RGB images)
    embedding_size = 256  # Increase embedding size for better power

    ae = AE(embedding_size, input_channels=input_size[0])
    x = torch.randn(1, *input_size)  # Create a random input
    output = ae(x)

    print(f"Input size: {x.size()}")
    print(f"Output size: {output.size()}")  # Output should match input size