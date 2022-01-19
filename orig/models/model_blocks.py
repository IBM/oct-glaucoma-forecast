import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import create_resnet50


# encoder block (used in encoder and discriminator)
class EncoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out, dropout_rate=None):
        super(EncoderBlock, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=2,
                              bias=True)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

        self.dropout = None if dropout_rate is None else nn.Dropout2d(dropout_rate)

    def forward(self, ten, out=False, t=False):
        # here we want to be able to take an intermediate output for reconstruction error
        if out:
            ten = self.conv(ten)
            ten_out = ten
            ten = self.bn(ten)
            ten = Swish()(ten)  # F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = Swish()(ten)  # F.relu(ten, True)
            if (self.dropout): ten = self.dropout(ten)
            return ten


class EncoderBlockMaxPool(nn.Module):
    def __init__(self, channel_in, channel_out, dropout_rate=None):
        super(EncoderBlockMaxPool, self).__init__()
        # convolution to halve the dimensions
        self.conv = nn.Conv2d(in_channels=channel_in, out_channels=channel_out, kernel_size=5, padding=2, stride=1,
                              bias=True)
        self.bn = nn.BatchNorm2d(num_features=channel_out, momentum=0.9)

        self.dropout = None if dropout_rate is None else nn.Dropout2d(dropout_rate)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = F.relu(ten, True)
        if (self.dropout): ten = self.dropout(ten)
        ten = nn.MaxPool2d(kernel_size=2)(ten)
        return ten


class Encoder(nn.Module):
    def __init__(self, input_shape, channel_in=3, z_size=128, num_downsamples=3, dropout_rate=None, latent_dim=None):
        """
        if latent dim is not None, then the last layer output will be 2* latent_dim, otherwise it will be z_size
        :param input_shape:
        :param channel_in:
        :param z_size:
        :param num_downsamples:
        :param dropout_rate:
        :param latent_dim:
        """
        super(Encoder, self).__init__()
        self.size = channel_in
        self.latent_dim = latent_dim

        layers_list = []
        # the first time 3->64, for every other double the channel size
        for i in range(num_downsamples):
            if i == 0:
                layers_list.append(EncoderBlock(channel_in=self.size, channel_out=32, dropout_rate=dropout_rate))
                self.size = 32
            else:
                layers_list.append(
                    EncoderBlock(channel_in=self.size, channel_out=self.size * 2, dropout_rate=dropout_rate))
                self.size *= 2

        final_shape = [int(s / 2 ** num_downsamples) for s in input_shape]
        # final shape Bx256x8x8
        self.conv = nn.Sequential(*layers_list)
        self.fc = nn.Sequential(
            nn.Linear(in_features=final_shape[0] * final_shape[1] * self.size, out_features=z_size, bias=True),
            nn.BatchNorm1d(num_features=z_size, momentum=0.9),
            Swish())  # nn.ReLU(True)

        if (self.latent_dim is not None):
            # two linear to get the mu vector and the diagonal of the log_variance
            self.l_mu_logvar = nn.Linear(in_features=z_size, out_features=latent_dim * 2)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        if (self.latent_dim is None):
            return ten
        else:
            return self.l_mu_logvar(ten)

    def __call__(self, *args, **kwargs):
        return super(Encoder, self).__call__(*args, **kwargs)


class EncoderResnet(nn.Module):
    def __init__(self, input_shape, channel_in=3, z_size=128, num_downsamples=None, dropout_rate=None):
        super(EncoderResnet, self).__init__()
        self.resnet = create_resnet50(include_top=False, channels_in=channel_in)
        self.fc = nn.Linear(in_features=2048, out_features=z_size)

    def forward(self, x, mdata=None):
        x = self.resnet(x)
        x = self.fc(x)
        return x

    def __call__(self, *args, **kwargs):
        return super(EncoderResnet, self).__call__(*args, **kwargs)


# decoder block (used in the decoder)
class DecoderBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(DecoderBlock, self).__init__()
        # transpose convolution to double the dimensions
        self.conv = nn.ConvTranspose2d(channel_in, channel_out, kernel_size=5, padding=2, stride=2, output_padding=1,
                                       bias=True)
        self.bn = nn.BatchNorm2d(channel_out, momentum=0.9)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = self.bn(ten)
        ten = Swish()(ten)  # F.relu(ten, True)
        return ten


class Decoder(nn.Module):
    def __init__(self, z_size, image_size, channel_out=3, num_upsamples=3):
        super(Decoder, self).__init__()
        # numer of feature maps to start with
        n_start_fm = num_upsamples * 32 * 2
        # start from B*z_size
        self.start_fm_size = int(image_size // (2 ** num_upsamples))
        assert image_size == 2 ** num_upsamples * self.start_fm_size, 'Image size and num_upsamples are not consistent'

        num_out_features = self.start_fm_size * self.start_fm_size * n_start_fm
        self.fc = nn.Sequential(nn.Linear(in_features=z_size,
                                          out_features=num_out_features,
                                          bias=True), nn.BatchNorm1d(num_features=num_out_features,
                                                                     momentum=0.9), Swish())
        self.size = n_start_fm
        layers_list = []
        for i in range(1, num_upsamples + 1):
            layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // i))
            self.size = self.size // i

        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=channel_out, kernel_size=5, stride=1, padding=2)
            #    , nn.Sigmoid()
        ))
        # Note no sigmpoid here, see train script
        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, self.start_fm_size, self.start_fm_size)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


vft_fBase = 32
vft_imgChans = 1


class VFTDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, z_size):
        super(VFTDecoder, self).__init__()
        self.dec = nn.Sequential(

            # input size: (z_size x 1 x 1
            nn.ConvTranspose2d(z_size, vft_fBase * 4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(vft_fBase * 4),
            Swish(),
            # size: (fBase * 4) x 3 x 3

            nn.ConvTranspose2d(vft_fBase * 4, vft_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(vft_fBase * 2),
            Swish(),
            # size: (fBase * 2) x 6 x 6

            nn.ConvTranspose2d(vft_fBase * 2, vft_fBase, 4, 2, 1, bias=False),
            nn.BatchNorm2d(vft_fBase),
            Swish(),
            # size: (fBase * 1) x 12 x 12
            nn.ConvTranspose2d(vft_fBase, vft_imgChans, 3, 1, 0, bias=False)
            # Output size: 1 x 14 x 14 (no sigmoid)

        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        return out  # Output size: 1 x 14 x 14

    def __call__(self, *args, **kwargs):
        return super(VFTDecoder, self).__call__(*args, **kwargs)


class VFTEncoder(nn.Module):
    """ Generate latent parameters for  image data or feature  """

    def __init__(self, latent_dim=None, z_size=None):
        super(VFTEncoder, self).__init__()
        assert latent_dim is not None or z_size is not None, 'One of the latent_dim or z_size should be provided'

        self.enc = nn.Sequential(
            # input size: 1 x 14 x 14
            nn.Conv2d(vft_imgChans, vft_fBase, 3, 1, 1, bias=False),
            nn.BatchNorm2d(vft_fBase),
            Swish(),
            # size: (fBase) x 14 x 14

            nn.Conv2d(vft_fBase, vft_fBase * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(vft_fBase * 2),
            Swish(),
            # size: (fBase*2) x 7 x 7
            nn.Conv2d(vft_fBase * 2, vft_fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(vft_fBase * 4),
            Swish()
            # size: (fBase*4) x 3 x 3 ?? check

        )

        if (z_size is not None):
            self.fc = nn.Sequential(
                nn.Conv2d(vft_fBase * 4, z_size, 3, 1, 0, bias=False),
                nn.BatchNorm2d(num_features=z_size, momentum=0.9),
                Swish())  # nn.ReLU(True)
        if (latent_dim is not None):
            self.c1 = nn.Conv2d(vft_fBase * 4, latent_dim, 3, 1, 0, bias=False)
            self.c2 = nn.Conv2d(vft_fBase * 4, latent_dim, 3, 1, 0, bias=False)

        self.latent_dim = latent_dim

    def forward(self, x):
        e = self.enc(x)
        if (self.latent_dim is not None):
            return torch.cat([self.c1(e).squeeze(), self.c2(e).squeeze()],
                             dim=1)  # F.softplus(self.c2(e)).squeeze() + eta
        else:
            return self.fc(e).squeeze()

    def __call__(self, *args, **kwargs):
        return super(VFTEncoder, self).__call__(*args, **kwargs)



class RNFLEncoder(nn.Module):
    """ Generate latent parameters for  image data or feature  """

    def __init__(self, latent_dim, z_size=None, rnfl_imgChans=1, rnfl_fBase=32):
        super(RNFLEncoder, self).__init__()
        self.enc = nn.Sequential(
            # input size: 1 x 64 x 64
            nn.Conv2d(rnfl_imgChans, rnfl_fBase, 3, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase),
            Swish(),
            # size: (fBase) x 32 x 32

            nn.Conv2d(rnfl_fBase, rnfl_fBase * 2, 3, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 2),
            Swish(),
            # size: (fBase*2) x 16 x 16

            nn.Conv2d(rnfl_fBase * 2, rnfl_fBase * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 4),
            Swish(),
            # size: (fBase*4) x 8 x 8

            nn.Conv2d(rnfl_fBase * 4, rnfl_fBase * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 8),
            Swish(),
            # size: (fBase*8) x 4 x 4

        )

        if (z_size is not None):
            kernel_size = 2
            self.fc = nn.Sequential(
                nn.Conv2d(rnfl_fBase * 8, z_size, kernel_size, 1, 0, bias=False),
                nn.BatchNorm2d(num_features=z_size, momentum=0.9),
                Swish())  # nn.ReLU(True)
        if (latent_dim is not None):
            self.c1 = nn.Conv2d(rnfl_fBase * 8, latent_dim, 4, 1, 0, bias=False)
            self.c2 = nn.Conv2d(rnfl_fBase * 8, latent_dim, 4, 1, 0, bias=False)

        self.latent_dim = latent_dim

    def forward(self, x):
        e = self.enc(x)
        if (self.latent_dim is not None):
            return torch.cat([self.c1(e).squeeze(), self.c2(e).squeeze()],
                             dim=1)  # F.softplus(self.c2(e)).squeeze() + eta
        else:
            return self.fc(e).squeeze()

    def __call__(self, *args, **kwargs):
        return super(RNFLEncoder, self).__call__(*args, **kwargs)



class RNFLDecoder(nn.Module):
    """ Generate a SVHN image given a sample from the latent space. """

    def __init__(self, z_size, rnfl_imgChans =1, rnfl_fBase=32):
        super(RNFLDecoder, self).__init__()
        self.dec = nn.Sequential(

            # input size: (z_size x 1 x 1
            nn.ConvTranspose2d(z_size, rnfl_fBase * 8, 6, 1, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 8),
            Swish(),
            # size: (fBase * 4) x 4 x4

            nn.ConvTranspose2d(rnfl_fBase * 8, rnfl_fBase * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 4),
            Swish(),
            # size: (fBase * 2) x 8 x 8

            nn.ConvTranspose2d(rnfl_fBase * 4, rnfl_fBase * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 2),
            Swish(),
            # size: (fBase * 1) x 16 x 16

            nn.ConvTranspose2d(rnfl_fBase * 2, rnfl_fBase * 1, 4, 2, 1, bias=False),
            nn.BatchNorm2d(rnfl_fBase * 1),
            Swish(),
            # size: (fBase * 1) x 32 x 32


            nn.ConvTranspose2d(rnfl_fBase, rnfl_imgChans, 4, 2, 1, bias=False)
            # Output size: 1 x 64 x 64 (no sigmoid)

        )

    def forward(self, z):
        z = z.unsqueeze(-1).unsqueeze(-1)  # fit deconv layers
        out = self.dec(z.view(-1, *z.size()[-3:]))
        return out  # Output size: 1 x 14 x 14

    def __call__(self, *args, **kwargs):
        return super(RNFLDecoder, self).__call__(*args, **kwargs)



class AuxilNetClassification(nn.Module):

    def __init__(self, input_shape, channel_in=3, num_downsamples=3):
        super(AuxilNetClassification, self).__init__()
        self.encoder = Encoder(input_shape, channel_in, z_size=1, num_downsamples=num_downsamples)
        # self.sigmoid=nn.Sigmoid()

    def forward(self, im):
        out = self.encoder(im)
        # out = self.sigmoid(out)
        return out


class AuxillaryNet(nn.Module):

    def __init__(self, input_shape, channel_in=3, num_downsamples=3, num_out=1, last_activation=nn.Sigmoid()):
        super(AuxillaryNet, self).__init__()
        self.encoder = Encoder(input_shape, channel_in, z_size=num_out, num_downsamples=num_downsamples)
        self.last_actication = last_activation

    def forward(self, im):
        out = self.encoder(im)

        if (self.last_actication is not None):
            out = self.last_actication(out)
        return out


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * torch.sigmoid(x)
