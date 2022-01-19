import torch
import torch.nn as nn

from models.model_blocks import RNFLEncoder, RNFLDecoder, Encoder, Decoder
from models.model_blocks import VFTEncoder, VFTDecoder


# @todo implement info vae loss /mmd vae loss
class VAE(nn.Module):
    def __init__(self, latent_dim, type):
        super(VAE, self).__init__()

        self.latent_dim = latent_dim
        assert type in ['vft', 'rnfl', 'gcl'], 'invalid type'

        if (type == 'vft'):
            zfc_size = 32

            # self.encoder = VFTEncoder(latent_dim=latent_dim)
            self.encoder = VFTEncoder(z_size=zfc_size)
            self.decoder = VFTDecoder(z_size=latent_dim)
        else:
            zfc_size = 64

            #self.encoder = Encoder(input_shape=(32, 32), channel_in=1, z_size=zfc_size,
            #                       num_downsamples=4)
            self.decoder = Decoder(z_size=latent_dim, channel_out=1, num_upsamples=4,
                                   image_size=32)

            self.encoder = RNFLEncoder(latent_dim=None,  z_size=zfc_size, rnfl_imgChans=1, rnfl_fBase=32)
            #self.decoder = RNFLDecoder(z_size=latent_dim,rnfl_imgChans=1,rnfl_fBase=32)

        self.l_mu_logvar = nn.Linear(in_features=zfc_size, out_features=latent_dim * 2)

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def forward(self, x):

        #mu, logvar = self.infer(x)

        out = self.infer(x)
        #out = self.l_mu_logvar(out)
        mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]


        pred_z = self.reparametrize(mu, logvar)
        pred_x = self.decoder(pred_z)

        return [pred_x], [mu, logvar]

    def infer(self, x):
        """
        Posterior inference
        :param x:
        :return:
        """
        zfc_out = self.encoder(x)
        out = self.l_mu_logvar(zfc_out)
        #mu, logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
        #return mu, logvar
        return out

    def __call__(self, *args, **kwargs):
        return super(VAE, self).__call__(*args, **kwargs)

