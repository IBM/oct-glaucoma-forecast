import torch
import torch.nn as nn
import torch.nn.functional as F


class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """

    def forward(self, mu, logvar, w=None, eps=1e-8):
        w = torch.ones_like(mu) if w is None else w

        var = torch.exp(logvar)
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        T = T * w

        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar


class MixtureOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M+1 x B x D for M experts
    @param logvar: M+1 x B x D for M experts
    (row 0 is always all zeros)
    """

    def forward(self, mu, logvar, eps=1e-8):
        if mu.shape[0] == 2:
            return mu[-1], logvar[-1]
        elif mu.shape[0] == 3:
            B = mu.shape[1]
            mu, logvar = torch.cat([mu[1, :B // 2], mu[2, B // 2:]]), \
                         torch.cat([logvar[1, :B // 2], logvar[2, B // 2:]])
        elif mu.shape[0] == 4:
            B = mu.shape[1]
            a = B // 3 + B - B // 3 * 3
            b = a + B // 3
            c = b + B // 3

            mu, logvar = torch.cat([mu[1, :a], mu[2, a:b], mu[3, b:]]), \
                         torch.cat([logvar[1, :a], logvar[2, a:b], logvar[3, b:]])
        else:
            raise ValueError('More than three modalities are not supported')

        return mu, logvar


def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu = torch.autograd.Variable(torch.zeros(size))
    logvar = torch.autograd.Variable(torch.log(torch.ones(size)))

    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar


# https://github.com/mhw32/multimodal-vae-public/blob/master/vision/model.py
# @todo handle sample in test as done in mvae

from models.vae import VAE


class MultimodalVAE(nn.Module):
    def __init__(self, latent_dim, device=None, expert='poe'):
        super(MultimodalVAE, self).__init__()

        self.vae_rnfl = VAE(latent_dim=latent_dim, type='rnfl')
        self.vae_gcl = None  # VAE(latent_dim=latent_dim, type='gcl')
        self.vae_vft = VAE(latent_dim=latent_dim, type='vft')

        self.latent_dim = latent_dim
        self.device = device

        assert expert in ['moe', 'poe'], 'invalid option for experts'
        self.experts = MixtureOfExperts() if expert == "moe" else ProductOfExperts()

        self.metadata_input_dim = None
        self.use_cuda = 'cuda' in device.type

    def reparametrize__(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            # return torch.normal(mu, std)
            esp = torch.randn(*mu.size()).to(self.device)
            z = mu + std * esp
            return z
        else:
            return mu

    def forward_test(self, x_list, batch_size=None):
        """
        adds singmoid activation over the output
        :param x_list:
        :param batch_size:
        :return:
        """
        maps, [mu, logvar] = self.forward(x_list=x_list, batch_size=batch_size)
        maps = [torch.sigmoid(m) for m in maps]
        return maps, [mu, logvar]

    def forward(self, x_list, batch_size=None):
        """

        :param ts:
        :param x_list:
        :param batch_size: when None, then batching is not performed
        :return:
        """

        if (batch_size is None):
            return self.forward_(x_list)

        def batch(iterable, n=1):
            l = len(iterable)
            for ndx in range(0, l, n):
                yield iterable[ndx:min(ndx + n, l)]

        def collect(out):
            """

            :param out:  list of list of ndarrays i.e [ [rnfl, gcl, vft],.....] where each rnfl, gcl and vft are ndarrays
            :return: concatenate the ndarrays in the list and produces [rnfl, gcl, vft]
            """

            collection = [[] for i in range(len(out[0]))]
            for i in range(len(out)):
                for j in range(len(out[i])):
                    collection[j].append(out[i][j])
            # each element in collecton is a list of an output. they must be concateed

            final = [torch.cat(item, dim=0) for item in collection]
            return final

        bs = self.get_batchsize(x_list)
        idx = list(range(bs))
        out_pred = []
        out_latent = []

        for chunk in batch(idx, batch_size):
            input_i = [inp[chunk] if inp is not None else None for inp in x_list]
            pred_i, latent_i = self.forward_(input_i)
            out_pred.append(pred_i)
            out_latent.append(latent_i)

        return collect(out_pred), collect(out_latent)

    def forward_(self, x_list):
        """

        :param x: (N,t,1,H,W)
        :param t: (N,t)
        :param mod_select array of Boolean to indicate which to use, When None all will be used
        :return:
        """

        x_rnfl, x_gcl, x_vft = x_list

        mu, logvar = self.get_posterior(x_rnfl=x_rnfl, x_gcl=x_gcl, x_vft=x_vft)

        pred_z_ = self.reparametrize(mu, logvar)

        pred_rnfl = self.vae_rnfl.decoder.forward(pred_z_) if x_rnfl is not None else None
        pred_gcl = self.vae_gcl.decoder.forward(pred_z_) if x_gcl is not None else None
        pred_vft = self.vae_vft.decoder.forward(pred_z_) if x_vft is not None else None

        return [pred_rnfl, pred_gcl, pred_vft], [mu, logvar]

    def infer(self, x_list):
        x_rnfl, x_gcl, x_vft = x_list
        mu, logvar = self.get_posterior(x_rnfl=x_rnfl, x_gcl=x_gcl, x_vft=x_vft)
        return mu, logvar

    def get_posterior(self, x_rnfl=None, x_gcl=None, x_vft=None, x_proj=None):

        # define universal expert
        batch_size = self.get_batchsize([x_rnfl, x_gcl, x_vft, x_proj])
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert

        mu_all = []
        logvar_all = []

        if x_rnfl is not None:
            out = self.vae_rnfl.infer(x_rnfl) #self.vae_rnfl.encoder.forward(x_rnfl)
            rnfl_mu, rnfl_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(rnfl_mu)
            logvar_all.append(rnfl_logvar)

        if x_gcl is not None:
            out = self.vae_gcl.infer(x_gcl) #self.vae_gcl.encoder.forward(x_gcl, )
            gcl_mu, gcl_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(gcl_mu)
            logvar_all.append(gcl_logvar)

        if x_vft is not None:
            out = self.vae_vft.infer(x_vft) #self.vae_vft.encoder.forward(x_vft)
            vft_mu, vft_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(vft_mu)
            logvar_all.append(vft_logvar)

        # if (len(mu_all) > 1):
        mu_prior, logvar_prior = prior_expert((batch_size, self.latent_dim),
                                              use_cuda=use_cuda)
        mu_all.insert(0, mu_prior)
        logvar_all.insert(0, logvar_prior)

        mu = torch.cat([m.unsqueeze(0) for m in mu_all], dim=0)
        logvar = torch.cat([lv.unsqueeze(0) for lv in logvar_all], dim=0)

        mu, logvar = self.experts(mu, logvar)
        # else:
        #    mu= mu_all[0]
        #    logvar = logvar_all[0]

        return mu, logvar

    def get_batchsize(self, args):
        for arg in args:
            if arg is not None:
                return arg.size(0)


class Swish(nn.Module):
    """https://arxiv.org/abs/1710.05941"""

    def forward(self, x):
        return x * F.sigmoid(x)
