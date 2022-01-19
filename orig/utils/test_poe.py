import torch
from torch import nn



class ProductOfExperts(nn.Module):
    """Return parameters for product of independent experts.
    See https://arxiv.org/pdf/1410.7827.pdf for equations.

    @param mu: M x D for M experts
    @param logvar: M x D for M experts
    """
    def forward(self, mu, var, w=None, eps=1e-8):

        w = torch.ones_like(mu) if w is None else w


        #var = torch.exp(logvar)
        T = 1 / (var + eps)  # precision of i-th Gaussian expert at point x
        #T=T*w

        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        #pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_var #pd_logvar¥



def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = torch.autograd.Variable(torch.zeros(size))
    var = torch.autograd.Variable((torch.ones(size)))

    if use_cuda:
        mu, logvar = mu.cuda(), var.cuda()
    return mu, var



def test():
    mu= torch.randn((1,1,2))*10
    var = torch.randn((1,1,2))*2

    prior = prior_expert((1,1,2))

    prior_mu = torch.zeros((1,1,2))
    prior_var = torch.ones_like(prior_mu)

    muall = torch.stack([prior[0], mu], dim=0)
    varall = torch.stack([prior[1], var], dim=0)
    poe = ProductOfExperts()
    out = poe( muall, varall)

    print(out[0][0,:])
    print(out[1][0, :])




##
    muall = torch.stack([prior_mu, mu], dim=0)
    varall = torch.stack([prior_var, var], dim=0)
    poe = ProductOfExperts()
    out = poe(muall, varall)

    print(out[0][0, :])
    print(out[1][0, :])

    print('Done')

if(__name__=='__main__'):
    test()


