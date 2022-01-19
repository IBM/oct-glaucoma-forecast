import torch
import torch.nn as nn

from models.multiodal_vae import prior_expert, ProductOfExperts, MixtureOfExperts


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0

    def forward(self, t, x):
        """

        :param t:  (N,1)
        :param x: (N,latent_dim)
        :return: (N,latent_dim)
        """
        x = torch.cat([x, t.view(1, -1)], dim=1)
        self.nfe += 1
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out



class RecognitionGRU(nn.Module):
    """
    GRU cell with encoder that takes image as input and hidden state  and produces the the hidden state and output
    where the dimension of outpt is latent_dim*2
    """

    def __init__(self, type, z_size, latent_dim, nhidden=64):
        super(RecognitionGRU, self).__init__()



        self.grucell = nn.GRUCell(z_size, nhidden)
        self.nhidden = nhidden

        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def load_inner_encoder(self, encoder_inner_weight_path):
        self.encoder.load_state_dict(torch.load(encoder_inner_weight_path))

    def forward(self, x, hc, ):
        h, c = hc
        #x1 = self.encoder(x)
        h = self.grucell(x, h)
        out = self.h2o(h)
        return out, (h, c)

    def initHidden(self, nbatch):
        return (torch.zeros(nbatch, self.nhidden), torch.zeros(nbatch, self.nhidden))

    def get_params_finetune(self):
        """
        Paraeters other than the encoder as the encoder is pre trained
        :return:
        """
        return  list(self.grucell.parameters()) + list(self.h2o.parameters())





class RecognitionODEGRU(nn.Module):

    def __init__(self, type, latent_dim, z_size=128, device=None, ode_solver=None):
        super(RecognitionODEGRU, self).__init__()

        self.use_odernn = ode_solver is not None
        self.ode_solver = ode_solver
        nhidden1 = z_size #latent_dim * 2

        self.rnn = RecognitionGRU(type, z_size=z_size,
                                  latent_dim=latent_dim, nhidden=nhidden1)
        self.hfunc = LatentODEfunc(latent_dim=self.rnn.nhidden, nhidden=self.rnn.nhidden * 2)

        self.device = device

    def initialize_temproary_vars(self, batch_size):

        h, c = self.rnn.initHidden(nbatch=batch_size)
        return h.to(self.device), c.to(self.device)

    def forward(self, x, ts):

        h, c = self.initialize_temproary_vars(batch_size=x.shape[0])
        for t in reversed(range(x.size(1))):
            obs = x[:, t, :]

            if (t < x.size(1) - 1 and self.use_odernn):  # the first cell, there is not hidden dynamics
                ode_solver, method = self.ode_solver
                h_next = [ode_solver(self.hfunc, torch.unsqueeze(hi, dim=0), ti[torch.arange(t + 1, t - 1, -1)],
                                     method=method) for hi, ti in zip(h, ts)]
                h = torch.cat(h_next, dim=1)[1]  # index 1 contains the forecasted value

            out, (h, c) = self.rnn(obs, (h, c))
        return out, (h, c)


    def get_params_finetune(self):
        """
        Paraeters other than the encoder as the encoder is pre trained
        :return:
        """
        return list(self.rnn.get_params_finetune()) + list(self.hfunc.parameters())







class MultimodalLatentODE(nn.Module):
    def __init__(self, latent_dim, device=None, ode_solver=None, expert='poe'):
        super(MultimodalLatentODE, self).__init__()

        self.rnn_rnfl = RecognitionODEGRU(type='rnfl', latent_dim=latent_dim,
                                          device=device,
                                          ode_solver=None, z_size=64)

        self.rnn_gcl = None #RecognitionODEGRU(type='gcl', latent_dim=latent_dim,
        #                                 device=device,
        #                                 ode_solver=ode_solver, z_size=64)

        self.rnn_vft = RecognitionODEGRU(type='vft', latent_dim=latent_dim,
                                         device=device,
                                         ode_solver=None, z_size=32)

        self.forecast_ode = LatentODEfunc(latent_dim=latent_dim)

        self.decoder_rnfl = Decoder(z_size=latent_dim, channel_out=1, num_upsamples=4, image_size=64)
        self.decoder_gcl = None#Decoder(z_size=latent_dim, channel_out=1, num_upsamples=4, image_size=64)
        self.decoder_vft = None#VFTDecoder(z_size=latent_dim)

        self.latent_dim = latent_dim
        self.ode_solver = ode_solver
        self.device = device


        assert expert in ['moe', 'poe'], 'invalid option for experts'
        self.experts = MixtureOfExperts() if expert == "moe" else ProductOfExperts()


        self.metadata_input_dim = None
        self.use_cuda = 'cuda' in device.type
        self.generalized_poe = False


    def get_params_finetune(self):

        return list(self.rnn_rnfl.get_params_finetune()) + list(self.rnn_vft.get_params_finetune()) + list(self.forecast_ode.parameters())

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu


    def infer(self, ts, x_list):
        x_rnfl, x_gcl, x_vft = x_list
        mu, logvar = self.get_posterior(x_rnfl=x_rnfl, x_gcl=x_gcl, x_vft=x_vft, ts=ts)
        return mu, logvar


    #disable batch prediction for deubugging, since subset of output could be None, this needs to be taken into
    #account while batching
    def forward_(self, ts, x_list, batch_size=None):
        """

        :param ts:
        :param x_list:
        :param batch_size: when None, then batching is not performed
        :return:
        """

        if (batch_size is None):
            return self.forward_(ts, x_list)

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

        idx = list(range(ts.shape[0]))
        out_pred = []
        out_latent = []

        for chunk in batch(idx, batch_size):
            ts_i = ts[chunk]
            input_i = [inp[chunk] if inp is not None else None for inp in x_list]
            pred_i, latent_i = self.forward_(ts_i, input_i)
            out_pred.append(pred_i)
            out_latent.append(latent_i)

        return collect(out_pred), collect(out_latent)

    def forward(self, ts, x_list):
        """

        :param x: (N,t,1,H,W)
        :param t: (N,t)
        :param mod_select array of Boolean to indicate which to use, When None all will be used
        :return:
        """

        x_rnfl, x_gcl, x_vft = x_list

        mu, logvar = self.get_posterior(ts, x_rnfl=x_rnfl, x_gcl=x_gcl, x_vft=x_vft)

        z0 = self.reparametrize(mu, logvar)

        z0 = torch.unsqueeze(z0, dim=1)
        ode_solver, method = self.ode_solver
        method = 'euler'#'rk4'
        pz = [ode_solver(self.forecast_ode, z0i, tsi, method=method) for z0i, tsi in zip(z0, ts)]

        pred_z = torch.cat(pz, dim=1).permute(1, 0, 2)

        na, nb, nc = pred_z.shape
        pred_z_ = pred_z.reshape(na * nb, nc)
        pred_rnfl = self.decoder_rnfl(pred_z_) if x_rnfl is not None else None
        pred_gcl = self.decoder_gcl(pred_z_) if x_gcl is not None else None
        pred_vft = self.decoder_vft(pred_z_) if x_vft is not None else None

        batch_size, time_size_ = self.get_batch_time_size(x_rnfl, x_gcl, x_vft)

        time_size = ts.shape[1]
        pred_rnfl = self.match_size(pred_rnfl, batch_size, time_size)
        pred_gcl = self.match_size(pred_gcl, batch_size, time_size)
        pred_vft = self.match_size(pred_vft, batch_size, time_size)

        # last time slice denotes the forecasted values. discard GCL and VFT for evaluation??

        return [pred_rnfl, pred_gcl, pred_vft], [mu, logvar]

    def match_size(self, x, batch_size, time_size):

        if(x is not None):
            Bnt, c, H, W = x.shape

            assert Bnt == batch_size * time_size, 'Parameters not matching'
            return x.view(batch_size, time_size, c, H, W)
        else:
            return x

    def get_posterior(self, ts, x_rnfl=None, x_gcl=None, x_vft=None, x_proj=None):

        # define universal expert
        batch_size, _ = self.get_batch_time_size(x_rnfl, x_gcl, x_vft, x_proj)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert

        mu_all = []
        logvar_all = []
        w_all = []

        if x_rnfl is not None:
            out, (h, c) = self.rnn_rnfl(x_rnfl, ts)
            rnfl_mu, rnfl_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(rnfl_mu)
            logvar_all.append(rnfl_logvar)

        if x_gcl is not None:
            out, (h, c) = self.rnn_gcl(x_gcl, ts)
            gcl_mu, gcl_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(gcl_mu)
            logvar_all.append(gcl_logvar)

        if x_vft is not None:
            out, (h, c) = self.rnn_vft(x_vft, ts)
            vft_mu, vft_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim]
            mu_all.append(vft_mu)
            logvar_all.append(vft_logvar)

        # product of experts to combine gaussians

        #if (len(mu_all) > 1):  # prior expert is not required for single modality forward pass
        mu_prior, logvar_prior = prior_expert((batch_size, self.latent_dim),
                                              use_cuda=use_cuda)
        mu_all.insert(0,mu_prior)
        logvar_all.insert(0, logvar_prior)

        mu = torch.cat([m.unsqueeze(0) for m in mu_all], dim=0)
        logvar = torch.cat([lv.unsqueeze(0) for lv in logvar_all], dim=0)
        #w = torch.cat([wi.unsqueeze(0) for wi in w_all], dim=0)

        #if (not self.generalized_poe): w = None

        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def get_batch_time_size(self, *args):
        for arg in args:
            if arg is not None:
                return arg.size(0), arg.size(1)
