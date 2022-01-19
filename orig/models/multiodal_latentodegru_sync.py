import torch
import torch.nn as nn

from models.model_blocks import Decoder, VFTDecoder, VFTEncoder, RNFLEncoder
from models.multiodal_vae import prior_expert, ProductOfExperts, MixtureOfExperts
from models.model_blocks import Swish


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        self.nfe = 0
        self.swish = Swish()

    def forward(self, t, x):
        """

        :param t:  (N,1)
        :param x: (N,latent_dim)
        :return: (N,latent_dim)
        """
        x = torch.cat([x, t.view(1, -1)], dim=1)
        self.nfe += 1
        out = self.fc1(x)
        out = self.swish(out)
        out = self.fc2(out)
        out = self.swish(out)
        out = self.fc3(out)
        return out


class RecognitionGRU(nn.Module):
    """
    GRU cell with encoder that takes image as input and hidden state  and produces the the hidden state and output
    where the dimension of outpt is latent_dim*2
    """

    def __init__(self, type, z_size, latent_dim, nhidden=64):
        super(RecognitionGRU, self).__init__()

        if (type == 'vft'):
            self.encoder = VFTEncoder(latent_dim=None, z_size=z_size)
        else:
            # self.encoder = Encoder(input_shape=(32, 32), channel_in=1, z_size=z_size,
            #              num_downsamples=4, latent_dim=None)
            self.encoder = RNFLEncoder(latent_dim=None, z_size=z_size, rnfl_imgChans=1, rnfl_fBase=32)

        self.grucell = nn.GRUCell(z_size, nhidden)
        self.nhidden = nhidden

        self.h2o = nn.Linear(nhidden, latent_dim * 2)

    def load_inner_encoder(self, encoder_inner_weight_path):
        self.encoder.load_state_dict(torch.load(encoder_inner_weight_path))

    def forward(self, x, hc, ):
        h, c = hc
        x1 = self.encoder(x)
        h = self.grucell(x1, h)
        h = Swish()(h)
        out = self.h2o(h)
        return out, (h, c)

    def initHidden(self, nbatch):
        return (torch.zeros(nbatch, self.nhidden), torch.zeros(nbatch, self.nhidden))

    def get_params_finetune(self):
        """
        Paraeters other than the encoder as the encoder is pre trained
        :return:
        """
        return list(self.grucell.parameters()) + list(self.h2o.parameters())


'''
LSTM decoder part
h - hidden state
c  - cell state
'''


class DecoderGRU(nn.Module):

    def __init__(self, z_size, size, image_size, channel_out=3, num_upsamples=3, nhidden=64):
        super(DecoderGRU, self).__init__()

        self.mdata_net = None
        metadata_hidden_dim = 0

        self.lstmcell = nn.GRUCell(z_size + metadata_hidden_dim, nhidden)
        self.decoder = Decoder(nhidden, size, image_size, channel_out, num_upsamples)

        self.nhidden = nhidden

    def forward(self, x, hc):
        h, c = hc

        h = self.lstmcell(x, h)
        out = self.decoder(h)
        return out, (h, c)

    def initHidden(self, nbatch):
        return (torch.zeros(nbatch, self.nhidden), torch.zeros(nbatch, self.nhidden))




class RecognitionODEGRU(nn.Module):

    def __init__(self, type, latent_dim,  z_size=128, device=None, ode_solver=None, ):
        super(RecognitionODEGRU, self).__init__()

        self.use_odernn = ode_solver is not None
        self.ode_solver = ode_solver
        nhidden1 = z_size  # latent_dim * 2

        self.rnn = RecognitionGRU(type, z_size=z_size,
                                  latent_dim=latent_dim, nhidden=nhidden1)
        self.hfunc =  LatentODEfunc(latent_dim=self.rnn.nhidden, nhidden=self.rnn.nhidden * 2)

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
                                     method=method, rtol=0.001, atol=0.0001) for hi, ti in zip(h, ts)]
                h = torch.cat(h_next, dim=1)[1]  # index 1 contains the forecasted value

            out, (h, c) = self.rnn(obs, (h, c))
        return out, (h, c)

    def get_params_finetune(self):
        """
        Paraeters other than the encoder as the encoder is pre trained
        :return:
        """
        return list(self.rnn.get_params_finetune()) + list(self.hfunc.parameters())


class DecoderODEGRU(nn.Module):

    def __init__(self, z_size, size, image_size, channel_out=3, num_upsamples=3, nhidden=64, device=None,
                 ode_solver=None, metadata_input_dim=None):
        super(DecoderODEGRU, self).__init__()

        self.use_odernn = ode_solver is not None
        self.ode_solver = ode_solver

        self.rnn = DecoderGRU(z_size, size, image_size, channel_out=channel_out, num_upsamples=num_upsamples,
                              nhidden=nhidden)

        self.hfunc = LatentODEfunc(latent_dim=self.rnn.nhidden, nhidden=self.rnn.nhidden * 2)

        self.device = device

    def initialize_temproary_vars(self, batch_size):

        h, c = self.rnn.initHidden(nbatch=batch_size)
        return h.to(self.device), c.to(self.device)

    def forward(self, pred_z, ts):

        pred_x = []
        # Note:  reversed -> torch.arange(t+1, t - 1, -1)  & t < pred_z.size(1) -1
        # Note : no reversed -> torch.arange(t-1, t + 1, +1) & t >0

        h, c = self.initialize_temproary_vars(batch_size=pred_z.shape[0])
        for t in (range(pred_z.size(1))):
            # mdatai = None  # mdata[:, t, :] if self.metadata_input_dim is not None else None
            if (t > 0 and self.use_odernn):  # in the first cell, there is no hidden dynamics i.e  h1'= h0
                h_next = [
                    self.ode_solver(self.hfunc, torch.unsqueeze(hi, dim=0), ti[torch.arange(t - 1, t + 1, +1)],
                                    method='rk4') for hi, ti in zip(h, ts)]
                h = torch.cat(h_next, dim=1)[1]  # index 1 contains the forecasted value

            out, (h, c) = self.rnn.forward(pred_z[:, t, :], (h, c))
            pred_x.append(out)

        # pred_x = list(reversed(pred_x))
        pred_x = torch.stack(pred_x, dim=1)
        return pred_x, (h, c)


class NODE:

    def __init__(self, func, solver, method):
        self.func = func
        self.solver = solver
        self.method = method

    def predict(self, h0, t0_t):
        """

        :param h0: (N,1,d) initial value
        :param t0_t: (N,t) # the first col is time for initial values, remaining cols are time that needs to be evaluated
        :return: N,t,d: the predicted values at the the remaining times (initial value of first col across t time dimension is unchanged)
        """

        allsame = lambda x: torch.sum(x[1:] - x[:-1]) ==0
        repeat = lambda x: torch.cat([torch.unsqueeze(x, dim=0),torch.unsqueeze(x, dim=0)], dim=0)
        #if all the elements in t are same, then the odesolver is not evaluated
        h_pred = [self.solver(self.func, hi, ti, method=self.method) if not allsame(ti) else repeat(hi) for hi, ti in
                  zip(h0, t0_t)]
        # each element in h_pred has shape of (t,1,d)
        h_pred = torch.cat(h_pred, dim=1)  # (t,N,d)
        h_pred = h_pred.permute([1, 0, 2])  # (N,t,d)
        return h_pred



def unique_exp():
    x = torch.tensor([-10, 5, 6])
    y = torch.tensor([3, 5, 8])

    #x = torch.rand((4,3))
    #y = torch.rand((4,5))

    xy, idx = torch.unique(torch.cat([x, y]), return_inverse=True, sorted=True)
    idxs = torch.argsort((xy))
    xysorted  =  xy[idxs] #torch.gather(xy,None, idxs)
    idxsorted = idx[idxs] #torch.gather(idx,None, idxs) #


def unique(xy):
    xy, idx = torch.unique(xy, return_inverse=True, sorted=True) #sorts in ascending order e.g 0,2,4,5
    #idxs = torch.argsort((xy))
    #xysorted = xy[idxs]  # torch.gather(xy,None, idxs)
    #idxsorted = idx[idxs]  # torch.gather(idx,None, idxs) #
    #return  xysorted, idxsorted
    return xy, idx




class MultimodalLatentODESync(nn.Module):
    def __init__(self, latent_dim, device=None, ode_solver=None, expert='poe', split_latent=False):
        super(MultimodalLatentODESync, self).__init__()

        nhidden_size_rnfl = 64
        nhidden_size_vft = 32
        #self.hsyncfunc = LatentODEfunc(latent_dim=nhidden_size, nhidden=nhidden_size * 2)
        #self.h2musigma = nn.Linear(nhidden_size, latent_dim * 2)

        self.rnn_rnfl = RecognitionODEGRU(type='rnfl', latent_dim=latent_dim,
                                          device=device,
                                          ode_solver=ode_solver, z_size=nhidden_size_rnfl)


        self.rnn_vft = RecognitionODEGRU(type='vft', latent_dim=latent_dim,
                                         device=device,
                                         ode_solver=ode_solver, z_size=nhidden_size_vft)

        latent_dim_forecast = int(latent_dim / 2) if split_latent else latent_dim

        self.forecast_ode = LatentODEfunc(latent_dim=latent_dim_forecast, nhidden=128)

        self.decoder_rnfl = Decoder(z_size=latent_dim, channel_out=1, num_upsamples=4, image_size=32)
        self.decoder_gcl = None  # Decoder(z_size=latent_dim, channel_out=1, num_upsamples=4, image_size=64)
        self.decoder_vft = VFTDecoder(z_size=latent_dim)

        self.latent_dim = latent_dim
        self.ode_solver = ode_solver
        self.device = device
        self.split_latent = split_latent

        assert expert in ['moe', 'poe'], 'invalid option for experts'
        self.experts = MixtureOfExperts() if expert == "moe" else ProductOfExperts()

        self.metadata_input_dim = None
        self.use_cuda = 'cuda' in device.type
        self.generalized_poe = False

        ode_solver_, method = self.ode_solver

        method = 'euler'  # 'euler'#'rk4'
        self.node_sync_rnfl = NODE(self.rnn_rnfl.hfunc, ode_solver_, method)
        self.node_sync_vft = NODE(self.rnn_vft.hfunc, ode_solver_, method)

    def get_params_finetune(self):

        return list(self.rnn_rnfl.get_params_finetune()) + list(self.rnn_vft.get_params_finetune()) + list(
            self.forecast_ode.parameters())

    def reparametrize_(self, mu, logvar):
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

    def infer(self, ts_list, x_list):
        #x_rnfl, x_vft = x_list
        mu, logvar = self.get_posterior(ts_list, x_list)
        return mu, logvar

    # disable batch prediction for deubugging, since subset of output could be None, this needs to be taken into
    # account while batching
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

    def forward(self, ts_list, x_list):
        """

        :param x: (N,t,1,H,W)
        :param t: (N,t), if x_list[i] is none , ts_list[i] should be none
        :param mod_select array of Boolean to indicate which to use, When None all will be used
        :return:
        """

        assert len(ts_list) == 2, 'two inputs expected, rnfl and vft'
        assert len(x_list) == 2, 'two inputs expected, rnfl and vft'

        for i in range(len(x_list)):
            if(x_list[i] is None): assert ts_list[i] is None, 'if x is none, ts should be none for index ' +str(i)

        x_rnfl, x_vft = x_list
        batch_size, time_size_ = self.get_batch_time_size(x_rnfl, x_vft)

        ts = torch.cat([ts for ts in ts_list if ts is not None], dim=1)
        time_size = ts.shape[1]

        mu, logvar = self.get_posterior(ts_list, x_list)

        z0 = self.reparametrize(mu, logvar)

        z0 = torch.unsqueeze(z0, dim=1)
        if (self.split_latent):
            z0_static = z0[:, :, : int(self.latent_dim / 2)]
            z0_static = z0_static.repeat((1, time_size, 1))  # at all the time points same latent vector

            z0_dynamic = z0[:, :, int(self.latent_dim / 2):]
        else:
            z0_dynamic = z0

        ode_solver, method = self.ode_solver
        method = 'dopri5'  # 'euler'#'rk4'

        def solve(z0_,t_):
            #note unique will sort times in ascending order so the first element is the lowest time amng all modalities
            tu, tui = unique(t_)
            pred_ = ode_solver(self.forecast_ode, z0_, tu, method=method)
            return pred_, tui

        pz_idx = [solve(z0i, tsi) for z0i, tsi in zip(z0_dynamic, ts)]
        #each element in pz_idx is a tuple, whose first element is pz and second element is the index of the original concatenated array of times
        pz = [x[0][x[1]] for x in pz_idx]
        #idx = [x[1] for x in pz_idx]

        #pz = [ode_solver(self.forecast_ode, z0i, tsi, method=method) for z0i, tsi in zip(z0_dynamic, ts)]

        pred_z = torch.cat(pz, dim=1).permute(1, 0, 2)

        if (self.split_latent):
            pred_z = torch.cat([z0_static, pred_z], dim=2)


        flatten_time = lambda x: x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        # works when the list has two sequence and both are not None
        if (x_rnfl is not None and x_vft is not None):
            tsize_rnfl = ts_list[0].shape[1]
            tsize_vft = ts_list[1].shape[1]

            pred_z_rnfl = pred_z[:, :tsize_rnfl]
            pred_z_vft = pred_z[:, tsize_rnfl:]

        else:
            pred_z_rnfl = pred_z
            pred_z_vft = pred_z

            tsize_ = [ts.shape[1] for ts in ts_list if ts is not None][0]
            tsize_rnfl = tsize_
            tsize_vft = tsize_


        pred_z_rnfl = flatten_time(pred_z_rnfl)
        pred_z_vft = flatten_time(pred_z_vft)

        # Note* cross modality prediction in test phase only
        pred_rnfl = self.decoder_rnfl(pred_z_rnfl) if (x_rnfl is not None or not self.training) else None
        # pred_gcl = self.decoder_gcl(pred_z_) if x_gcl is not None else None
        pred_vft = self.decoder_vft(pred_z_vft) if (x_vft is not None or not self.training) else None

        pred_rnfl = self.match_size(pred_rnfl, batch_size, tsize_rnfl)
        # pred_gcl = self.match_size(pred_gcl, batch_size, time_size)
        pred_vft = self.match_size(pred_vft, batch_size, tsize_vft)

        # last time slice denotes the forecasted values. discard GCL and VFT for evaluation??

        return [pred_rnfl, pred_vft], [mu, logvar]

    def match_size(self, x, batch_size, time_size):

        if (x is not None):
            Bnt, c, H, W = x.shape

            assert Bnt == batch_size * time_size, 'Parameters not matching'
            # return x.view(batch_size, time_size, c, H, W)
            return x.reshape(batch_size, time_size, c, H, W)
        else:
            return x

    def get_posterior(self, ts_list, x_list):

        x_rnfl, x_vft = x_list
        ts_rnfl, ts_vft = ts_list
        batch_size, _ = self.get_batch_time_size(x_rnfl, x_vft)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert

        mu_all = []
        logvar_all = []

        if x_rnfl is not None:
            out, (h_rnfl, c) = self.rnn_rnfl(x_rnfl, ts_rnfl)
        if x_vft is not None:
            out, (h_vft, c) = self.rnn_vft(x_vft, ts_vft)

        if (x_rnfl is not None and x_vft is not None):  # sync

            t0_rnfl = ts_rnfl[:, 0]
            t0_vft = ts_vft[:, 0]
            ts_min = torch.min(t0_rnfl, t0_vft)

            #pred_t_rnfl = torch.cat([t0_rnfl, ts_min])
            pred_t_rnfl = torch.cat([torch.unsqueeze(t0_rnfl, dim=1), torch.unsqueeze(ts_min, dim=1)], dim=1)
            pred_t_vft = torch.cat([torch.unsqueeze(t0_vft, dim=1), torch.unsqueeze(ts_min, dim=1)], dim=1)
            #pred_t_vft = torch.cat([t0_vft, ts_min])

            h_rnfl = h_rnfl.unsqueeze(dim=1) #(N,1,d)
            h_rnfl_pred = self.node_sync_rnfl.predict(h_rnfl, pred_t_rnfl)
            h_rnfl = h_rnfl_pred[:,1,:]  #index 1 has the synchronized values

            h_vft = h_vft.unsqueeze(dim=1)  # (N,1,d)
            h_vft_pred = self.node_sync_vft.predict(h_vft, pred_t_vft)
            h_vft = h_vft_pred[:, 1, :]  # index 1 has the synchronized values



        if x_rnfl is not None:
            out = self.rnn_rnfl.rnn.h2o(h_rnfl)

            rnfl_mu, rnfl_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            mu_all.append(rnfl_mu)
            logvar_all.append(rnfl_logvar)

        if x_vft is not None:
            out = self.rnn_vft.rnn.h2o(h_vft)
            vft_mu, vft_logvar = out[:, :self.latent_dim], out[:, self.latent_dim:]
            mu_all.append(vft_mu)
            logvar_all.append(vft_logvar)

        # product of experts to combine gaussians

        # if (len(mu_all) > 1):  # prior expert is not required for single modality forward pass
        mu_prior, logvar_prior = prior_expert((batch_size, self.latent_dim),
                                              use_cuda=use_cuda)
        mu_all.insert(0, mu_prior)
        logvar_all.insert(0, logvar_prior)

        mu = torch.cat([m.unsqueeze(0) for m in mu_all], dim=0)
        logvar = torch.cat([lv.unsqueeze(0) for lv in logvar_all], dim=0)

        mu, logvar = self.experts(mu, logvar)

        return mu, logvar

    def get_batch_time_size(self, *args):
        for arg in args:
            if arg is not None:
                return arg.size(0), arg.size(1)
