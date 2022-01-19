import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resnet import create_resnet50



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
        T=T*w

        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1 / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar



def prior_expert(size, use_cuda=False):
    """Universal prior expert. Here we use a spherical
    Gaussian: N(0, 1).

    @param size: integer
                 dimensionality of Gaussian
    @param use_cuda: boolean [default: False]
                     cast CUDA on variables
    """
    mu     = torch.autograd.Variable(torch.zeros(size))
    logvar = torch.autograd.Variable(torch.log(torch.ones(size)))

    if use_cuda:
        mu, logvar = mu.cuda(), logvar.cuda()
    return mu, logvar



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
            ten = F.relu(ten, False)
            return ten, ten_out
        else:
            ten = self.conv(ten)
            ten = self.bn(ten)
            ten = F.relu(ten, True)
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
    def __init__(self, input_shape, channel_in=3, z_size=128, num_downsamples=3, dropout_rate=None):
        super(Encoder, self).__init__()
        self.size = channel_in
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
            nn.Linear(in_features=final_shape[0] * final_shape[1] * self.size, out_features=256, bias=True),
            nn.BatchNorm1d(num_features=256, momentum=0.9),
            nn.ReLU(True))
        # two linear to get the mu vector and the diagonal of the log_variance
        self.l_mu = nn.Linear(in_features=256, out_features=z_size)

    def forward(self, ten):
        ten = self.conv(ten)
        ten = ten.view(len(ten), -1)
        ten = self.fc(ten)
        mu = self.l_mu(ten)
        # logvar = self.l_var(ten)
        return mu  # , logvar

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


class EncoderVAE(nn.Module):
    def __init__(self, input_shape, channel_in=3, z_size=128, num_downsamples=3, latent_dim=2):
        super(EncoderVAE, self).__init__()
        self.encoder = Encoder(input_shape, channel_in, z_size, num_downsamples, dropout_rate=None)
        self.h1o = nn.Linear(z_size, latent_dim * 2)

    def forward(self, x):
        x = self.encoder(x)
        x = F.relu(x)
        mu_logvar = self.h1o(x)
        return mu_logvar

    def __call__(self, *args, **kwargs):
        return super(EncoderVAE, self).__call__(*args, **kwargs)


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

    def __init__(self, input_shape, channel_in=3, z_size=128, num_downsamples=3,
                 latent_dim=4, nhidden=64):
        super(RecognitionGRU, self).__init__()

        self.encoder = Encoder(input_shape, channel_in, z_size, num_downsamples)

        self.grucell = nn.GRUCell(z_size, nhidden)  # nn.LSTMCell(z_size, nhidden)
        self.nhidden = nhidden

        self.h2o = nn.Linear(nhidden, latent_dim * 2 +1)

    def load_inner_encoder(self, encoder_inner_weight_path):
        self.encoder.load_state_dict(torch.load(encoder_inner_weight_path))

    def forward(self, x, hc, ):
        h, c = hc
        x1 = self.encoder(x)
        h = self.grucell(x1, h)
        out = self.h2o(h)
        return out, (h, c)

    def initHidden(self, nbatch):
        return (torch.zeros(nbatch, self.nhidden), torch.zeros(nbatch, self.nhidden))




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
        ten = F.relu(ten, True)
        return ten


class Decoder(nn.Module):
    def __init__(self, z_size, size, image_size, channel_out=3, num_upsamples=3):
        super(Decoder, self).__init__()
        # start from B*z_size
        self.start_fm_size = int(image_size // (2 ** num_upsamples))
        assert image_size == 2 ** num_upsamples * self.start_fm_size, 'Image size and num_upsamples are not consistent'

        num_out_features = self.start_fm_size * self.start_fm_size * size
        self.fc = nn.Sequential(nn.Linear(in_features=z_size,
                                          out_features=num_out_features,
                                          bias=True), nn.BatchNorm1d(num_features=num_out_features,
                                                                     momentum=0.9), nn.ReLU(True))
        self.size = size
        layers_list = []
        for i in range(1, num_upsamples + 1):
            layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // i))
            self.size = self.size // i

        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size))
        # self.size = self.size

        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 2))
        # self.size = self.size // 2

        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 4))
        # self.size = self.size // 4

        # layers_list.append(DecoderBlock(channel_in=self.size, channel_out=self.size // 8))
        # self.size = self.size // 8

        # final conv to get 3 channels and tanh layer
        layers_list.append(nn.Sequential(
            nn.Conv2d(in_channels=self.size, out_channels=channel_out, kernel_size=5, stride=1, padding=2)
            , nn.Sigmoid()
        ))

        self.conv = nn.Sequential(*layers_list)

    def forward(self, ten):
        ten = self.fc(ten)
        ten = ten.view(len(ten), -1, self.start_fm_size, self.start_fm_size)
        ten = self.conv(ten)
        return ten

    def __call__(self, *args, **kwargs):
        return super(Decoder, self).__call__(*args, **kwargs)


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





class RecognitionODEGRU(nn.Module):

    def __init__(self, image_dim, latent_dim, device=None, ode_solver=None):
        super(RecognitionODEGRU, self).__init__()

        self.use_odernn = ode_solver is not None
        self.ode_solver = ode_solver
        nhidden1 = latent_dim * 2

        self.rnn = RecognitionGRU(input_shape=(image_dim, image_dim), channel_in=1, num_downsamples=4, z_size=128,
                                  latent_dim=latent_dim, nhidden=nhidden1)
        self.hfunc = LatentODEfunc(latent_dim=self.rnn.nhidden, nhidden=self.rnn.nhidden * 2)

        self.device = device

    def initialize_temproary_vars(self, batch_size ):

        h, c = self.rnn.initHidden(nbatch=batch_size)
        return  h.to(self.device), c.to(self.device)

    def forward(self, x, ts):


        h,c = self.initialize_temproary_vars(batch_size=x.shape[0])
        for t in reversed(range(x.size(1))):
            obs = x[:, t, :]

            if (t < x.size(1) - 1 and self.use_odernn):  # the first cell, there is not hidden dynamics
                ode_solver, method = self.ode_solver
                h_next = [ode_solver(self.hfunc, torch.unsqueeze(hi, dim=0), ti[torch.arange(t + 1, t - 1, -1)],
                                          method=method) for hi, ti in zip(h, ts)]
                h = torch.cat(h_next, dim=1)[1]  # index 1 contains the forecasted value


            out, (h, c) = self.rnn.forward(obs, (h, c))
        return out, (h, c)


class DecoderODEGRU(nn.Module):

    def __init__(self, z_size, size, image_size, channel_out=3, num_upsamples=3, nhidden=64,  device=None, ode_solver=None,metadata_input_dim=None):
        super(DecoderODEGRU, self).__init__()

        self.use_odernn = ode_solver is not None
        self.ode_solver = ode_solver

        self.rnn =  DecoderGRU(z_size, size, image_size, channel_out=channel_out, num_upsamples= num_upsamples, nhidden=nhidden)

        self.hfunc = LatentODEfunc(latent_dim=self.rnn.nhidden, nhidden=self.rnn.nhidden * 2)

        self.device = device

    def initialize_temproary_vars(self, batch_size):

        h, c = self.rnn.initHidden(nbatch=batch_size)
        return  h.to(self.device), c.to(self.device)

    def forward(self, pred_z, ts):

        pred_x = []
        # Note:  reversed -> torch.arange(t+1, t - 1, -1)  & t < pred_z.size(1) -1
        # Note : no reversed -> torch.arange(t-1, t + 1, +1) & t >0

        h,c = self.initialize_temproary_vars(batch_size=pred_z.shape[0])
        for t in (range(pred_z.size(1))):
            #mdatai = None  # mdata[:, t, :] if self.metadata_input_dim is not None else None
            if (t > 0 and self.use_odernn):  # in the first cell, there is no hidden dynamics i.e  h1'= h0
                h_next = [
                    self.ode_solver(self.hfunc, torch.unsqueeze(hi, dim=0), ti[torch.arange(t - 1, t + 1, +1)],
                                    method='rk4') for hi, ti in zip(h, ts)]
                h = torch.cat(h_next, dim=1)[1]  # index 1 contains the forecasted value

            out, (h, c) = self.rnn.forward(pred_z[:, t, :], (h, c))
            pred_x.append(out)

        # pred_x = list(reversed(pred_x))
        pred_x = torch.stack(pred_x, dim=1)
        return pred_x,  (h,c)




#https://github.com/mhw32/multimodal-vae-public/blob/master/vision/model.py
# @todo handle sample in test as done in mvae

class MultimodalLatentODE(nn.Module):
    def __init__(self, image_dim, vfim_dim, latent_dim, device=None, ode_solver=None, generalized_poe =False):
        super(MultimodalLatentODE, self).__init__()
        self.rnn_rnfl = RecognitionODEGRU(image_dim=image_dim, latent_dim=latent_dim, device=device,
                                          ode_solver=ode_solver)

        self.rnn_gcl = RecognitionODEGRU(image_dim=image_dim, latent_dim=latent_dim, device=device,
                                         ode_solver=ode_solver)

        self.rnn_vft = RecognitionODEGRU(image_dim=vfim_dim, latent_dim=latent_dim, device=device,
                                         ode_solver=ode_solver)


        self.forecast_ode = LatentODEfunc(latent_dim=latent_dim)

        self.decoder_rnfl =Decoder(z_size=latent_dim, size=128, channel_out=1, num_upsamples=4, image_size=image_dim)
        self.decoder_gcl = Decoder(z_size=latent_dim, size=128, channel_out=1, num_upsamples=4, image_size=image_dim)
        self.decoder_vft = Decoder(z_size=latent_dim, size=64, channel_out=1, num_upsamples=3, image_size=vfim_dim)




        self.latent_dim = latent_dim
        self.ode_solver = ode_solver
        self.device = device

        self.experts = ProductOfExperts()

        self.metadata_input_dim = None
        self.use_cuda = 'cuda' in device.type
        self.generalized_poe=generalized_poe

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:  # return mean during inference
            return mu



    def forward(self, ts, x_list, batch_size = None):
        """

        :param ts:
        :param x_list:
        :param batch_size: when None, then batching is not performed
        :return:
        """

        if(batch_size is None):
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



    def forward_(self, ts, x_list):
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
        method = 'euler'
        pz = [ode_solver(self.forecast_ode, z0i, ts, method=method) for z0i, ts in zip(z0, ts)]

        pred_z = torch.cat(pz, dim=1).permute(1, 0, 2)


        na, nb, nc = pred_z.shape
        pred_z_ = pred_z.reshape(na * nb, nc)
        pred_rnfl = self.decoder_rnfl(pred_z_)
        pred_gcl = self.decoder_gcl(pred_z_)
        pred_vft = self.decoder_vft(pred_z_)

        batch_size,time_size_ = self.get_batch_time_size(x_rnfl, x_gcl, x_vft)

        time_size = ts.shape[1]
        pred_rnfl = self.match_size(pred_rnfl,batch_size, time_size) # the output time ponts
        pred_gcl = self.match_size(pred_gcl, batch_size,time_size)
        pred_vft = self.match_size(pred_vft, batch_size,time_size)



       #last time slice denotes the forecasted values. discard GCL and VFT for evaluation??

        return [pred_rnfl, pred_gcl, pred_vft], [mu, logvar]

    def match_size(self, x, batch_size,time_size):
        Bnt,c,H,W = x.shape

        assert Bnt == batch_size*time_size, 'Parameters not matching'
        return  x.view(batch_size,time_size, c,H,W)

    def get_posterior(self,ts, x_rnfl = None, x_gcl=None, x_vft=None, x_proj=None):

        # define universal expert
        batch_size,_ = self.get_batch_time_size(x_rnfl, x_gcl, x_vft, x_proj)
        use_cuda = next(self.parameters()).is_cuda  # check if CUDA
        # initialize the universal prior expert


        mu_all=[]
        logvar_all=[]
        w_all=[]

        if x_rnfl is not None:
            out, (h, c) = self.rnn_rnfl.forward(x_rnfl, ts)
            rnfl_mu, rnfl_logvar, w1 = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim], out[:,
                                                                                                                2 * self.latent_dim:]
            mu_all.append(rnfl_mu)
            logvar_all.append(rnfl_logvar)
            w_all.append(w1)





        if x_gcl is not None:
            out, (h, c) = self.rnn_gcl.forward(x_gcl, ts)
            gcl_mu, gcl_logvar, w2 = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim], out[:,
                                                                                                  2 * self.latent_dim:]
            mu_all.append(gcl_mu)
            logvar_all.append(gcl_logvar)
            w_all.append(w2)




        if x_vft is not None:
            out, (h, c) = self.rnn_vft.forward(x_vft, ts)
            vft_mu, vft_logvar, w3 = out[:, :self.latent_dim], out[:, self.latent_dim:2 * self.latent_dim], out[:,
                                                                                                                2 * self.latent_dim:]
            mu_all.append(vft_mu)
            logvar_all.append(vft_logvar)
            w_all.append(w3)


        # product of experts to combine gaussians


        if(len(mu_all) >1): #prior expert is not required for single modality forward pass
            mu_prior, logvar_prior = prior_expert((batch_size, self.latent_dim),
                                                  use_cuda=use_cuda)
            w_prior = torch.autograd.Variable(torch.ones((batch_size, 1)))
            w_prior = w_prior.cuda() if self.use_cuda else w_prior
            mu_all.append(mu_prior)
            logvar_all.append(logvar_prior)
            w_all.append(w_prior)

        mu = torch.cat([m.unsqueeze(0) for m in mu_all], dim=0)
        logvar = torch.cat([lv.unsqueeze(0) for lv in logvar_all], dim=0)
        w = torch.cat([wi.unsqueeze(0) for wi in w_all], dim=0)

        if (not self.generalized_poe): w = None

        mu, logvar = self.experts(mu, logvar, w)


        return mu, logvar


    def get_batch_time_size(self, *args):
        for arg in args:
            if arg is not None:
                return arg.size(0), arg.size(1)

