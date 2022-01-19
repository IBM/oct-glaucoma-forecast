import torch
import torch.nn as nn
import torch.nn.functional as F
from models.multiodal_latentodegru_sync import  RecognitionGRU
from models.model_blocks import Decoder, VFTDecoder, Swish


class LatentODEfunc(nn.Module):

    def __init__(self, latent_dim=4, nhidden=20):
        super(LatentODEfunc, self).__init__()
        self.elu = nn.ELU(inplace=True)
        self.fc1 = nn.Linear(latent_dim + 1, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, latent_dim)
        #self.fc4 = nn.Linear(nhidden, latent_dim)
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
        #out = self.swish(out)
        #out = self.fc4(out)

        return out



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
        for t in (range(x.size(1))):
            obs = x[:, t, :]

            if (t >0  and self.use_odernn):  # the first cell, there is not hidden dynamics
                ode_solver, method = self.ode_solver
                h_next = [ode_solver(self.hfunc, torch.unsqueeze(hi, dim=0), ti[torch.arange(t - 1, t +1, +1)],
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



class ODEGRU(nn.Module):
    def __init__(self, type, device=None, ode_solver=None):
        super(ODEGRU, self).__init__()

        assert type in ['rnfl', 'vft'], 'type should be one of rnfl or vft'
        self.type= type
        latent_dim=2 # we dont need latent code but api needs it so we can pass an int
        nhidden_size = 64 if type=='rnfl' else 32
        self.ode_gru = RecognitionODEGRU(type=type, latent_dim=latent_dim,
                                          device=device,
                                          ode_solver=ode_solver, z_size=nhidden_size)

        if(type=='rnfl'):
            self.decoder = Decoder(z_size=nhidden_size, channel_out=1, num_upsamples=4, image_size=32)
        else:
            self.decoder = VFTDecoder(z_size=nhidden_size)


        self.ode_solver = ode_solver
        self.device = device

        self.use_cuda = 'cuda' in device.type

    def forward(self, ts, x, batch_size=None):
        """

        :param ts:
        :param x_list:
        :param batch_size: when None, then batching is not performed
        :return:
        """

        if (batch_size is None):
            return self.forward_(ts, x)

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


        for chunk in batch(idx, batch_size):
            ts_i = ts[chunk]
            input_i = [inp[chunk] if inp is not None else None for inp in x]
            pred_i = self.forward_(ts_i, input_i)
            out_pred.append(pred_i)

        return collect(out_pred)

    def forward_(self, ts, x):
        """

        :param x_list : list of (N,t,1,H,W)
        :param t_list:  list of (N,t)
        :param mod_select array of Boolean to indicate which to use, When None all will be used
        :return:
        """


        out, (h, c) = self.ode_gru.forward(x, ts)



        pred_rnfl = self.decoder(h) #(N,c,H,W)
        pred_rnfl = pred_rnfl.unsqueeze(dim=1) #add time dimension (N,t,c,H,W)

        outlist =[pred_rnfl]



        return outlist

