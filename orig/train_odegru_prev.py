import os

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataflow import SelectComponent

from data import ODEMAPDataLoader
from losses import mae_globalmean
from models.ode_gru import ODEGRU

# from ode.viz_latent import latent_viz, latent_viz_random

adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint

GPU = 0
device = torch.device('cuda:' + str(GPU)

                      if torch.cuda.is_available() else 'cpu')

print('Device:', device.type)

BATCH_SIZE = 32

image_dim = 64

number_pixel_out = image_dim * image_dim * 1


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


class BatchPredictor():

    def __init__(self, mode):
        super(BatchPredictor, self).__init__()

        self.model = mode

    def predict(self, ds):
        """

        :param ds: instance of  BatchDataPT
        :return:
        """
        out = []
        latent = []
        for obs0, obs1, obs2, obs3, ts, mdata in ds.get_data():
            # last one is the value to be forecsted
            x0 = obs0[:, :-1]
            x1 = obs1[:, :-1]
            x2 = obs2[:, :-1]
            x3 = obs3[:, :-1]  # proj

            input = [x0, x1, x2]
            y_pred, mulogvar = self.model.forward(ts, input)
            out.append(y_pred)
            latent.append(mulogvar)
        final_ypred = collect(out)
        final_latent = collect(latent)

        return final_ypred, final_latent


rnfldata = ODEMAPDataLoader(image_dim=image_dim, device=device)


# ds_val_batch= rnfldata.get_data_rnfl_map_val(BATCH_SIZE=16)

# ds_val  = RNFLData(obsmaps_val, age_at_vd_val,dx_val)


def log_normal_pdf(x, mean, logvar):
    const = torch.from_numpy(np.array([2. * np.pi])).float().to(x.device)
    const = torch.log(const)
    return -.5 * (const + logvar + (x - mean) ** 2. / torch.exp(logvar))


def normal_kl(mu1, lv1, mu2, lv2):
    v1 = torch.exp(lv1)
    v2 = torch.exp(lv2)
    lstd1 = lv1 / 2.
    lstd2 = lv2 / 2.

    kl = lstd2 - lstd1 + ((v1 + (mu1 - mu2) ** 2.) / (2. * v2)) - .5
    return kl


def squared_error(x, y):
    """

    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d
    return z.sum(dim=1)


def mean_squared_error(x, y):
    """

    :param x: (B,d)
    :param y:  (B,d)
    :return: (B,)
    """
    d = x - y
    z = d * d
    return z.mean(dim=1)


bce_loss = nn.BCEWithLogitsLoss()  # nn.BCELoss()


class RunningAverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, momentum=0.99):
        self.momentum = momentum
        self.reset()
        self.loss_history = []

    def reset(self):
        self.val = None
        self.avg = 0
        self.loss_history = []

    def update(self, val):
        if self.val is None:
            self.avg = val
        else:
            self.avg = self.avg * self.momentum + val * (1 - self.momentum)
        self.val = val
        self.loss_history.append(val)

    def save_csv(self, file='loss_history.csv'):
        np.savetxt(file, self.loss_history, delimiter=',')


def sigmoid(x):
    z = 1 / (1 + np.exp(-x))
    return z



def mse(pred, gt):
    """

    :param preds: list each element ais an  ndarry (N, n_t, c, H, W)
    :param gts: list  each element ais an  ndarry (N, n_t, c, H, W)
    :return:
    """

    forecast_loss = squared_error(gt.view(gt.shape[0], -1),
                                  pred.view(pred.shape[0], -1))
    MSE = torch.mean(forecast_loss)
    return MSE











def save_model(model, dir, file_path):
    # save the models
    if (not os.path.exists(dir)):
        os.mkdir(dir)
    torch.save(model.state_dict(), os.path.join(dir, file_path))


class MLODECOnfig:
    ROOT_DIR = 'out'
    PREFIX = 'mlode_v2'
    NEPOCHS = 80


class Config:
    latent_dim = 64
    image_dim = 64
    vfim_dim = 32
    use_odernn = True
    ode_method ='fixed_adams'
    rnn_type = 'gru' #always fixed

    prefix ='odegru' if use_odernn else 'gru'



    def create_model(self):
        config=self
        ode_solver  =  [odeint, config.ode_method]  if config.use_odernn else None
        print('ODE solver', str(ode_solver))
        latentode = ODEGRU(image_dim=config.image_dim, vfim_dim=config.vfim_dim,
                                        latent_dim=config.latent_dim,
                                        ode_solver=ode_solver,
                                        device=device).to(device)
        return latentode


def train(config: Config):

    latentode = config.create_model()

    prefix = config.prefix

    result_dir = 'results' + prefix
    if (not os.path.exists(result_dir)):
        os.mkdir(result_dir)

    params = (list(latentode.parameters()))
    optimizer = optim.Adam(params, lr=0.001)
    loss_meter = RunningAverageMeter()
    kl_loss_meter = RunningAverageMeter()
    dx_loss_meter = RunningAverageMeter()

    nepochs = 80
    niter_per_epoch = int(rnfldata.age_at_vd_train.shape[0] / BATCH_SIZE)

    print('Training data size', rnfldata.age_at_vd_train.shape[0])
    print('Val data size', rnfldata.age_at_vd_val.shape[0])
    print('Prefix', prefix)
    print('#Total iterations ', nepochs * niter_per_epoch)

    kl_weight = 0
    min_val_loss = np.inf
    val_loss_prev = np.inf
    valloss_increase_count = 0
    for epoch in range(1, nepochs + 1):
        start = time.time()
        for itr in range(1, niter_per_epoch + 1):
            rnfl_trajs, gcl_trajs, vft_traj, proj_traj, samp_ts, dx, mdata = rnfldata.get_data_rnfl_map_train(
                batch_size=BATCH_SIZE)
            # samp_ts = samp_ts - samp_ts[:,[0]]
            optimizer.zero_grad()

            x_list = [rnfl_trajs[:, :-1], None, None]
            out = latentode.forward(ts=samp_ts, x_list=x_list)
            rnfl_pred =out[0]
            train_elbo_loss = mse(rnfl_pred, rnfl_trajs[:,-1])

            train_elbo_loss.backward()
            optimizer.step()
            loss_meter.update(train_elbo_loss.item())
        end = time.time()

        # decay learning rate every 10 epoch
        # if(epoch % 10==0):
        #    for g in optimizer.param_groups:
        #        g['lr'] = g['lr'] * 0.45
        #

        with torch.no_grad():

            latent_ode_eval = latentode.eval()

            if (epoch % 5 == 0 and epoch <= nepochs / 2):
                save_path = 'latentode' + prefix + '.pt'
                print('Epoch', epoch, ' lr:', optimizer.param_groups[0]['lr'], 'kl_weight:', kl_weight,
                      'saving model to ', save_path)
                save_model(latentode, dir='trained_models', file_path=save_path)

            # validation

            ds_val_batch = rnfldata.get_data_rnfl_map_val(BATCH_SIZE=-1)
            ds_pred_batch = SelectComponent(ds_val_batch, [0, 1, 2, 3, 4])  # rnfl and diagnosis
            rnfl_val, gcl_val, vft_val, proj_val, ts_val = next(ds_pred_batch.get_data())


            x_list_val = [rnfl_val[:, :-1], None, None]
            out_val = latent_ode_eval.forward(ts=ts_val, x_list=x_list_val, batch_size=32)
            rnfl_pred_val = out_val[0]
            elbo_loss_val = mse(rnfl_pred_val, rnfl_val[:,-1])





            gm_forecast_mae = torch.mean(mae_globalmean(rnfl_val[:, -1] * 200, rnfl_pred_val[:, -1] * 200))

            disp = [['Epoch', epoch], ['trainloss', loss_meter.avg], ['kl_loss', kl_loss_meter.avg],
                    ['valloss', elbo_loss_val.item()],  ['gm_mae_pred', gm_forecast_mae],
                    ['time/ep', end - start]]

            print(''.join([str(d[0]) + ': ' + '{:.2f}'.format(d[1]) + ', ' for d in disp]).rstrip(', '), flush=True)

            if (elbo_loss_val < min_val_loss):
                min_val_loss = elbo_loss_val
                if (epoch > nepochs / 2):
                    save_path = 'latentode' + prefix + '.pt'
                    save_model(latentode, dir='trained_models', file_path=save_path)
                    print('Model saved to ', save_path)
                valloss_increase_count = 0

            if (elbo_loss_val > val_loss_prev):
                valloss_increase_count += 1
            else:
                valloss_increase_count = 0

            if (valloss_increase_count >= 2):
                for g in optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.1
                print('Validation loss did not improve in 2 consecutive epochs: learning rate changed to',
                      str(optimizer.param_groups[0]['lr']))

            val_loss_prev = elbo_loss_val

    # save the losses
    if(not os.path.exists('loss_history')): os.mkdir('loss_history')
    loss_meter.save_csv('losshistory/loss_history' + prefix + '.csv')


def test(config):
    #latentode = MultimodalLatentODE(image_dim=config.image_dim, vfim_dim=config.vfim_dim,
    #                                latent_dim=config.latent_dim, ode_solver=odeint,
    #                                device=device).to(device)

    latentode = config.create_model()

    model_file = os.path.join('trained_models', 'latentode' + config.prefix + '.pt')
    latentode.load_state_dict(torch.load(model_file, map_location='cpu'))
    latent_ode_eval = latentode.eval()

    ds_val_batch = rnfldata.get_data_rnfl_map_val(BATCH_SIZE=-1)
    ds_pred_batch = SelectComponent(ds_val_batch, [0, 1, 2, 3, 4])  # rnfl and diagnosis
    rnfl_val, gcl_val, vft_val, proj_val, ts_val = next(ds_pred_batch.get_data())

    #comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #for rnfl vft
    comb = [[1, 0, 1], [1, 0, 0], [0, 0, 1] ]

    input_names = ['rnfl', 'gcl', 'vft']

    def compute_mean_std(x, y):
        temp = mae_globalmean(x, y)
        return torch.mean(temp), torch.std(temp)

    for nv in [3]:
        start=  3 - nv
        print('#NV ', nv)
        for c in comb:
            input_xi = [rnfl_val[:, start:-1], gcl_val[:, start:-1], vft_val[:, start:-1]]
            input_xi_sub = subsample(input_xi, c)
            ts_val_ = ts_val[:, start:]
            [pred_rnfl, pred_gcl, pred_vft], mulogvar = latent_ode_eval.forward(ts=ts_val_,
                                                                                x_list=input_xi_sub,
                                                                                batch_size=32)

            forecast_mae, foecast_std = compute_mean_std(rnfl_val[:,  -1] * 200, pred_rnfl[:, -1] * 200)
            rec_mae, rec_std = compute_mean_std(rnfl_val[:, start:-1] * 200, pred_rnfl[:,  start:-1] * 200)

            print(subsample(input_names, c), 'forecast mae {:.2f}+-{:.2f}'.format(forecast_mae, foecast_std),
                  ' recon mae{:.2f}+-{:.2f}'.format(rec_mae, rec_std), flush=True)


if (__name__ == '__main__'):
    config = Config()
    train(config)
    #test(config)
