import os

import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from dataflow import SelectComponent

from data import ODEMAPDataLoader
from losses import mae_globalmean
from models.multiodal_latentodegru import MultimodalLatentODE

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

latent_dim = 64
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


def compute_elbo_loss(pred_rnfl, rnfl, pred_gcl, gcl, pred_vft, vft, mu, logvar, annealing_factor=1.):
    BCE = 0
    n_modalities = 0
    if pred_rnfl is not None and rnfl is not None:
        reconstruction_loss = squared_error(rnfl[:, :-1, :].view(rnfl.shape[0], -1),
                                            pred_rnfl[:, :-1, :].view(pred_rnfl.shape[0], -1))
        forecast_loss = squared_error(rnfl[:, -1, :].view(rnfl.shape[0], -1),
                                      pred_rnfl[:, -1, :].view(pred_rnfl.shape[0], -1))

        BCE += (reconstruction_loss + forecast_loss) / 2.0
        n_modalities += 1

    if pred_gcl is not None and gcl is not None:
        reconstruction_loss = squared_error(gcl[:, :-1, :].view(gcl.shape[0], -1),
                                            pred_gcl[:, :-1:].view(pred_gcl.shape[0], -1))
        BCE += reconstruction_loss
        n_modalities += 1

    if pred_vft is not None and vft is not None:
        reconstruction_loss = squared_error(vft[:, :-1, :].view(vft.shape[0], -1),
                                            pred_vft[:, :-1:].view(pred_vft.shape[0], -1))
        BCE += reconstruction_loss
        n_modalities += 1

    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    ELBO = torch.mean(BCE / float(n_modalities) + annealing_factor * KLD)
    return ELBO, torch.mean(KLD)


def elbo_general(preds, gts, mu, logvar, annealing_factor=1.):
    """

    :param preds: list each element ais an  ndarry (N, n_t, c, H, W)
    :param gts: list  each element ais an  ndarry (N, n_t, c, H, W)
    :param mu:
    :param logvar:
    :param annealing_factor:
    :return:
    """

    n_modalities = len(preds)
    assert len(preds) == len(gts), 'Prediction list and ground truth list length does not match'

    n_t = preds[0].shape[1]

    MSE = 0
    for pred, gt in zip(preds, gts):

        assert all([pred is not None, gt is not None]) or all(
            [pred is None, gt is None]), ' either both gt and pred should be None or both of them shoud be provided'
        if (pred is not None and gt is not None):
            reconstruction_loss = squared_error(gt[:, :-1, :].view(gt.shape[0], -1),
                                                pred[:, :-1, :].view(pred.shape[0], -1))

            forecast_loss = squared_error(gt[:, -1, :].view(gt.shape[0], -1),
                                          pred[:, -1, :].view(pred.shape[0], -1))

            MSE += (reconstruction_loss + forecast_loss) / 2.0

    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    # NOTE: we use lambda_i = 1 for all i since each modality is roughly equal
    ELBO = torch.mean(MSE / float(n_modalities) + annealing_factor * KLD)
    return ELBO, torch.mean(KLD)


def subsample(xx, cc):
    f = lambda x, c: x if c == 1 else None
    return [f(xi, ci) for xi, ci in zip(xx, cc)]


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def batch_predict(model, ts, input, batch_size):
    idx = list(range(ts.shape[0]))
    out_pred = []
    out_latent = []

    for chunk in batch(idx, batch_size):
        ts_i = ts[chunk]
        input_i = [inp[chunk] if inp else None for inp in input]
        pred_i, latent_i = model.forward(ts_i, input_i)
        out_pred.append(pred_i)
        out_latent.append(latent_i)

    return collect(out_pred), collect(out_latent)


def subsampled_losses(model, t_ts, rnfl_ts, gcl_ts, vft_ts, annealing_factor=1., batch_size=None):
    inputs_ts = [rnfl_ts, gcl_ts, vft_ts]
    input_xi = [x[:, :-1] for x in inputs_ts]

    comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]

    loss = 0
    KLD = 0
    for c in comb:
        input_xi_sub = subsample(input_xi, c)
        pred_list, [mu, logvar] = model.forward(ts=t_ts, x_list=input_xi_sub, batch_size=batch_size)
        # if the loss of gcl or vft not reqired then  set the correspndng index in pred_list and input_ts to None before passing below
        loss_, KLD_ = elbo_general(preds=pred_list, gts=inputs_ts, mu=mu, logvar=logvar,
                                   annealing_factor=annealing_factor)

        loss += loss_
        KLD += KLD_

    return loss, KLD


def rnfl_loss(model, t_ts, rnfl_ts, gcl_ts=None, vft_ts=None, annealing_factor=1., batch_size=None):
    inputs_ts = [rnfl_ts, gcl_ts, vft_ts]
    input_xi = [x[:, :-1] if x is not None else None for x in inputs_ts]

    comb = [[1, 0, 0]]

    loss = 0
    KLD = 0
    for c in comb:
        input_xi_sub = subsample(input_xi, c)
        pred_list, [mu, logvar] = model.forward(ts=t_ts, x_list=input_xi_sub, batch_size=batch_size)
        # if the loss of gcl or vft not reqired then  set the correspndng index in pred_list and input_ts to None before passing below
        pred_list[1] = None;
        pred_list[2] = None
        inputs_ts[1] = None
        inputs_ts[2] = None
        loss_, KLD_ = elbo_general(preds=pred_list, gts=inputs_ts, mu=mu, logvar=logvar,
                                   annealing_factor=annealing_factor)

        loss += loss_
        KLD += KLD_

    return loss, KLD


def save_model(model, dir, file_path):
    # save the models
    if (not os.path.exists(dir)):
        os.mkdir(dir)
    torch.save(model.state_dict(), os.path.join(dir, file_path))


class MLODECOnfig:
    ROOT_DIR = 'out'
    PREFIX = 'mlode_v2'
    NEPOCHS = 80


if (__name__ == '__main__'):

    use_odernn = True
    suffix = 'hode_pred' if use_odernn else 'pred'
    ode_method = 'rk4'
    ode_solver = [odeint, ode_method] if use_odernn else None

    print('ODE solver ', str(ode_solver))
    for rnn_type in ['gru']:  ##['vanila', 'lstm', 'gru']:

        mdata_input_dim = rnfldata.metadata_train.shape[3]
        latentode = MultimodalLatentODE(image_dim=image_dim, vfim_dim=32, latent_dim=latent_dim, ode_solver=ode_solver,
                                        device=device).to(device)

        # V1 - joint + individual elbo
        # v2 - joint + individual elbo but the loss is constructed for nnfl only i.e the decoder for other modality are not used
        # No v1 - only joint elbo
        prefix = 'rnfl_ldim_' + str(latent_dim) + suffix + rnn_type + 'elbo'

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

                rnfl_xi = rnfl_trajs[:, :-1]
                yi = rnfl_trajs[:, -1]

                gcl_xi = gcl_trajs[:, :-1]
                vft_xi = vft_traj[:, :-1]

                sample_dist = np.random.binomial(1, 0.5, 3)
                while (np.sum(sample_dist) < 2): sample_dist = np.random.binomial(1, 0.5, 3)

                kl_weight = np.max([(epoch - 7) / 10.0, 0])
                kl_weight = np.min([kl_weight, 1])

                train_elbo_loss, kld = rnfl_loss(latentode, samp_ts, rnfl_trajs, annealing_factor=kl_weight)

                train_elbo_loss.backward()
                optimizer.step()
                loss_meter.update(train_elbo_loss.item())
                kl_loss_meter.update(torch.mean(kld).item())
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

                elbo_loss_val, kld_val = rnfl_loss(model=latent_ode_eval, t_ts=ts_val, rnfl_ts=rnfl_val, batch_size=32)

                [pred_rnfl, pred_gcl, pred_vft], mulogvar = latent_ode_eval.forward(ts=ts_val,
                                                                                    x_list=[rnfl_val[:, :-1],
                                                                                            None,
                                                                                            None],
                                                                                    batch_size=32)

                gm_forecast_mae = torch.mean(mae_globalmean(rnfl_val[:, -1] * 200, pred_rnfl[:, -1] * 200))
                gm_rec_mae = torch.mean(mae_globalmean(rnfl_val[:, :-1] * 200, pred_rnfl[:, :-1] * 200))

                disp = [['Epoch', epoch], ['trainloss', loss_meter.avg], ['kl_loss', kl_loss_meter.avg],
                        ['valloss', elbo_loss_val.item()], ['gm_mae_rec', gm_rec_mae], ['gm_mae_pred', gm_forecast_mae],
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
        if (not os.path.exists('loss_history')): os.mkdir('loss_history')
        loss_meter.save_csv('losshistory/loss_history' + prefix + '.csv')

# for spawning multiple forward pass on same gpu- could be useful for ode forward pass
# https://discuss.pytorch.org/t/split-single-gpu/18651/8
