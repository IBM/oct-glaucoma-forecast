import os
import time

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import umap
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

from data.data_loader import get_data_ts_onh_mac
from losses import binary_cross_entropy_with_logits, mae_globalmean, mae_loss
from models.ode_gru import ODEGRU
from utils.utils import save_model, RunningAverageMeter, select_modalities, subsample
from train_multimodal_latentodegru_sync import  MultimodalTimeSeriesData, VAEData
from losses import  mean_loss_across_time
from scripts.eval_mlode_sync import evaluate_reconstruction_error, create_rnfl_mask
from scripts.eval_linear_regression import create_vft_mask

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


image_dim = 64

number_pixel_out = image_dim * image_dim * 1


def plot_z(mu, val_dx, save_path):
    labels = {2: 'Glaucoma', 1: 'GS', 0: 'Normal'}
    replace = lambda x: labels[x]

    groups = np.array(list(map(replace, val_dx)))
    cdict = {'Glaucoma': 'red', 'Normal': 'green', 'GS': 'yellow'}
    fig, ax = plt.subplots()
    for g in np.unique(groups):
        ix = np.where(groups == g)
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        ax.scatter(mu[ix, 0], mu[ix, 1], color="None", edgecolors=cdict[g], linewidth=2, label=g, s=100)
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def plot_zembedded(mu, val_dx, save_path, type='tsne'):
    assert type in ['tsne', 'umap'], 'type can be one of umap or tsne'
    labels = {2: 'Glaucoma', 1: 'GS', 0: 'Normal'}

    if (type == 'tsne'):
        mu = TSNE(n_components=2, perplexity=30).fit_transform(mu)
    else:
        mu = umap.UMAP(n_neighbors=10,
                       min_dist=0.1, n_components=2,
                       metric='euclidean').fit_transform(mu)

    replace = lambda x: labels[x]
    groups = np.array(list(map(replace, val_dx)))
    cdict = {'Glaucoma': 'red', 'Normal': 'green', 'GS': 'yellow'}
    fig, ax = plt.subplots()
    for g in np.unique(groups):
        ix = np.where(groups == g)
        ax.scatter(mu[ix, 0], mu[ix, 1], color="None", edgecolors=cdict[g], linewidth=2, label=g, s=100)
    ax.legend()
    plt.savefig(save_path)
    plt.close()



def extract_input(x, flags):
    idx = 0 if flags[0] ==1 else 1
    return x[idx]


def compute_loss(model, ts_list, inputs, modality_flags,
                           batch_size=None, compute_error=False):
    """

    :param model:
    :param t_ts:
    :param inputs: list of input modalities; see config  and data loader for the order
    :param modality_flags: array of binary number, e.g [1,1,0];  0 entry denotes the modailty not to be used
    :param annealing_factor:
    :param batch_size:
    :return:
    """

    assert len(inputs) == 2, 'size of input list should be 2'

    assert len(modality_flags) == len(ts_list), 'modality_flags array size should match combintation array'

    input_xi = extract_input(inputs, modality_flags)
    ts = extract_input(ts_list, modality_flags)
    sliceinput = lambda x: x[:,:-1]

    input_xi_sliced = sliceinput(input_xi)

    pred_list = model(ts, input_xi_sliced)
    pred = pred_list[0]
    gt = input_xi[:,[-1]]

    forecast_loss = mean_loss_across_time(gt, pred, 'mse')

    #forecast_loss = torch.sum(
    #    binary_cross_entropy_with_logits(pred.view(pred.shape[0], -1), gt.view(gt.shape[0], -1)), dim=1)



    return torch.mean(forecast_loss)



def compute_forecast_error(model, ts_list, inputs, modality_flags):
    """

    :param model:
    :param ts_list:
    :param inputs: input and ground truth where last channel is the ground truth
    :param modality_flags:
    :return:
    """
    assert len(inputs) == 2, 'size of input list should be 2'
    assert len(modality_flags) == len(ts_list), 'modality_flags array size should match combintation array'
    input_xi = extract_input(inputs, modality_flags)
    ts = extract_input(ts_list, modality_flags)
    sliceinput = lambda x: x[:,:-1]
    input_xi_sliced = sliceinput(input_xi)
    pred_list = model(ts, input_xi_sliced)
    pred = pred_list[0]
    gt = input_xi[:,[-1]]
    pred = torch.sigmoid(pred)
    reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))


    if(modality_flags[0] ==1):
        rnfl_mask = create_rnfl_mask(inputs[0], disc_dia_mm=0.5)
        error = mae_globalmean(reshape(gt[:, [-1]]) * 200,
                            reshape(pred[:, [-1]]) * 200,
                            mask=rnfl_mask[:, [-1]])
    else:
        vft_mask  = create_vft_mask(inputs[1])
        error = mae_loss(reshape(gt[:, [-1]]) * 41, reshape(pred[:, [-1]]) * 41,
                      mask=reshape(vft_mask[:, [-1]]))

    return error, pred, gt






def plot_losses_(train_loss, val_loss, save_path):
    fig, ax = plt.subplots()
    ax.plot(train_loss, label='train loss')
    ax.plot(val_loss, label='val loss')
    ax.legend()
    plt.savefig(save_path)
    plt.close()


def save_losses(config):
    df = pd.DataFrame(list(zip(config.loss_meter.loss_history, config.kl_loss_meter.loss_history,
                               config.loss_meter_test.loss_history,
                               config.kl_loss_meter_test.loss_history)),
                      columns=['trainloss_elbo', 'trainloss_kl', 'val_loss_elbo', 'val_loss_kl'])
    df.to_csv(os.path.join(config.RESULT_DIR, 'losses.csv'))


class Config:

    def create_model(self, load_weights=False):
        raise NotImplementedError('imlimentation required')

    def plot_losses(self):
        train_loss = self.loss_meter.loss_history
        val_loss = self.loss_meter_test.loss_history
        plot_losses_(train_loss, val_loss, save_path=os.path.join(self.RESULT_DIR, 'lossplot.jpeg'))
        train_loss_kld = self.kl_loss_meter.loss_history
        val_loss_kld = self.kl_loss_meter_test.loss_history
        plot_losses_(train_loss_kld, val_loss_kld, save_path=os.path.join(self.RESULT_DIR, 'lossplot_kld.jpeg'))



def getConfig(modalities_, fold_seed_, useode_=True):
    class MyConfig(Config):
        EPOCHS = 100
        BATCH_SIZE=32
        learning_rate = 0.001
        fold_seed = fold_seed_
        use_ode = useode_

        _prefix='odegru' if use_ode else 'gru'
        _suffix = ''
        _suffix = _suffix+ '_foldseed'+str(fold_seed) if fold_seed !=4 else ''

        LOG_ROOT_DIR = _prefix + _suffix
        MODALITIES = modalities_
        assert np.sum(MODALITIES)==1, 'only one modality at a time should be provided'
        type = 'rnfl'  if(MODALITIES[0] ==1)  else 'vft'
        ode_method = 'euler'

        modalities_str = ''.join([str(xi) for xi in MODALITIES]) +type
        prefix = _prefix  + modalities_str
        RESULT_DIR = os.path.join(LOG_ROOT_DIR, prefix)
        MODEL_DIR = os.path.join(RESULT_DIR, 'trained_models')

        loss_meter = RunningAverageMeter()
        loss_meter_test = RunningAverageMeter()

        kl_loss_meter = RunningAverageMeter()
        kl_loss_meter_test = RunningAverageMeter()

        # dx_loss_meter = RunningAverageMeter()

        def __str__(self):
            fields = ['EPOCHS', 'latent_dim', 'image_dim']

        def __init__(self, initialize_model=True):
            mkdir = lambda x: os.mkdir(x) if not os.path.exists(x) else 0
            mkdir(self.LOG_ROOT_DIR)
            mkdir(self.RESULT_DIR)
            mkdir(self.MODEL_DIR)

        def initialize_model_optimizer(self):
            """
            Initializes model and optimizer for trainin
            :return:
            """

            self.model = self.create_model(load_weights=False)
            self.optimizer = optim.Adam( (list(self.model.parameters())), lr=self.learning_rate)



        def create_model(self, load_weights=False, suffix_model_path=''):
            config = self
            ode_solver = [odeint, config.ode_method] if config.use_ode else None

            latentode = ODEGRU(type=self.type, device=device,  ode_solver=ode_solver).to(device)
            if (load_weights):
                model_file = os.path.join(config.RESULT_DIR, 'trained_models',
                                          config.prefix + suffix_model_path + '.pt')
                print('loading weighhts ', model_file)
                # latentode.load_state_dict(torch.load(model_file, map_location='cpu'))
                latentode.load_state_dict(torch.load(model_file, map_location=torch.device('cpu')))

            return latentode

    return MyConfig


def train(epoch, config: Config, data: VAEData, niter_per_epoch=None):
    assert hasattr(config, 'model') and hasattr(config,
                                                'optimizer'), 'model not initialized for training. Call config.initialize_model_optimizer() before '
    model = config.model.train()

    if (niter_per_epoch is None): niter_per_epoch = int(data.size_train() / config.BATCH_SIZE)

    for itr in range(1, niter_per_epoch + 1):


        minibatches, ts_list = data.get_train_minibatch(config.BATCH_SIZE)
        minibatches = subsample(minibatches, config.MODALITIES)

        config.optimizer.zero_grad()
        train_elbo_loss = compute_loss(model, ts_list, minibatches, config.MODALITIES)
        train_elbo_loss.backward()
        config.optimizer.step()
        config.loss_meter.update(train_elbo_loss.item(), accumulate=itr == niter_per_epoch, smooth=True)




def test(epoch, config: Config, data: VAEData):
    model = config.model.eval()
    with torch.no_grad():
        ds_val_batches, ts_list_val, val_dx = data.get_val()
        val_elbo_loss = compute_loss(model, ts_list_val, ds_val_batches, config.MODALITIES)
        val_elbo_loss = np.round(val_elbo_loss.item(), 2)
        config.loss_meter_test.update(val_elbo_loss, accumulate=True, smooth=False)


import io


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents





def print_losses(config, start, end, extra_str=''):
    disp = [['Epoch', epoch], ['trainloss', config.loss_meter.avg],
            ['valloss', config.loss_meter_test.avg], ['time/ep', end - start]
            ]

    print(''.join([str(d[0]) + ': ' + '{:.2f}'.format(d[1]) + ', ' for d in disp]).rstrip(', '), extra_str, flush=True)


def get_forecast_errors(config, data):
    model = config.model.eval()
    with torch.no_grad():
        ds_val_batches, ts_list_val, val_dx = data.get_val()
        error,_,_ = compute_forecast_error(model,ts_list_val,ds_val_batches,config.MODALITIES)
        out = 'error: '+str(torch.mean(error).detach().cpu().numpy())

    return out


# Experiments parameters:
# error type in loss : ce or se
# freeze weights of encoder and decoder
# RNN : turn ode transistion on/off
# ode method of forecasting function : rk4/euler

# 21 aug - lr annealing changed to every 10 epochs and enn oder used dopri5 + se with sum across times loss

if (__name__ == '__main__'):

    idx_r =[ [107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3], [] ]
    modalities_exp = [[1, 0], [0, 1]]

    fold_seed=2
    use_ode = True

    for mm, idxr in zip(modalities_exp, idx_r):

        Config = getConfig(mm , fold_seed_=fold_seed, useode_=use_ode)
        config = Config()
        data = MultimodalTimeSeriesData(fold_seed=config.fold_seed, idx_r=idxr)


        nepochs = config.EPOCHS
        niter_per_epoch = int(data.train_rnflonh.shape[0] / config.BATCH_SIZE)  # 30000#

        print('Training data size', data.size_train())
        print('Val data size', data.size_val())
        print('#Total iterations ', nepochs * niter_per_epoch)
        print('Log directory', config.RESULT_DIR)
        print('Prefix', config.prefix)
        print('Learning rate ', config.learning_rate)

        metrics = []
        config.initialize_model_optimizer()
        for epoch in range(1, nepochs + 1):

            start = time.time()
            train(epoch, config, data, niter_per_epoch=niter_per_epoch)
            test(epoch, config, data)
            end = time.time()

            forecast_eval = get_forecast_errors(config, data)
            print_losses(config, start, end, extra_str=forecast_eval)
            metrics.append(forecast_eval)

            if (epoch % 5 == 0):
                save_model(config.model, config.MODEL_DIR, config.prefix)

                config.plot_losses()
                save_losses(config)

                # recontructon error
                # errors, inputs, preds = evaluate_reconstruction_error(config, data, mode='rec', nv_fc=3)
                # for i, e in zip(inputs, errors):
                #    print(i, ["{0:0.2f}".format(i) if i is not None else None for i in e])

            if (epoch == 25):
                save_model(config.model, config.MODEL_DIR, config.prefix, epoch=epoch)

            if (epoch % 10 == 0):
                for g in config.optimizer.param_groups:
                    g['lr'] = g['lr'] * 0.5
                print('learning rate changed to', str(config.optimizer.param_groups[0]['lr']))

# https://kornia.readthedocs.io/en/latest/augmentation.html
