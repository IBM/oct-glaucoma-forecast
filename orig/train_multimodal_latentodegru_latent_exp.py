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
from losses import elbo_general_timeseries
from models.multiodal_latentodegru import MultimodalLatentODE
from scripts.eval_mlode import evaluate_reconstruction_error
from utils.utils import save_model, RunningAverageMeter, select_modalities, subsample

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


def subsampled_elbo_losses(model, t_ts, inputs, modality_flags, annealing_factor=1.,
                           batch_size=None):
    """

    :param model:
    :param t_ts:
    :param inputs: list of input modalities; see config  and data loader for the order
    :param modality_flags: array of binary number, e.g [1,1,0];  0 entry denotes the modailty not to be used
    :param annealing_factor:
    :param batch_size:
    :return:
    """

    assert len(inputs) == 3, 'size of input list should be 3'
    comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    # comb = [[1, 1, 0], [0, 1, 0], [1, 0, 0]  ]
    comb = select_modalities(comb, modality_flags)
    assert len(comb) > 0, 'no modality is selected'

    loss = 0
    KLD = 0
    for c in comb:
        # print('LOss for ', str(c))
        assert len(c) == len(modality_flags), 'modality_flags array size should match combintation array'
        # c = [a * b for a, b in zip(c, modality_flags)]
        input_xi_sub = subsample(inputs, c)
        pred_list, [mu, logvar] = model(t_ts, input_xi_sub)
        # if the loss of gcl or vft not reqired then  set the correspndng index in pred_list and input_ts to None before passing below

        pred_list_ = subsample(pred_list, c)
        inputs_ = subsample(inputs, c)

        # pred_list[2] = None
        # inputs_ts[2] = None

        # pred_list[1] = None
        # inputs_ts[1] = None

        loss_, KLD_ = elbo_general_timeseries(preds=pred_list_, gts=inputs_, mu=mu, logvar=logvar,
                                              annealing_factor=annealing_factor, loss_type='ce')

        loss += loss_
        KLD += KLD_

    return loss, KLD


class VAEData:

    def get_train_minibatch(self, batch_size):
        raise NotImplementedError('imlimentation required')

    def get_val(self):
        raise NotImplementedError('imlimentation required')

    def size_train(self):
        raise NotImplementedError('imlimentation required')

    def size_val(self):
        raise NotImplementedError('imlimentation required')


from data.utils import resize_stack


def process_maps(train, val, test):
    train = np.vstack([train, test])

    N, t, H, W, c = train.shape
    train = train.transpose([0, 1, 4, 2, 3])  # (N,t, c, H,W)

    N, t, H, W, c = val.shape
    val = val.transpose([0, 1, 4, 2, 3])  # (N,t,c,H,W)

    train = torch.from_numpy(train).float().to(device)
    val = torch.from_numpy(val).float().to(device)
    return train, val


def process_labels(train, val, test):
    train = np.vstack([train, test])

    train = torch.from_numpy(train).float().to(device)
    val = torch.from_numpy(val).float().to(device)
    return train, val


from cutils.common import normalize_range

# import kornia as K

from data.data_onh import filteridx_by_progression_rate


class MultimodalTimeSeriesData(VAEData):

    def __init__(self, fold_seed):
        train, val, test = get_data_ts_onh_mac(mask_onhrnfl_disc=True,fold_seed=fold_seed)

        rnfls_onh = train[0][0], val[0][0], test[0][0]
        rnfls_onh = [resize_stack(d, (32, 32)) for d in rnfls_onh]

        self.train_rnflonh, self.val_rnflonh = process_maps(rnfls_onh[0], rnfls_onh[1], rnfls_onh[2])
        # self.train_gclmac, self.val_gclmac = process_maps(train[0][3], val[0][3], test[0][3])
        self.train_rnflmac, self.val_rnflmac = process_maps(train[0][2], val[0][2], test[0][2])

        self.train_vft, self.val_vft = process_maps(train[0][4], val[0][4], test[0][4])
        assert self.train_rnflonh.shape[0] == self.train_rnflmac.shape[0], ' Number of maps should be same '

        self.train_dx, self.val_dx = process_labels(train[2][0], val[2][0], test[2][0])

        self.age_at_vd_train, self.age_at_vd_val = process_labels(train[1], val[1], test[1])

        # to years
        self.age_at_vd_train = self.age_at_vd_train / 12.0
        self.age_at_vd_val = self.age_at_vd_val / 12.0

        self.filter()

        # transform in [-1, 1]
        self.age_at_vd_train = normalize_range(self.age_at_vd_train, [20, 80], [-1, 1])
        self.age_at_vd_val = normalize_range(self.age_at_vd_val, [20, 80], [-1, 1])

        # self.rotate = K.augmentation.RandomRotation(4, same_on_batch=True)

    def filter(self):
        # Filter unreaslitic samples that have positive growth rate
        idx_trn, mb_ = filteridx_by_progression_rate(self.train_rnflonh * 200, self.age_at_vd_train,
                                                     slope_threshold=0.5)
        idx_trn = idx_trn[0]
        self.train_rnflonh, self.train_rnflmac, self.train_vft, self.age_at_vd_train, self.train_dx = [d[idx_trn] for d
                                                                                                       in
                                                                                                       [
                                                                                                           self.train_rnflonh,
                                                                                                           self.train_rnflmac,
                                                                                                           self.train_vft,
                                                                                                           self.age_at_vd_train,
                                                                                                           self.train_dx]]

        slope_threshold_val = 0  # 0#0.5
        idx_val, mb_ = filteridx_by_progression_rate(self.val_rnflonh * 200, self.age_at_vd_val,
                                                     slope_threshold=slope_threshold_val, mask_base=mb_)
        self.val_rnflonh, self.val_rnflmac, self.val_vft, self.age_at_vd_val, self.val_dx = [d[idx_val] for d in
                                                                                             [self.val_rnflonh,
                                                                                              self.val_rnflmac,
                                                                                              self.val_vft,
                                                                                              self.age_at_vd_val,
                                                                                              self.val_dx]]

    def augment_rot(self, amap):
        # apply same rotation for a sample for all the time point
        temp = [self.rotate(d) for d in amap]
        return torch.stack(temp)

    def get_train_minibatch(self, batch_size):
        """
        :param dx_filter:
        :return: maps each os size (batch_size, t,c,H,W) and ts of size (batch_size, t)
        """
        # idx = np.random.permutation(len(self.train))
        idx = torch.randperm(self.train_rnflonh.shape[0], device=device)

        aug_rnfl_batch = self.train_rnflonh[idx[:batch_size]]  # self.augment_rot(self.train_rnflonh[idx[:batch_size]])

        maps = aug_rnfl_batch, self.train_rnflmac[idx[:batch_size]], self.train_vft[
            idx[:batch_size]]
        ts = self.age_at_vd_train[idx[:batch_size]]
        return maps, ts

    def get_val(self, dx_filter=None):
        """

        :param dx_filter:
        :return: maps each os size (N, t,c,H,W) and dx of size (N,)
        """
        maps = [self.val_rnflonh, self.val_rnflmac, self.val_vft]

        # reduce from N,t to N,1 ie one diagnosis for time sample

        ts = self.age_at_vd_val

        dx = torch.max(self.val_dx, dim=1)[0]  # (N,1)
        if (dx_filter):
            maps = [m[dx == dx_filter] for m in maps]
            dx = dx[dx == dx_filter]
            ts = ts[dx == dx_filter]

        return maps, ts, dx

    def size_train(self):
        return self.train_rnflonh.shape[0]

    def size_val(self):
        return self.val_rnflonh.shape[0]


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


# from train_multimodalvae import getConfig as getConfigMVAE
from train_multimodal_latentodegru_pretrain_latent_exp import getConfig as getConfigMLODE_pretrained


def getConfig(modalities_, expert_, latent_dim_, fold_seed_):
    class MyConfig(Config):
        EPOCHS = 50
        learning_rate = 0.001
        annealing_epochs = 10
        latent_dim = latent_dim_
        image_dim = 64
        split_latent = False
        vfim_dim = 32
        #fold_seed =1
        fold_seed = fold_seed_
        _suffix = '_spl' if split_latent else ''
        _suffix = _suffix+ '_foldseed'+str(fold_seed) if fold_seed !=4 else ''
        #LOG_ROOT_DIR = 'nonadj_euler_ce_temp_mlode' + str(latent_dim) + _suffix
        LOG_ROOT_DIR = 'latent_exp_mlode' + _suffix
        MODALITIES = modalities_  # [0, 0, 1]
        expert = expert_  # 'moe'
        ode_method = 'euler'

        modalities_str = ''.join([str(xi) for xi in MODALITIES])
        prefix = 'multimoda_latentode' + str(latent_dim) + modalities_str + '_' + expert
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

            load_pretrained_weights = True

            if (load_pretrained_weights):
                ConfigMLODEPT = getConfigMLODE_pretrained(self.MODALITIES, self.expert, self.latent_dim)
                configmlode = ConfigMLODEPT()
                configmlode.model = configmlode.create_model(load_weights=True)
                #configmlode.model.forecast_ode=None
                #updated_params.pop('fc.weight', None)
                #updated_params.pop('fc.bias', None)

                source_model_state_dict = configmlode.model.state_dict()
                exclude_struct = ['forecast_ode.fc1.weight', 'forecast_ode.fc2.weight', 'forecast_ode.fc3.weight',
                                  'forecast_ode.fc1.bias', 'forecast_ode.fc2.bias', 'forecast_ode.fc3.bias']
                for es in exclude_struct:
                    _ = source_model_state_dict.pop(es)

                #print('A ',source_model_state_dict.keys())
                #pretrained_dict = {k: v for k, v in source_model_state_dict.items() if k not in exclude_struct}
                #source_model_state_dict.update(pretrained_dict)
                #print('B ',source_model_state_dict.keys())

                dst_model = self.model
                dst_model.load_state_dict(source_model_state_dict, strict=False)

            # for param in dst_model.parameters():
            #    param.requires_grad = False

            # source_model = configmlode.model.vae_rnfl.decoder
            # dst_model = self.model.decoder_rnfl
            # dst_model.load_state_dict(source_model.state_dict())
            # for param in dst_model.parameters():
            #    param.requires_grad = False

            self.optimizer = optim.Adamax(filter(lambda p: p.requires_grad, self.model.parameters()),
                                          lr=self.learning_rate)

            # rnn_encoder_path ='temp_mvaeld32'
            # self.model.rnn_rnfl.rnn.load_inner_encoder(rnn_encoder_path)

        def create_model(self, load_weights=False, suffix_model_path=''):
            config = self
            ode_solver = [odeint, config.ode_method]

            latentode = MultimodalLatentODE(latent_dim=config.latent_dim,
                                            device=device, expert=config.expert, ode_solver=ode_solver,
                                            split_latent=self.split_latent).to(device)
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

    if (niter_per_epoch is None): niter_per_epoch = int(data.size_train() / BATCH_SIZE)

    for itr in range(1, niter_per_epoch + 1):

        if epoch < config.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(itr + (epoch - 1) * niter_per_epoch + 1) /
                                float(config.annealing_epochs * niter_per_epoch))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        minibatches, ts = data.get_train_minibatch(BATCH_SIZE)
        minibatches = subsample(minibatches, config.MODALITIES)

        train_elbo_loss, kld = subsampled_elbo_losses(model, ts, minibatches, config.MODALITIES,
                                                      annealing_factor=annealing_factor)

        config.optimizer.zero_grad()

        train_elbo_loss.backward()
        config.optimizer.step()
        config.loss_meter.update(train_elbo_loss.item(), accumulate=itr == niter_per_epoch, smooth=True)
        config.kl_loss_meter.update(torch.mean(kld).item(), accumulate=itr == niter_per_epoch, smooth=True)


def visualize_embedding(config: Config, data: VAEData, epoch=None, type='umap'):
    # if the latent dimension >2 the type algorithm is used to embed the latent space to 2 dimensions
    if epoch is None: epoch = ''

    model = config.model.eval()
    with torch.no_grad():
        ds_val_batches, ts_val, val_dx = data.get_val()
        ds_val_batches_sub = subsample(ds_val_batches, config.MODALITIES)
        val_dx = val_dx.cpu().numpy()
        mu, _ = model.infer(ts_val, ds_val_batches_sub)
        mu = mu.cpu().numpy()
        if (config.latent_dim == 2): plot_z(mu, val_dx, save_path=os.path.join(config.RESULT_DIR,
                                                                               'zplots' + str(epoch) + '.jpeg'))
        if (config.latent_dim > 2): plot_zembedded(mu, val_dx, save_path=os.path.join(config.RESULT_DIR,
                                                                                      'embed_zplots' + str(
                                                                                          epoch) + '.jpeg'), type=type)


def test(epoch, config: Config, data: VAEData):
    model = config.model.eval()
    with torch.no_grad():
        ds_val_batches, ts_val, val_dx = data.get_val()
        # preds, [mu, logvar] = model(ds_val_batches)
        val_elbo_loss, kld = subsampled_elbo_losses(model, ts_val, ds_val_batches, config.MODALITIES)
        val_elbo_loss = np.round(val_elbo_loss.item(), 2)
        config.loss_meter_test.update(val_elbo_loss, accumulate=True, smooth=False)
        config.kl_loss_meter_test.update(torch.mean(kld).item(), accumulate=True, smooth=False)


import io


def print_to_string(*args, **kwargs):
    output = io.StringIO()
    print(*args, file=output, **kwargs)
    contents = output.getvalue()
    output.close()
    return contents


def viz_latent_space(model, modalities, save_path, image_size):
    from utils.viz_latent import generate_random_reconstructions, generate_latent_space
    import cv2

    generators = [model.decoder_rnfl, model.decoder_gcl, model.decoder_vft]
    assert len(generators) == len(modalities), 'check modality numbers '
    generators = [g for g, m in zip(generators, modalities) if m]

    file_ext = save_path.split('.')
    if (model.latent_dim >= 2):
        viz_rand = generate_random_reconstructions(generators, device=device, latent_dim=model.latent_dim,
                                                   image_size=image_size)
        cv2.imwrite(file_ext[0] + 'randsample' + '.' + file_ext[1], viz_rand)

    if (model.latent_dim == 2):
        viz_rand = generate_latent_space(generators, device=device, latent_dim=model.latent_dim, image_size=image_size,
                                         grid_size=16)
        cv2.imwrite(file_ext[0] + 'latentspace' + '.' + file_ext[1], viz_rand)


def print_losses(config, start, end, extra_str=''):
    disp = [['Epoch', epoch], ['trainloss', config.loss_meter.avg], ['kl_loss', config.kl_loss_meter.avg],
            ['valloss', config.loss_meter_test.avg], ['time/ep', end - start]
            ]

    print(''.join([str(d[0]) + ': ' + '{:.2f}'.format(d[1]) + ', ' for d in disp]).rstrip(', '), extra_str, flush=True)


def get_rec_errors(config, data, mode, nv_fc):
    # recontructon error
    out = ''
    errors, inputs_c, preds, _ = evaluate_reconstruction_error(config, data, mode=mode, nv_fc=nv_fc)
    for i, err in zip(inputs_c, errors):

        out += str(i) + ' ' + str(["{0:0.2f}".format(np.mean(i)) if i is not None else None for i in err])
        # print(i, ["{0:0.2f}".format(i) if i is not None else None for i in e])

    return out


# Experiments parameters:
# error type in loss : ce or se
# freeze weights of encoder and decoder
# RNN : turn ode transistion on/off
# ode method of forecasting function : rk4/euler

# 21 aug - lr annealing changed to every 10 epochs and enn oder used dopri5 + se with sum across times loss

if (__name__ == '__main__'):
    # vftdata = VFTData()
    # modalities_exp = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    # experts = ['moe', 'poe']

    # running on elboall

    experts = ['poe']

    # running on mvae
    # modalities_exp = [[0,0,1] ]
    # experts = ['moe']

    latent_dims = [2,4,8, 16, 32,64]

    fold_seed=4
    for latent_dim  in latent_dims:
        for expert in experts:
            #if (expert == 'moe'): modalities_exp = [[1, 0, 0], [0, 0, 1], [1, 0, 1]]
            if (expert == 'poe'):  modalities_exp = [[1, 0, 1]]

            for mm in modalities_exp:

                Config = getConfig(mm, expert, latent_dim_=latent_dim, fold_seed_=fold_seed)
                config = Config()
                data = MultimodalTimeSeriesData(fold_seed=config.fold_seed)

                BATCH_SIZE = 32
                nepochs = config.EPOCHS
                niter_per_epoch = int(data.train_rnflonh.shape[0] / BATCH_SIZE)  # 30000#

                print('Training data size', data.size_train())
                print('Val data size', data.size_val())
                print('#Total iterations ', nepochs * niter_per_epoch)
                print('Log directory', config.RESULT_DIR)
                print('Prefix', config.prefix)
                print('Learning rate ', config.learning_rate)

                config.initialize_model_optimizer()
                for epoch in range(1, nepochs + 1):

                    start = time.time()
                    train(epoch, config, data, niter_per_epoch=niter_per_epoch)
                    test(epoch, config, data)
                    end = time.time()

                    rec_eval = get_rec_errors(config, data, mode='rec', nv_fc=-1)
                    print_losses(config, start, end, extra_str=rec_eval)

                    if (epoch % 5 == 0):
                        save_model(config.model, config.MODEL_DIR, config.prefix)
                        viz_latent_space(config.model.eval(), config.MODALITIES,
                                         os.path.join(config.RESULT_DIR, 'latent' + str(epoch) + '.png'), image_size=32)

                        config.plot_losses()
                        save_losses(config)
                        visualize_embedding(config, data, epoch=epoch, type='umap')

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
