import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from losses import elbo_general
from models.vae import VAE
from utils.utils import RunningAverageMeter, save_model
from data.data_loader import get_data_ts
from matplotlib import pyplot as plt


GPU = 0
device = torch.device('cuda:' + str(GPU)

                      if torch.cuda.is_available() else 'cpu')

print('Device:', device.type)

BATCH_SIZE = 256

image_dim = 64









class VAEData:

    def get_train_minibatch(self, batch_size):
        return None
    def get_val(self):
        return None

class VFTData (VAEData):

    def __init__(self, path='../oct_forecasting/temp/vft_data.npz'):
        data = np.load(path)
        images_ = data['images'][:, np.newaxis, :, :]
        images_ = images_ / 40.0

        self.images = torch.from_numpy(images_).float().to(device)
        split = int(0.9 * self.images.shape[0])
        self.train = self.images[:split]
        self.val = self.images[split:]

    def get_train_minibatch(self, batch_size):
        # idx = np.random.permutation(len(self.train))
        idx = torch.randperm(self.train.shape[0], device=device)

        return self.train[idx[:batch_size]]

    def get_val(self):
        return self.val


class RNFLData(VAEData):

    def __init__(self):
        maps, _,_,_ = get_data_ts('train',mask_disc=True,filter_missing_vft=False)
        self.train = maps[0]
        maps, _, _, _ = get_data_ts('val',mask_disc=True, filter_missing_vft=False)
        self.val = maps[0]
        maps, _, _, _ = get_data_ts('test', mask_disc=True, filter_missing_vft=False)
        self.test = maps[0]

        self.train = np.vstack([self.train, self.test])
        #stack across time and make channel first
        N,t,H,W,c = self.train.shape
        self.train = self.train.reshape(N*t, H,W,c)
        self.train = self.train.transpose([0,3,1,2])

        N, t, H, W, c = self.val.shape
        self.val = self.val.reshape(N * t, H, W, c)
        self.val = self.val.transpose([0, 3, 1, 2])


        self.train = torch.from_numpy(self.train).float().to(device)
        self.val = torch.from_numpy(self.val).float().to(device)



    def get_train_minibatch(self, batch_size):
        # idx = np.random.permutation(len(self.train))
        idx = torch.randperm(self.train.shape[0], device=device)

        return self.train[idx[:batch_size]]

    def get_val(self):
        return self.val

#
#
# class Config:
#     EPOCHS = 200
#     learning_rate = 0.001
#     annealing_epochs = 20
#     latent_dim = 2
#     image_dim = 14
#     LOG_ROOT_DIR = 'temp'
#     type = 'vft'
#     prefix = 'vae_' + str(latent_dim) + '_' + type
#     RESULT_DIR = os.path.join(LOG_ROOT_DIR, prefix)
#     MODEL_DIR = os.path.join(RESULT_DIR, 'trained_models')
#
#     loss_meter = RunningAverageMeter()
#     kl_loss_meter = RunningAverageMeter()
#     dx_loss_meter = RunningAverageMeter()
#     loss_meter_test = RunningAverageMeter()
#
#     def __str__(self):
#         fields = ['EPOCHS', 'latent_dim', 'image_dim']
#
#     def __init__(self):
#         mkdir = lambda x: os.mkdir(x) if not os.path.exists(x) else 0
#         mkdir(self.LOG_ROOT_DIR)
#         mkdir(self.RESULT_DIR)
#         mkdir(self.MODEL_DIR)
#
#         self.model = self.create_model()
#         params = (list(self.model.parameters()))
#         self.optimizer = optim.Adam(params, lr=self.learning_rate)
#
#     def create_model(self, load_weights=False):
#         config = self
#         latentode = VAE(latent_dim=config.latent_dim, type=config.type).to(device)
#         if (load_weights):
#             model_file = os.path.join(config.RESULT_DIR, 'trained_models', config.prefix + '.pt')
#             latentode.load_state_dict(torch.load(model_file, map_location='cpu'))
#
#         return latentode


class Config:
    EPOCHS = 200
    learning_rate = 0.001
    annealing_epochs = 20
    latent_dim = 2
    image_dim = 64
    LOG_ROOT_DIR = 'temp'
    type = 'rnfl'
    prefix = 'vae_' + str(latent_dim) + '_' + type
    RESULT_DIR = os.path.join(LOG_ROOT_DIR, prefix)
    MODEL_DIR = os.path.join(RESULT_DIR, 'trained_models')

    loss_meter = RunningAverageMeter()
    kl_loss_meter = RunningAverageMeter()
    dx_loss_meter = RunningAverageMeter()
    loss_meter_test = RunningAverageMeter()

    def __str__(self):
        fields = ['EPOCHS', 'latent_dim', 'image_dim']

    def __init__(self):
        mkdir = lambda x: os.mkdir(x) if not os.path.exists(x) else 0
        mkdir(self.LOG_ROOT_DIR)
        mkdir(self.RESULT_DIR)
        mkdir(self.MODEL_DIR)

        self.model = self.create_model()
        params = (list(self.model.parameters()))
        self.optimizer = optim.Adam(params, lr=self.learning_rate)

    def create_model(self, load_weights=False):
        config = self
        latentode = VAE(latent_dim=config.latent_dim, type=config.type).to(device)
        if (load_weights):
            model_file = os.path.join(config.RESULT_DIR, 'trained_models', config.prefix + '.pt')
            latentode.load_state_dict(torch.load(model_file, map_location='cpu'))

        return latentode




def train(epoch, config: Config, vftdata: VAEData,niter_per_epoch=None):
    model = config.model.train()

    if(niter_per_epoch is None): niter_per_epoch = int(vftdata.train.shape[0] / BATCH_SIZE)

    for itr in range(1, niter_per_epoch + 1):

        if epoch < config.annealing_epochs:
            # compute the KL annealing factor for the current mini-batch in the current epoch
            annealing_factor = (float(itr + (epoch - 1) * niter_per_epoch + 1) /
                                float(config.annealing_epochs * niter_per_epoch))
        else:
            # by default the KL annealing factor is unity
            annealing_factor = 1.0

        minibatch = vftdata.get_train_minibatch(BATCH_SIZE)
        [pred], [mu, logvar] = model(minibatch)
        train_elbo_loss, kld = elbo_general([pred], [minibatch], mu, logvar, annealing_factor=annealing_factor)

        config.optimizer.zero_grad()

        kl_weight = annealing_factor
        train_elbo_loss.backward()
        config.optimizer.step()
        config.loss_meter.update(train_elbo_loss.item())
        config.kl_loss_meter.update(torch.mean(kld).item())


def test(epoch, config: Config, vftdata: VAEData):
    model = config.model.eval()
    with torch.no_grad():
        ds_val_batch = vftdata.get_val()
        [pred], [mu, logvar] = model(ds_val_batch)
        val_elbo_loss, kld = elbo_general([pred], [ds_val_batch], mu, logvar)
        val_elbo_loss = np.round(val_elbo_loss.item(), 2)
        config.loss_meter_test.update(val_elbo_loss)

        if(config.latent_dim==2 and epoch % 10 ==0):
            mu, _= model.infer(ds_val_batch)
            mu = mu.detatch().cpu().numpy()
            plt.scatter(mu[:,0], mu[:,1])
            plt.savefig(os.path.join(config.RESULT_DIR, 'zplots'+str(epoch)+'.jpeg'))
            plt.close()



def viz_latent_space(model, save_path,image_size):
    from utils.viz_latent import generate_random_reconstructions, generate_latent_space
    import cv2

    modalities = [model.decoder]
    file_ext = save_path.split('.')
    if (model.latent_dim >= 2):
        viz_rand = generate_random_reconstructions(modalities, device=device, latent_dim=model.latent_dim, image_size=image_size)
        cv2.imwrite(file_ext[0]+'randsample'+'.'+file_ext[1], viz_rand)

    if (model.latent_dim  == 2):
        viz_rand = generate_latent_space(modalities, device=device, latent_dim=model.latent_dim, image_size=image_size, grid_size=16)
        cv2.imwrite(file_ext[0]+'latentspace'+'.'+file_ext[1], viz_rand)





def print_losses(config, start, end):
    disp = [['Epoch', epoch], ['trainloss', config.loss_meter.avg], ['kl_loss', config.kl_loss_meter.avg],
            ['valloss', config.loss_meter_test.avg], ['time/ep', end - start]
            ]

    print(''.join([str(d[0]) + ': ' + '{:.2f}'.format(d[1]) + ', ' for d in disp]).rstrip(', '), flush=True)


if (__name__ == '__main__'):
    #vftdata = VFTData()
    vftdata = RNFLData()
    config = Config()

    BATCH_SIZE =64
    nepochs = config.EPOCHS
    niter_per_epoch = 500#int(vftdata.train.shape[0] / BATCH_SIZE)   #30000#

    print('Training data size', vftdata.train.shape[0])
    print('Val data size', vftdata.val.shape[0])
    print('#Total iterations ', nepochs * niter_per_epoch)
    print('Prefix', config.prefix)



    for epoch in range(1, nepochs + 1):

        start = time.time()
        train(epoch,config, vftdata)
        test(epoch, config, vftdata)
        end = time.time()
        print_losses(config, start, end)

        if(epoch %10 ==0):
            save_model(config.model,config.MODEL_DIR,config.prefix)
            viz_latent_space(config.model.eval(), os.path.join(config.RESULT_DIR, 'latent'+str(epoch)+'.png'), image_size=32)
