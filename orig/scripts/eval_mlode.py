import numpy as np
import torch

from losses import mae_loss, mae_globalmean
from utils.oct_utils import get_mask_rnfl
from utils.utils import select_modalities, subsample

GPU = 0
device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')

one_ = torch.tensor(1.0).to(device)
zero_ = torch.tensor(0.0).to(device)


def create_vft_mask(vft_imgs):
    N, t, c, H, W = vft_imgs.shape
    temp = torch.sum(vft_imgs, dim=[0, 1, 2], keepdim=True)
    temp = torch.where(temp > 0, one_, zero_)
    temp = temp.repeat((N, t, c, 1, 1))
    return temp


def create_rnfl_mask(rnfl_imgs):
    N, t, c, H, W = rnfl_imgs.shape
    masks = get_mask_rnfl(H, W, rnfl_diam_mm=7).astype(np.uint8)
    masks = masks[np.newaxis, np.newaxis, np.newaxis, :, :]
    masks = torch.from_numpy(masks).to(device)
    masks = masks.repeat(N, t, c, 1, 1)
    return masks


def meanstd(x):
    return torch.mean(x), torch.std(x)


def evaluate_forecast_error(model, ts, inputs, masks, modalities, nv_fc):
    """
    Evaluates the forecasting model.
    If there are N visits, then last nv_fc-1 visits will be used to forecast nv_fc th visit
    :param model:
    :param ts: time at each observation (N,t)
    :param inputs: (N,t,1,H,W)
    :param masks: (N,t,1,H,W)
    :param modalities: flag denoting which modality to use
    :param nv_fc: number of input visits to use for forecasting
    :return:  list of errors, list of predictions, list of inputs+target where taget is last index


    """

    assert nv_fc > 0, 'number of visits to used for forecastin as input should be provided'
    slice_inputs = lambda x: x[:, x.shape[1] - (nv_fc + 1):-1]
    # inputs_ = [i[:, :nv_fc] if i is not None else None for i in inputs]
    inputs_ = [slice_inputs(i) if i is not None else None for i in inputs]
    masks = [slice_inputs(i) if i is not None else None for i in masks]

    slice_inputs_target = lambda x: x[:, x.shape[1] - (nv_fc + 1):]  # inputs + target forecast
    ts_ = slice_inputs_target(ts)

    inputs_ = subsample(inputs_, modalities)
    outlist, mulogvar = model(ts=ts_, x_list=inputs_)

    [pred_rnfl, pred_gcl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

    NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

    reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

    inputs_target = [slice_inputs_target(inp) for inp in inputs]
    rnfl_xi_val, gcl_xi_val, vft_xi_val = inputs_target
    rnfl_mask, gcl_mask, vft_mask = masks
    error0 = mae_globalmean(reshape(rnfl_xi_val[:, [-1]]) * 200,
                            reshape(pred_rnfl[:, [-1]]) * 200,
                            mask=rnfl_mask[:, [-1]]) if pred_rnfl is not None else NoneError
    error1 = NoneError

    error2 = mae_loss(reshape(vft_xi_val[:, [-1]]) * 40, reshape(pred_vft[:, [-1]]) * 40,
                      mask=reshape(vft_mask[:, [-1]])) if pred_vft is not None else NoneError

    return [error0, error1, error2], [pred_rnfl, pred_gcl, pred_vft], inputs_target


def evaluate_reconstruction_error(config, data, mode='rec', nv_fc=-1):
    """
    Compute the reconstruction from different modality inputs
    :param config:
    :param data:
    :return:
    """
    assert mode in ['rec', 'forecast'], 'expected one of [rec, forecast]'
    if (mode == 'forecast'):
        assert nv_fc > 0, 'number of visits to used for forecastin as input shoulld be provided'

    model = config.model.eval()
    comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    #comb =[[1,1],[1,0], [0,1]] #Suman

    comb = select_modalities(comb, config.MODALITIES)

    with torch.no_grad():
        ds_val_batches, ts_val, val_dx = data.get_val(dx_filter=None)
        if(len(ds_val_batches)==2):
            rnfl_xi_val = ds_val_batches[0]
            vft_xi_val = ds_val_batches[1]
        else:
            rnfl_xi_val = ds_val_batches[0]
            gcl_xi_val = ds_val_batches[1]
            vft_xi_val = ds_val_batches[2]

        vft_mask = create_vft_mask(vft_xi_val)
        rnfl_mask = create_rnfl_mask(rnfl_xi_val)

        def evaluate_rec_error(inputs, modalities):

            inputs_ = subsample(inputs, modalities)
            #ts_val_ = subsample([ts_val, ts_val], modalities) #Suman
            if (nv_fc > 0):
                inputs_ = [inp[:, :nv_fc, :, :, :] if inp is not None else None for inp in inputs_]

            outlist, mulogvar = model(ts=ts_val, x_list=inputs_) ###Suman

            [pred_rnfl, pred_gcl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

            NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

            reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            # mae_globalmean
            error0 = mae_loss(reshape(rnfl_xi_val[:, :]) * 200,
                              reshape(pred_rnfl[:, :]) * 200) if pred_rnfl is not None else NoneError

            error1 = NoneError

            error2 = mae_loss(reshape(vft_xi_val[:, :]) * 40, reshape(pred_vft[:, :]) * 40,
                              mask=reshape(vft_mask[:, :])) if pred_vft is not None else NoneError

            return [error0, error1, error2], [pred_rnfl, pred_gcl, pred_vft], inputs

        masks = [rnfl_mask, None, vft_mask]
        errors = [];
        inputs_modalities = []
        preds_all = []
        inputs_all = []

        # if (config.MODALITIES[1] == 1):  # if RNFL is used in training
        for c in comb:
            # x_list_c = subsample(ds_val_batches, c)
            x_list = ds_val_batches
            if (mode == 'rec'):
                error, preds, inputs = evaluate_rec_error(x_list, c)
            else:
                error, preds, inputs = evaluate_forecast_error(model, ts_val, x_list, masks, c, nv_fc)

            error = [e.cpu().numpy() for e in error]
            error = [e.astype(np.float32) for e in error]
            # error = subsample(error, config.MODALITIES)

            preds = subsample(preds, config.MODALITIES)
            preds_all.append(preds)
            inputs_modalities.append(c)
            errors.append(error)
            inputs_all.append(inputs)

    return errors, inputs_modalities, preds_all, inputs_all
