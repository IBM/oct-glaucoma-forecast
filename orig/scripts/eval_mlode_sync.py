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
    temp[:, :, :, 3:-3, 3:-3] = True
    return temp


def create_rnfl_mask(rnfl_imgs, rnfl_diam_mm=7, disc_dia_mm=1.92):
    N, t, c, H, W = rnfl_imgs.shape
    masks = get_mask_rnfl(H, W, rnfl_diam_mm=rnfl_diam_mm, disc_dia_mm=disc_dia_mm).astype(np.uint8)
    masks = masks[np.newaxis, np.newaxis, np.newaxis, :, :]
    masks = torch.from_numpy(masks).to(device)
    masks = masks.repeat(N, t, c, 1, 1)
    return masks


def meanstd(x):
    return torch.mean(x), torch.std(x)


def evaluate_struct_function(model, ts_list_val, inputs, masks, nv_fc_rnfl=3):
    """

    :param model:
    :param ts_list_val:
    :param inputs:
    :param masks:
    :param nv_fc_rnfl: can range from 0-N_t: example nv_fc_rnfl 0 uses RNFL from 1 visit to estimate VFT for the same
    visit, 2 uses RNFL from last 2 visits to estimate VFT for last visit and so on.
    :return:
    """
    model = model.eval()
    modalities = [1, 0]

    nv_fc_vft = -1
    nvs = [nv_fc_rnfl, nv_fc_vft]

    # slice_inputs = lambda x, nv_fc: x[:, x.shape[1] - (nv_fc + 1):-1]
    slice_inputs_target = lambda x, nv_fc: x[:, x.shape[1] - (nv_fc + 1):]  # inputs + target forecast

    #inputs_ = [slice_inputs_target(i, nv) if i is not None and nv > 0 else None for i, nv in zip(inputs, nvs)]
    inputs_ = [slice_inputs_target(i, nv) if i is not None and nv >= 0 else None for i, nv in zip(inputs, nvs)]

    #ts_list = [slice_inputs_target(ts, nv) if nv > 0 else None for ts, nv in zip(ts_list_val, nvs)]
    ts_list = [slice_inputs_target(ts, nv) if nv >= 0 else None for ts, nv in zip(ts_list_val, nvs)]
    inputs_ = subsample(inputs_, modalities)
    ts_list_ = subsample(ts_list, modalities)
    reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

    with torch.no_grad():
        outlist, mulogvar = model(ts_list=ts_list_, x_list=inputs_)
        [pred_rnfl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

        nvs = [nv_fc_rnfl,
               nv_fc_rnfl]  # since the returend vft will be at times at rnfl observation and target forecast
        inputs_target = [slice_inputs_target(inp, nv) if nv >=0 else None for inp, nv in zip(inputs, nvs)]
        masks = [slice_inputs_target(i, nv) if i is not None and nv >=0 else None for i, nv in zip(masks, nvs)]

        rnfl_xi_val, vft_xi_val = inputs_target
        rnfl_mask, vft_mask = masks
        NoneError = torch.ones((5,)) * -100

        errors = []
        for i in range(vft_xi_val.shape[1]):
            error2 = mae_loss(reshape(vft_xi_val[:, [i]]) * 40, reshape(pred_vft[:, [i]]) * 40,
                              mask=reshape(vft_mask[:, [i]])) if vft_xi_val is not None else NoneError
            errors.append(error2)

        return errors


def evaluate_forecast_matrix(model, ts_list_val, inputs, masks, modalities):
    """
    Evaluates the forecasting model by varying number of visits from 0 N_t-1 for each modality, so the output is
    N_t1 x N_t2 matrix
    If there are N visits, then last nv_fc-1 visits will be used to forecast nv_fc th visit
    :param model:
    :param ts: time at each observation (N,t)
    :param inputs: (N,t,1,H,W)
    :param masks: (N,t,1,H,W)
    :param modalities: flag denoting which modality to use
    :param nv_fc: number of input visits to use for forecasting
    :return:  list of errors, list of predictions, list of inputs+target where taget is last index

    """
    model = model.eval()
    NoneError = torch.ones((5,)) * -200  # pred_rnfl is always not none in test phase

    N_t = inputs[0].shape[1]  #
    out = []
    for nv_fc_rnfl in (range(N_t)):  # nv_fc_rnvl [0,1,2,3] we.g, when 3 it uses last 3 visits
        row = []
        for nv_fc_vft in range(N_t):

            if (nv_fc_rnfl == 0 and nv_fc_vft == 0):
                row.append([[NoneError, NoneError]])
            else:
                res = evaluate_forecast_matrix_base(model, ts_list_val, inputs, masks, modalities, nv_fc_rnfl,
                                                    nv_fc_vft)
                row.append(res)

        out.append(row)
    return out


def evaluate_forecast_matrix_base(model, ts_list_val, inputs, masks, modalities, nv_fc_rnfl, nv_fc_vft):
    assert nv_fc_rnfl > 0 or nv_fc_vft > 0, ' one of sequence input visits should be > 0'

    nvs = [nv_fc_rnfl, nv_fc_vft]
    slice_inputs = lambda x, nv_fc: x[:, x.shape[1] - (nv_fc + 1):-1]
    inputs_ = [slice_inputs(i, nv) if i is not None and nv > 0 else None for i, nv in zip(inputs, nvs)]
    masks = [slice_inputs(i, nv) if i is not None and nv > 0 else None for i, nv in zip(masks, nvs)]

    slice_inputs_target = lambda x, nv_fc: x[:, x.shape[1] - (nv_fc + 1):]  # inputs + target forecast
    ts_list = [slice_inputs_target(ts, nv) if nv > 0 else None for ts, nv in zip(ts_list_val, nvs)]

    inputs_ = subsample(inputs_, modalities)
    ts_list_ = subsample(ts_list, modalities)
    with torch.no_grad():
        outlist, mulogvar = model(ts_list=ts_list_, x_list=inputs_)

        [pred_rnfl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

        NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

        reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

        inputs_target = [slice_inputs_target(inp, nv) if nv > 0 else None for inp, nv in zip(inputs, nvs)]
        rnfl_xi_val, vft_xi_val = inputs_target
        rnfl_mask, vft_mask = masks

        error0 = mae_globalmean(reshape(rnfl_xi_val[:, [-1]]) * 200,
                                reshape(pred_rnfl[:, [-1]]) * 200,
                                mask=rnfl_mask[:, [-1]]) if rnfl_xi_val is not None else NoneError

        error2 = mae_loss(reshape(vft_xi_val[:, [-1]]) * 40, reshape(pred_vft[:, [-1]]) * 40,
                          mask=reshape(vft_mask[:, [-1]])) if vft_xi_val is not None else NoneError

        return [error0, error2], [pred_rnfl, pred_vft], inputs_target


#####################################################################################
def evaluate_forecast_matrix_largegap(model, ts_list_val, inputs, masks, modalities):
    """
    Evaluates the forecasting model by varying number of visits from 0 N_t-1 for each modality, so the output is
    N_t1 x N_t2 matrix
    If there are N visits, then last nv_fc-1 visits will be used to forecast nv_fc th visit
    :param model:
    :param ts: time at each observation (N,t)
    :param inputs: (N,t,1,H,W)
    :param masks: (N,t,1,H,W)
    :param modalities: flag denoting which modality to use
    :param nv_fc: number of input visits to use for forecasting
    :return:  list of errors, list of predictions, list of inputs+target where taget is last index

    """
    model = model.eval()
    NoneError = torch.ones((5,)) * -200  # pred_rnfl is always not none in test phase

    N_t = inputs[0].shape[1]  #
    out = []
    for nv_fc_rnfl in (range(N_t-1)):  # nv_fc_rnvl [0,1,2,3] we.g, when 3 it uses last 3 visits
        row = []
        for nv_fc_vft in range(N_t-1):

            if (nv_fc_rnfl == 0 and nv_fc_vft == 0):
                row.append([[NoneError, NoneError]])
            else:
                res = evaluate_forecast_matrix_base_largergap(model, ts_list_val, inputs, masks, modalities, nv_fc_rnfl,
                                                    nv_fc_vft)
                row.append(res)

        out.append(row)
    return out


def evaluate_forecast_matrix_base_largergap(model, ts_list_val, inputs, masks, modalities, nv_fc_rnfl, nv_fc_vft):

    #slice_inputs = lambda x, nv_fc: x[:, x.shape[1] - (nv_fc + 2):-2]
    #list(range(x.shape[1] - (nv_fc + 2), x.shape[1] - (nv_fc + 2)+nv_fc))
    #list(range(x.shape[1] - (nv_fc + 2), x.shape[1] - 2)) #(inputs) derived from above nv_fc cancels out
    #list(range(x.shape[1] - (nv_fc + 2), x.shape[1] - 2)) +[-1] # inputs+ target


    assert nv_fc_rnfl > 0 or nv_fc_vft > 0, ' one of sequence input visits should be > 0'

    nvs = [nv_fc_rnfl, nv_fc_vft]
    slice_inputs = lambda x, nv_fc: x[:, list(range(x.shape[1] - (nv_fc + 2), x.shape[1] - 2))]
    inputs_ = [slice_inputs(i, nv) if i is not None and nv > 0 else None for i, nv in zip(inputs, nvs)]
    masks = [slice_inputs(i, nv) if i is not None and nv > 0 else None for i, nv in zip(masks, nvs)]

    slice_inputs_target = lambda x, nv_fc: x[:, list(range(x.shape[1] - (nv_fc + 2), x.shape[1] - 2)) +[-1]]  # inputs + target forecast
    ts_list = [slice_inputs_target(ts, nv) if nv > 0 else None for ts, nv in zip(ts_list_val, nvs)]

    inputs_ = subsample(inputs_, modalities)
    ts_list_ = subsample(ts_list, modalities)
    with torch.no_grad():
        outlist, mulogvar = model(ts_list=ts_list_, x_list=inputs_)

        [pred_rnfl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

        NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

        reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

        inputs_target = [slice_inputs_target(inp, nv) if nv > 0 else None for inp, nv in zip(inputs, nvs)]
        rnfl_xi_val, vft_xi_val = inputs_target
        rnfl_mask, vft_mask = masks

        error0 = mae_globalmean(reshape(rnfl_xi_val[:, [-1]]) * 200,
                                reshape(pred_rnfl[:, [-1]]) * 200,
                                mask=rnfl_mask[:, [-1]]) if rnfl_xi_val is not None else NoneError

        error2 = mae_loss(reshape(vft_xi_val[:, [-1]]) * 40, reshape(pred_vft[:, [-1]]) * 40,
                          mask=reshape(vft_mask[:, [-1]])) if vft_xi_val is not None else NoneError

        return [error0, error2], [pred_rnfl, pred_vft], inputs_target

#####################################################################################


def evaluate_forecast_error(model, ts_list_val, inputs, masks, modalities, nv_fc, start_index=None, target_index=-1):
    """
    Evaluates the forecasting model.
    If there are N visits, then last nv_fc-1 visits will be used to forecast nv_fc th visit
    :param model:
    :param ts: time at each observation (N,t)
    :param inputs: (N,t,1,H,W)
    :param masks: (N,t,1,H,W)
    :param modalities: flag denoting which modality to use
    :param nv_fc: number of input visits to use for forecasting, if +, the last nv_fc visits will be used to predict
    nv_fc+1th visit, if nv_fc is negative then first nv_fc visits will be used to predict 4th (last) visit
    :param start_index denotes the location from where the input should be sliced
    :return:  list of errors, list of predictions, list of inputs+target_index where taget is last index

    """

    Nt = inputs[0].shape[1]
    assert np.abs(nv_fc) < Nt, 'Number of input visits should be less than sequence length'

    if (start_index is None):  # to support existing api defination
        if (nv_fc < 0):
            start_index = 0  # i.e, use first nv_fc visits
            nv_fc = nv_fc * -1

        else:  # use last nv_fc_visits
            assert nv_fc > 0, 'number of visits to used for forecastin as input should be provided'
            start_index = Nt - (nv_fc + 1)
    else:
        assert nv_fc is not None and nv_fc > 0, 'expected nv_fc to be positive intiger'

    assert start_index + nv_fc < Nt, 'start index+nv_fc should be less than sequence length'

    print('start index, nv_fc', start_index, nv_fc, 'input target_index indices',
          list(range(start_index, nv_fc + start_index)) + [target_index])
    slice_inputs_target = lambda x: x[:, list(range(start_index, nv_fc + start_index)) + [target_index]]

    # select first nv_fc visits
    slice_inputs = lambda x: x[:, list(range(start_index, nv_fc + start_index))]

    # assert nv_fc != 0, 'number of visits to used for forecastin as input should be provided'
    # if(nv_fc >0):
    #     slice_inputs = lambda x: x[:, x.shape[1] - (nv_fc + 1):-1]
    #     slice_inputs_target = lambda x: x[:, x.shape[1] - (nv_fc + 1):]  # inputs + target_index forecast
    # else:
    #     nv_fc = nv_fc* -1
    #     slice_inputs_target = lambda x: x[:, list(range(nv_fc)) + [-1]]
    #     # select first nv_fc visits
    #     slice_inputs = lambda x: x[:, list(range(nv_fc))]

    inputs_ = [slice_inputs(i) if i is not None else None for i in inputs]
    masks = [slice_inputs(i) if i is not None else None for i in masks]

    ts_list = [slice_inputs_target(ts) for ts in ts_list_val]

    inputs_ = subsample(inputs_, modalities)
    ts_list_ = subsample(ts_list, modalities)
    outlist, mulogvar = model(ts_list=ts_list_, x_list=inputs_)

    [pred_rnfl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

    NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

    reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))

    inputs_target = [slice_inputs_target(inp) for inp in inputs]
    rnfl_xi_val, vft_xi_val = inputs_target
    rnfl_mask, gcl_mask, vft_mask = masks
    error0 = mae_globalmean(reshape(rnfl_xi_val[:, [-1]]) * 200,
                            reshape(pred_rnfl[:, [-1]]) * 200,
                            mask=rnfl_mask[:, [-1]]) if pred_rnfl is not None else NoneError
    error1 = NoneError

    error2 = mae_loss(reshape(vft_xi_val[:, [-1]]) * 40, reshape(pred_vft[:, [-1]]) * 40,
                      mask=reshape(vft_mask[:, [-1]])) if pred_vft is not None else NoneError

    return [error0, error2], [pred_rnfl, pred_vft], inputs_target


def evaluate_reconstruction_error(config, data, mode='rec', nv_fc=-1, pretrain_mode=False, start_index=None, target_index=-1):
    """
    Compute the reconstruction from different modality inputs
    :param config:
    :param data:
    :return:
    """
    assert mode in ['rec', 'forecast'], 'expected one of [rec, forecast]'
    if (mode == 'forecast'):
        assert nv_fc != 0, 'number of visits to used for forecastin as input shoulld be provided'

    if (pretrain_mode): assert mode == 'rec', 'when pretrain_mmode is true then mode should be rec'

    model = config.model.eval()
    # comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
    comb = [[1, 1], [1, 0], [0, 1]]  # Suman

    comb = select_modalities(comb, config.MODALITIES)

    with torch.no_grad():
        ds_val_batches, ts_list_val, val_dx = data.get_val(dx_filter=None)
        if (len(ds_val_batches) == 2):
            rnfl_xi_val = ds_val_batches[0]
            vft_xi_val = ds_val_batches[1]
        else:
            rnfl_xi_val = ds_val_batches[0]
            # gcl_xi_val = ds_val_batches[1]
            vft_xi_val = ds_val_batches[2]

        vft_mask = create_vft_mask(vft_xi_val)
        rnfl_mask = create_rnfl_mask(rnfl_xi_val)

        def evaluate_rec_error(inputs, modalities):

            inputs_ = subsample(inputs, modalities)
            ts_list_val_ = subsample(ts_list_val, modalities)  # Suman
            if (nv_fc > 0):
                inputs_ = [inp[:, :nv_fc, :, :, :] if inp is not None else None for inp in inputs_]

            outlist, mulogvar = model(ts_list=ts_list_val_, x_list=inputs_)  ###Suman

            if (pretrain_mode):
                rnfl_xi_val_ = rnfl_xi_val[:, :1]
                vft_xi_val_ = vft_xi_val[:, :1]
                vft_mask_ = vft_mask[:, :1]

            else:
                rnfl_xi_val_ = rnfl_xi_val
                vft_xi_val_ = vft_xi_val
                vft_mask_ = vft_mask

            # in pretrain mode only first visit is predicted from the future visits, so
            # for evaluation slice only the first one
            # if(pretrain_mode):  inputs_ = [inp[:, :1, :, :, :] if inp is not None else None for inp in inputs_]
            # if (pretrain_mode): rnfl_xi_val = rnfl_xi_val [:, :1, :, :, :]

            [pred_rnfl, pred_vft] = map(lambda x: x if x is None else torch.sigmoid(x), outlist)

            NoneError = torch.ones((pred_rnfl.shape[0],)) * -100  # pred_rnfl is always not none in test phase

            reshape = lambda x: x.reshape((x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4]))
            # mae_globalmean
            error0 = mae_loss(reshape(rnfl_xi_val_[:, :]) * 200,
                              reshape(pred_rnfl[:, :]) * 200) if pred_rnfl is not None else NoneError

            error1 = NoneError

            error2 = mae_loss(reshape(vft_xi_val_[:, :]) * 40, reshape(pred_vft[:, :]) * 40,
                              mask=reshape(vft_mask_[:, :])) if pred_vft is not None else NoneError

            return [error0, error2], [pred_rnfl, pred_vft], inputs

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
                error, preds, inputs = evaluate_forecast_error(model, ts_list_val, x_list, masks, c, nv_fc,
                                                               start_index=start_index, target_index=target_index)

            error = [e.cpu().numpy() for e in error]
            error = [e.astype(np.float32) for e in error]
            # error = subsample(error, config.MODALITIES)

            preds = subsample(preds, config.MODALITIES)
            preds_all.append(preds)
            inputs_modalities.append(c)
            errors.append(error)
            inputs_all.append(inputs)

    return errors, inputs_modalities, preds_all, inputs_all
