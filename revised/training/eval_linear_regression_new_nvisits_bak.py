import numpy as np
import torch

from models.pwlinear_regression import pwlr
from scripts.datautils import MatchedTestSet,save_error
from scripts.test_reconstruction_mlode import compute_rnfl_error_base, compute_vft_error_base

GPU = 0
device = torch.device('cuda:' + str(GPU) if torch.cuda.is_available() else 'cpu')

one_ = torch.tensor(1.0).to(device)
zero_ = torch.tensor(0.0).to(device)


def create_vft_mask(vft_imgs):
    N, t, c, H, W = vft_imgs.shape
    temp = torch.sum(vft_imgs, dim=[0, 1, 2], keepdim=True)
    temp = torch.where(temp > 0, one_, zero_)
    temp = temp.repeat((N, t, c, 1, 1))

    temp[:, :, :, :3, :] = False
    temp[:, :, :, -3:, :] = False

    temp[:, :, :, :, :3] = False
    temp[:, :, :, :, -3:] = False

    return temp


def prepare_data(ts_list_val, inputs, masks, nv_fc, start_index=None, target_index=-1):
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

    # slice_inputs = lambda x: x[:, list(range(start_index, nv_fc + start_index))]
    inputs_ = [slice_inputs_target(i) if i is not None else None for i in inputs]
    masks = [slice_inputs_target(i) if i is not None else None for i in masks] if masks is not None else None

    ts_list = [slice_inputs_target(ts) for ts in ts_list_val]

    return ts_list, inputs_, masks


if (__name__ == '__main__'):

    dopred_rnfl = True
    dopred_vft = True
    fold_seed = 2

    # data = MultimodalTimeSeriesData(fold_seed=fold_seed, filter_pr=True, idx_r=idx_r)
    data = MatchedTestSet()

    val_maps, val_ts, dx = data.get_val()
    val_maps = [t.detach().cpu().numpy() for t in val_maps]
    val_ts = [t.detach().cpu().numpy() for t in val_ts]

    nvisits_all = []
    errors_all_rnfl = []
    errors_all_vft = []


    ## variable visit but target index is fised, use MatchedTestSet
    exp_type = 'num_visits'
    for nv_fc, si, ti in zip([1, 2, 3, 4, 5], [4, 3, 2, 1, 0], [-1, -1, -1, -1, -1]):

    #print('Evaluation for larger time gaps')
    ##fixed visit and target index is moved to increase gap (note 0,0, 1 in si) to make sure e inc,
    ##use this with MatchedTesSet which gives data of sequence length=6
    #exp_type = 'larger_gap'
    #for nv_fc, si, ti in zip([3, 3, 3], [0, 0, 0], [3, 4, 5]):
        print('nv_fc', nv_fc, 'start_index', si, 'target_index', ti)
        ts_, inputs_, temp = prepare_data(val_ts, val_maps, masks=None, nv_fc=nv_fc, start_index=si, target_index=ti)
        nvisits_all.append(nv_fc)

        if (dopred_rnfl):
            print('for RNFL')
            rnfl, t = inputs_[0], ts_[0]
            rnfl_x = rnfl[:, :-1]
            pred_rnfl = pwlr(rnfl_x, t)

            [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl, pred_rnfl,
                                                                                        mode='forecast',
                                                                                        use_ma=True, quadrant=None)
            print('RNFL, global mean', err_gm, '+-', std_gm)
            errors_all_rnfl.append(errors_gm)

            for q in [0, 1, 2, 3]:
                [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl, pred_rnfl,
                                                                                            mode='forecast',
                                                                                            use_ma=True, quadrant=q,
                                                                                            disc_dia_mm=0.9)
                print('RNFL, global mean quadrant', q, 'error', err_gm, '+-', std_gm)

            # last time time channel of pred_rnfl have forecasted images
        if (dopred_vft):
            print('for VFT')
            vft, t = inputs_[1], ts_[1]
            vft_x = vft[:, :-1]
            print(vft_x.shape, t.shape)
            pred_vft = pwlr(vft_x, t)
            mask = create_vft_mask(torch.from_numpy(vft).to(device)).detach().cpu().numpy()
            mask = mask.astype(np.bool)

            [err, std], errors = compute_vft_error_base(vft, pred_vft, mask, mode='forecast')
            errors = (errors/40)*42.5
            print('VFT forecast error ', np.mean(errors), '+-', np.std(errors))
            errors_all_vft.append(errors)


    if(dopred_rnfl):
        save_error(model_name='pwlr', target_modality='rnfl', errors=errors_all_rnfl, nv=nvisits_all,
               exp_type=exp_type, save_dir='results')
    if(dopred_vft):
        save_error(model_name='pwlr', target_modality='vft', errors=errors_all_vft, nv=nvisits_all,
               exp_type=exp_type, save_dir='results')

    print('Done')
