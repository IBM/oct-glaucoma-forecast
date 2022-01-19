import numpy as np
import torch

from models.pwlinear_regression import pwlr
from scripts.test_reconstruction_mlode import compute_rnfl_error_base, compute_vft_error_base, create_vft_mask
from train_multimodal_latentodegru_sync import MultimodalTimeSeriesData



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



if (__name__ == '__main__'):

    dopred_rnfl = True
    dopred_vft = False
    fold_seed=2


    if (dopred_rnfl):
        idx_r = [ 107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0, seed=2
    else:
        idx_r=[]


    data = MultimodalTimeSeriesData(fold_seed=fold_seed, filter_pr=True, idx_r=idx_r)
    val_maps, val_ts, dx = data.get_val()
    val_maps = [t.detach().cpu().numpy() for t in val_maps]
    val_ts = [t.detach().cpu().numpy() for t in val_ts]

    #val_maps =[v[:2] for v in val_maps]
    #val_ts = [v[:2] for v in val_ts]

    #start index defination: start index 0 - 3 visits, start index 2 means 2 visits



    for start_index in [0, 1]:
        print('Start index, ', start_index)

        if(dopred_rnfl):
            rnfl = val_maps[0][:, start_index:4]
            rnfl_x = rnfl[:, start_index:3]
            ts = val_ts[0][:, start_index:4]
            pred_rnfl = pwlr(rnfl_x, ts)

            [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl, pred_rnfl, mode='forecast',
                                                                                        use_ma=True, quadrant=None)
            print('RNFL, global mean', err_gm, '+-', std_gm)
            for q in [0, 1, 2, 3]:
                [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl, pred_rnfl,
                                                                                            mode='forecast',
                                                                                            use_ma=True, quadrant=q, disc_dia_mm=0.9)
                print('RNFL, global mean quadrant', q, 'error', err_gm, '+-', std_gm)

            # last time time channel of pred_rnfl have forecasted images
        if(dopred_vft):
            vft = val_maps[1][:, start_index:4]
            vft_x = vft[:, start_index:3]
            ts = val_ts[1][:, start_index:4]
            pred_vft = pwlr(vft_x, ts)
            mask = create_vft_mask(torch.from_numpy(vft)).detach().numpy()
            mask = mask.astype(np.bool)


            [err, std], errors = compute_vft_error_base(vft, pred_vft, mask, mode='forecast')

            print('VFT forecast error ', err, '+-', std)



    #Evaluation for larger time gaps, given 1,2,3,4 obs, use 1 2 to forecast 4
    print('Evaluation for larger time gaps')
    for nv_start in [2]:
        print('Start index, ', nv_start)
        # select first nv_start visits and the last one (ground truth)
        slice_inputs_gt = lambda x, nv_fc: x[:, list(range(nv_fc)) + [-1]]
        # select first nv_fc visits
        slice_inputs = lambda x, nv_fc: x[:, list(range(nv_fc)) ]


        if (dopred_rnfl):
            rnfl = val_maps[0][:, :4]
            rnfl_x = slice_inputs(rnfl, nv_start)
            ts =  slice_inputs_gt(val_ts[0], nv_start)
            rnfl_gt = slice_inputs_gt(rnfl, nv_start)
            pred_rnfl = pwlr(rnfl_x, ts)

            [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl_gt, pred_rnfl,
                                                                                        mode='forecast',
                                                                                        use_ma=True, quadrant=None)
            print('RNFL, global mean', err_gm, '+-', std_gm)
            for q in [0, 1, 2, 3]:
                [err, std], [err_gm, std_gm], [errors, errors_gm] = compute_rnfl_error_base(rnfl_gt, pred_rnfl,
                                                                                            mode='forecast',
                                                                                            use_ma=True, quadrant=q,
                                                                                            disc_dia_mm=0.9)
                print('RNFL, global mean quadrant', q, 'error', err_gm, '+-', std_gm)

            # last time time channel of pred_rnfl have forecasted images
        if (dopred_vft):
            vft = val_maps[1][:, start_index:4]

            vft_x = slice_inputs(vft, nv_start)
            ts = slice_inputs_gt(val_ts[1],nv_start)
            vft_gt = slice_inputs_gt(vft, nv_start)
            pred_vft = pwlr(vft_x, ts)
            mask = create_vft_mask(torch.from_numpy(vft)).detach().numpy()
            mask = mask.astype(np.bool)

            [err, std], errors = compute_vft_error_base(vft, pred_vft, mask, mode='forecast')

            print('VFT forecast error ', err, '+-', std)



    print('Done')
