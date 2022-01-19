import os

import numpy as np
from scripts.datautils import MatchedTestSet, save_error
from train_multimodal_latentodegru_sync import VAEData
from train_odegru import getConfig, compute_forecast_error

np.set_printoptions(precision=2)
import cv2
from scripts.test_reconstruction_mlode import compute_rnfl_error, boxplot_vft, boxplot_rnfl
import pandas as pd




def evaluate_forecast_error(config, data: VAEData, nv_fc):
    # forecast the last visit from last nv_fc visits
    inputs, ts_list_val, dx = data.get_val()

    assert nv_fc > 0, 'number of visits to used for forecastin as input should be provided'
    slice_inputs = lambda x: x[:, x.shape[1] - (nv_fc + 1):]
    # inputs_ = [i[:, :nv_fc] if i is not None else None for i in inputs]
    inputs_ = [slice_inputs(i) if i is not None else None for i in inputs]

    slice_inputs_target = lambda x: x[:, x.shape[1] - (nv_fc + 1):]  # inputs + target forecast
    ts_list = [slice_inputs_target(ts) for ts in ts_list_val]

    errors, preds, gt = compute_forecast_error(config.model, ts_list=ts_list, inputs=inputs_,
                                               modality_flags=config.MODALITIES)

    return errors, preds, gt


def evaluate_forecast_error_largetimegap(config, data: VAEData, nv_fc):
    # forecast last visit from first nv_fc visits
    inputs, ts_list_val, dx = data.get_val()

    assert nv_fc > 0, 'number of visits to used for forecastin as input should be provided'
    slice_inputs = lambda x: x[:,
                             list(range(nv_fc)) + [-1]]  # select first nv_fc visits and the last one (ground truth)
    inputs_ = [slice_inputs(i) if i is not None else None for i in inputs]

    slice_inputs_target = lambda x: x[:, list(range(nv_fc)) + [
        -1]]  # select first nv_fc visits and the last one is target forecast  time
    ts_list = [slice_inputs_target(ts) for ts in ts_list_val]

    errors, preds, gt = compute_forecast_error(config.model, ts_list=ts_list, inputs=inputs_,
                                               modality_flags=config.MODALITIES)

    return errors, preds, gt


def evaluate_forecast_error_general(config, data: VAEData, nv_fc, start_index, target_index=-1):
    # forecast last visit from first nv_fc visits
    inputs, ts_list_val, dx = data.get_val()

    Nt = inputs[0].shape[1]
    assert nv_fc is not None and nv_fc > 0, 'expected nv_fc to be positive intiger'
    assert np.abs(nv_fc) < Nt, 'Number of input visits should be less than sequence length'

    assert start_index + nv_fc < Nt, 'start index+nv_fc should be less than sequence length'

    slice_inputs_target = lambda x: x[:, list(range(start_index, nv_fc + start_index)) + [target_index]]

    inputs_target = [slice_inputs_target(i) if i is not None else None for i in inputs]
    ts_list = [slice_inputs_target(ts) for ts in ts_list_val]

    # print('max ts list',ts_list[1].max())

    # Note this function expects inputs=  inputs+ target
    errors, preds, gt = compute_forecast_error(config.model, ts_list=ts_list, inputs=inputs_target,
                                               modality_flags=config.MODALITIES)

    return errors, preds, gt


if (__name__ == '__main__'):

    # modalities_exp = [ [1,0,0],[0,0,1],[1, 0, 1]]

    modalities_exp = [[1,0],[0,1]]
    # experts = ['moe']  # ,'poe'] This was suman's code. Can't be right
    experts = ['poe']
    fold_seed = 4
    results = {}

    # for seed 2

    # suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path = ''
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            target_modality = 'rnfl' if mm[0] == 1 else 'vft'

            if (mm[0] == 1):
                #idx_r = [107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]
                #         3]  # exclusioin for rnfl model to be consistent with other evaluatioins
                idx_r = []
            else:
                idx_r = []
            # data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
            data = MatchedTestSet(seed = fold_seed)
            print('Number of test samples', data.val_rnflonh.shape[0])

            Config = getConfig(mm, fold_seed_=fold_seed, useode_=True)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)

            results[expert + str(mm)] = [[], []]
            savedir = os.path.join(config.LOG_ROOT_DIR, 'testdata')
            if (not os.path.exists(savedir)): os.mkdir(savedir)
            key = expert + str(mm)
            errors_quads = []

            nvisits_all = []
            errors_all = []
            errors_all_seg = [[], [], [], []]

            ### variable visit but target index is fised, use MatchedTestSet
            exp_type = 'num_visits'

            # HHYU: I dont understand why Suman used different start indices
            #start_indices= [4,3,2,1, 0] if mm[0] ==1 else  [0, 2, 2, 1, 0]
            start_indices= [4,3,2,1, 0]
            for nv_fc, si, ti in zip([1, 2, 3, 4, 5], start_indices, [-1, -1, -1, -1, -1]):

            ##fixed visit and target index is moved to increase gap (note 0,0, 1 in si) to make sure e inc,
            ##use this with MatchedTesSet which gives data of sequence length=6
            
            #exp_type = 'larger_gap'
            
            # HHYU: Suman's original code seems to be wrong
            #for nv_fc, si, ti in zip([3, 3, 1], [0, 0, 0], [3, 4, 5]):
            #for nv_fc, si, ti in zip([3, 3, 3], [0, 0, 0], [3, 4, 5]):

                print(config.prefix)
                print('# NV ', nv_fc, 'start', si, 'target', ti)
                # if(nv_fc> 0):
                #    errors_,  preds, gt = evaluate_forecast_error(config, data, nv_fc=nv_fc)
                # else:
                #    errors_, preds, gt = evaluate_forecast_error_largetimegap(config, data, nv_fc=nv_fc*-1)

                errors_, preds, gt = evaluate_forecast_error_general(config, data, nv_fc=nv_fc, start_index=si,
                                                                     target_index=ti)

                errors_ = errors_.detach().cpu().numpy()
                results[key][0].append(errors_)
                results[key][1].append([gt, preds, config.MODALITIES])

                # HHYU: this code is strange. I don't understand it
                # maybe Suman was trying to fix an error in train_odegru.py:compute_forecast_error()
                # Since I changed that code, this line doesnt seem to be needed
                # if(mm[1] ==1): errors_ = (errors_/40.0) *42

                errors_all.append(errors_)
                nvisits_all.append(nv_fc)

                for ii, e in zip([config.MODALITIES], [errors_]):
                    print(ii, "{0:0.2f}+-{1:0.2f}".format(np.mean(e), np.std(e)))

                if (mm[0] == 1):
                    [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error([[gt]], [[preds]],
                                                                                             rnfl_dia_mm=7, use_ma=True,
                                                                                             disc_dia_mm=1.2,
                                                                                             quadrant=None)
                    print('Global [', err_gm, '+-', std_gm, ']')

                    for q in [0, 1, 2, 3]:
                        # HHYU: changed diameters to 7
                        [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error([[gt]], [[preds]],
                                                                                                 rnfl_dia_mm=7,
                                                                                                 use_ma=True,
                                                                                                 quadrant=q,
                                                                                                 disc_dia_mm=1)
                        print('Quad', q, '[', err_gm, '+-', std_gm, ']')
                        errors_all_seg[q].append(abs_err_gm)

            save_error(model_name='ode_gru', target_modality=target_modality, errors=errors_all, nv=nvisits_all,
                       exp_type=exp_type, save_dir='results_' + str(fold_seed))

            if mm == [1, 0]:
                for q in [0, 1, 2, 3]:
                    save_error(model_name='ode_gru_seg' + str(q), target_modality=target_modality, errors=errors_all_seg[q], nv=nvisits_all,
                       exp_type=exp_type, save_dir='results_' + str(fold_seed))

            np.save(os.path.join(savedir, 'testdata_forecast.npy'), results[key])
            e = np.asanyarray(results[key][0])
            if (mm[0] == 1):  # rnfl
                # e_rnfl = list(np.clip(e[:,:], 0, 20))
                e_rnfl = list(e[:, :])
                imbp = boxplot_rnfl(e_rnfl, title='RNFL global mean', show=True)
                cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl_global.jpeg'), imbp)
            else:
                e_vft = list(np.clip(e[:, :], 0, 15))
                #e_vft = list(e[:, :])
                imbp = boxplot_vft(e_vft, title='VFT', show=True)
                cv2.imwrite(os.path.join(savedir, 'box_plot_vft.jpeg'), imbp)

            # quadrants
            # errors_quads = np.asanyarray(errors_quads)
            # loc=['Superior', 'Inferior', 'Temporal','Nasal']
            # for i in range(errors_quads.shape[1]):
            #    imbp = boxplot_rnfl(list(errors_quads[:, i, :]), title ='RNFL '+loc[i], show=True)
            #    cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl'+loc[i]+'.jpeg'), imbp)
