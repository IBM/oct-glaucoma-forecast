import os

import numpy as np

from data.utils import mask_rnfl
from scripts.eval_mlode_sync import evaluate_reconstruction_error, create_vft_mask
from train_multimodal_latentodegru_sync import getConfig, MultimodalTimeSeriesData
from scripts.datautils import MatchedTestSet, save_error
from utils.oct_utils import get_quardrants

np.set_printoptions(precision=2)
import tempfile
import cv2
from scripts.test_reconstruction_mlode import  compute_vft_error, compute_rnfl_error, boxplot_vft, boxplot_rnfl


if (__name__ == '__main__'):

    # modalities_exp = [ [1,0,0],[0,0,1],[1, 0, 1]]

    modalities_exp = [[1, 0],[0,1]]
    experts = ['moe']  # ,'poe']
    fold_seed = 2
    results = {}
    # for seed 2
    idx_r = [ 107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0

    #idx_r = [ 84, 324, 162, 3]  # filter for [1,0] input ie rnfl comb index=2
    rnfl_comb_index=0

    #data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
    data=MatchedTestSet()
    print('Number of test samples', data.val_rnflonh.shape[0])
    #suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path=''
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            target_modality = 'rnfl' if mm[0] == 1 else 'vft'
            Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)

            results[expert + str(mm)] = [[], []]
            savedir = os.path.join(config.LOG_ROOT_DIR, 'testdata')
            if (not os.path.exists(savedir)): os.mkdir(savedir)
            key = expert + str(mm)
            errors_quads=[]

            nvisits_all = []
            errors_all = []


            ## variable visit but target index is fised, use MatchedTestSet
            #exp_type = 'num_visits'
            #for nv_fc, si, ti in zip([1, 2, 3, 4, 5], [4, 3, 2, 1, 0], [-1, -1, -1, -1, -1]):

            ##fixed visit and target index is moved to increase gap (note 0,0, 1 in si) to make sure e inc,
            ##use this with MatchedTesSet which gives data of sequence length=6
            exp_type = 'larger_gap'
            for nv_fc, si, ti in zip([3, 3, 2], [0, 0, 0], [3, 4, 5]):

                print('# NV ', nv_fc, 'start_index', si)
                errors, inputs_c, preds, inputs = evaluate_reconstruction_error(config, data, mode='forecast',
                                                                                nv_fc=nv_fc, start_index=si, target_index=ti)
                print(config.prefix)

                results[key][0].append(errors)
                results[key][1].append([inputs, preds, inputs_c])

                #compute_vft_error(inputs, preds,mode='forecast', exp_ind=0)

                #save  forecasting predictions
                #if(nv_fc ==3):
                #    compare_prediction_images(preds, inputs, errors,
                #                          save_dir=os.path.join(config.LOG_ROOT_DIR, 'testdata', 'viz'))

                if(mm[1] == 1): errors_all.append(errors[0][1])
                nvisits_all.append(nv_fc)


                for ii, e in zip(inputs_c, errors):
                    print(ii, ["{0:0.2f}+-{1:0.2f}".format(np.mean(i), np.std(i)) if i is not None else None for i in e])
                print('RNFL comb index', rnfl_comb_index)
                if (mm[0] == 1):
                    [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error(inputs, preds,
                                                                                             rnfl_dia_mm=5.65, use_ma=True,
                                                                                             quadrant=None, comb_index=rnfl_comb_index)
                    errors_all.append(abs_err_gm)

                    print('Global [', err_gm, '+-', std_gm, ']')

                    temp_eq=[]
                    for q in [0, 1, 2, 3]:
                        [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error(inputs, preds,
                                                                                                 rnfl_dia_mm=5.5,
                                                                                                 use_ma=True,
                                                                                                 quadrant=q,
                                                                                                 disc_dia_mm=1.5, comb_index=rnfl_comb_index)
                        temp_eq.append(abs_err_gm)
                        print('Quad', q, '[', err_gm, '+-', std_gm, ']')
                    errors_quads.append(temp_eq)

            print('saving errors with means', [np.mean(e) for e in errors_all])
            save_error(model_name='lode', target_modality=target_modality, errors=errors_all, nv=nvisits_all,
                       exp_type=exp_type, save_dir='results')

            np.save(os.path.join(savedir, 'testdata_forecast.npy'), results[key])
            # e = np.asanyarray(results[key][0])
            # e_rnfl = list(np.clip(e[:, 0, 0], 0, 20))
            #
            # e_vft = list(np.clip(e[:, 0, 2], 0, 15)) # [:,0,2] using rnfl+vft and [:,2,2] only using vft
            #
            # e_rnfl_fromvft = list(np.clip(e[:, 2, 0], 0, 30))
            #
            # e_vft_fromrnfl = list(np.clip(e[:, 1, 2], 0, 15))
            #
            # ##imbp = boxplotv1(e_rnfl, e_vft, show=True)
            # ##cv2.imwrite(os.path.join(savedir, 'box_plot.jpeg'), imbp)
            # ##imlp = line_plot(np.asanyarray(e), show=True)
            # ##cv2.imwrite(os.path.join(savedir, 'line_plot.jpeg'), imlp)
            #
            # imbp = boxplot_vft(e_vft, title='VFT', show=True)
            # cv2.imwrite(os.path.join(savedir, 'box_plot_vft.jpeg'), imbp)
            #
            # imbp = boxplot_vft(e_vft_fromrnfl, title='VFT from RNFL', show=True)
            # cv2.imwrite(os.path.join(savedir, 'box_plot_vft_from_rnfl.jpeg'), imbp)
            #
            # imbp = boxplot_rnfl(e_rnfl, title='RNFL global mean', show=True)
            # cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl_global.jpeg'), imbp)
            #
            # imbp = boxplot_rnfl(e_rnfl_fromvft, title='RNFL from VFT', show=True)
            # cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl_from_vft.jpeg'), imbp)
            #
            #
            # #quadrants
            # errors_quads = np.asanyarray(errors_quads)
            # loc=['Superior', 'Inferior', 'Temporal','Nasal']
            # for i in range(errors_quads.shape[1]):
            #     imbp = boxplot_rnfl(list(errors_quads[:, i, :]), title ='RNFL '+loc[i], show=True)
            #     cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl'+loc[i]+'.jpeg'), imbp)

    # print('Done')
    # for k in results.keys():
    #   e = np.asanyarray(results[k] [0] )
    #    boxplot(e)
