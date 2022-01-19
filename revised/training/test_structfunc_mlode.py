import os

import numpy as np
import pandas as pd

#from data.utils import mask_rnfl
from scripts.eval_mlode_sync import evaluate_struct_function, create_vft_mask, create_rnfl_mask
from train_multimodal_latentodegru_sync import getConfig, MultimodalTimeSeriesData
from scripts.datautils import MatchedTestSet



np.set_printoptions(precision=2)
import tempfile
import cv2
import torch

if (__name__ == '__main__'):

    # modalities_exp = [ [1,0,0],[0,0,1],[1, 0, 1]]

    rnfl_dia_mm =7
    modalities_exp = [[1, 1]] #[[1, 0, 1]]
    experts = ['poe']  # ,'poe']
    fold_seed = 5
    results = {}
    # for seed 2
    # idx_r = [ 107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0

    #idx_r = [ 84, 324, 162, 3]  # filter for [1,0] input ie rnfl comb index=2
    #rnfl_comb_Index 0 - inputs 11 and 2 means 1,0 and 1 means 01 the input comb in format [rnfl, vft]
    #rnfl_comb_index=2

    #data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
    data= MatchedTestSet(seed = fold_seed)
    print('Number of test samples', data.val_rnflonh.shape[0])
    #suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path=''

    df_estimation = pd.DataFrame()
    df_forecast = pd.DataFrame()
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)
            inputs, ts_list_val, dx = data.get_val()
            rnfl_mask = create_rnfl_mask(inputs[0],rnfl_diam_mm=rnfl_dia_mm)
            vft_mask = create_vft_mask(inputs[1])
            masks =[rnfl_mask, vft_mask]
            for nv in reversed(list(range(1, 6))):
                print('NV RNFL', nv)
                errors = evaluate_struct_function(config.model,ts_list_val,inputs, masks, nv_fc_rnfl=nv)
                e_estimation_ = errors[-2]
                e_estimation = e_estimation_.cpu().detach().numpy()
                print(e_estimation.shape)

                e_forecast_ = errors[-1]
                e_forecast = e_forecast_.cpu().detach().numpy()

                df_estimation[nv] = e_estimation
                df_forecast[nv] = e_forecast

                print("estimation error: ", e_estimation.mean(), "\t", e_estimation.std())
                print("forecast error: ", e_forecast.mean(), "\t", e_forecast.std())

                out_dir = "results_" + str(fold_seed) + '/'
                df_forecast.to_csv(out_dir + 'strucfunc_forecast.csv')
                df_estimation.to_csv(out_dir + 'strucfunc_estimation.csv')                
                
                
                #for e_ in errors:
                #    e = e_.cpu().detach().numpy()
                #    print(e.mean(), e.std())
            #print('Error Struct func', torch.mean(err))


