import os

import numpy as np

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
    fold_seed = 2
    results = {}
    # for seed 2
    idx_r = [ 107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0

    #idx_r = [ 84, 324, 162, 3]  # filter for [1,0] input ie rnfl comb index=2
    #rnfl_comb_Index 0 - inputs 11 and 2 means 1,0 and 1 means 01 the input comb in format [rnfl, vft]
    #rnfl_comb_index=2

    #data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
    data= MatchedTestSet()
    print('Number of test samples', data.val_rnflonh.shape[0])
    #suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path=''
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)
            inputs, ts_list_val, dx = data.get_val()
            rnfl_mask = create_rnfl_mask(inputs[0],rnfl_diam_mm=rnfl_dia_mm)
            vft_mask = create_vft_mask(inputs[1])
            masks =[rnfl_mask, vft_mask]
            for nv in reversed(list(range(0,6))):
                print('NV RNFL', nv)
                errors = evaluate_struct_function(config.model,ts_list_val,inputs, masks, nv_fc_rnfl=nv)
                for e in errors:
                    print(torch.mean(e), torch.std(e))
            #print('Error Struct func', torch.mean(err))


