from .eval_mlode import evaluate_forecast_error, create_vft_mask, create_rnfl_mask
from utils.utils import select_modalities, subsample
import torch
import numpy as np
class Evaluation:

    def __init__(self, config):
        self.model = config.model.eval()

        comb = [[1, 1, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.comb = select_modalities(comb, config.MODALITIES)



    def compute_errors(self, ts, inputs, masks, modalities, nv_fc):
        return  evaluate_forecast_error(self.model, ts, inputs,masks, modalities, nv_fc)


    def compute_errors_all(self, data, nv_fc):
        with torch.no_grad():
            ds_val_batches, ts_val, val_dx = data.get_val(dx_filter=None)
            rnfl_xi_val = ds_val_batches[0]
            #gcl_xi_val = ds_val_batches[1]
            vft_xi_val = ds_val_batches[2]

            vft_mask = create_vft_mask(vft_xi_val)
            rnfl_mask = create_rnfl_mask(rnfl_xi_val)

            masks = [rnfl_mask, None, vft_mask]
            errors = [];
            inputs_modalities = []
            preds_all = []
            inputs_all = []
            # if (config.MODALITIES[1] == 1):  # if RNFL is used in training
            for c in self.comb:
                # x_list_c = subsample(ds_val_batches, c)
                x_list = ds_val_batches


                error, preds, inputs = evaluate_forecast_error(self.model, ts_val, x_list, masks, c, nv_fc)

                error = [e.cpu().numpy() for e in error]
                error = [e.astype(np.float32) for e in error]
                # error = subsample(error, config.MODALITIES)

                preds = subsample(preds, self.config.MODALITIES)
                preds_all.append(preds)
                inputs_modalities.append(c)
                errors.append(error)
                inputs_all.append(inputs)

        return errors, inputs_modalities, preds_all, inputs_all








