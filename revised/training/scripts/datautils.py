import torch
from cutils.common import denormalize_range
from train_multimodal_latentodegru_sync import MultimodalTimeSeriesData

import os
import pandas as pd

class MatchedTestSet():

    def __init__(self, seed=2):
        datas2v4 = MultimodalTimeSeriesData(fold_seed=seed)  # 4 visits

        datas2v6 = MultimodalTimeSeriesData(fold_seed=seed, suffix='6')  # 6 visits
        sid4val = datas2v4.get_val_subject_ids()

        sid6train = datas2v6.get_train_subject_ids()
        sid6val = datas2v6.get_val_subject_ids()
        idx_train_matched = [idx for idx, sid in enumerate(sid6train) if sid in sid4val]
        idx_val_matched = [idx for idx, sid in enumerate(sid6val) if sid in sid4val]

        filter = lambda x_trn, x_val: torch.cat([x_trn[idx_train_matched], x_val[idx_val_matched]], dim=0)

        self.val_rnflonh = filter(datas2v6.train_rnflonh, datas2v6.val_rnflonh)
        self.val_vft = filter(datas2v6.train_vft, datas2v6.val_vft)
        self.age_at_vd_val = filter(datas2v6.age_at_vd_train, datas2v6.age_at_vd_val)
        self.val_dx = filter(datas2v6.train_dx, datas2v6.val_dx)

        # HHYU: added to export subjects
        self.val_sid = filter(datas2v6.train_sid, datas2v6.val_sid)
        self.age_at_vd_val_0 = denormalize_range(self.age_at_vd_val, [20, 80], [-1, 1])
        #print(self.val_rnflonh.shape)
        #print(self.val_vft.shape)
        #print(self.val_sid.shape)
        #print(self.age_at_vd_val.shape)
        #print(self.age_at_vd_val_0.shape)

    def get_val(self, dx_filter=None):
        """

        :param dx_filter:
        :return: maps each os size (N, t,c,H,W) and dx of size (N,)
        """
        maps = [self.val_rnflonh,  self.val_vft]

        # reduce from N,t to N,1 ie one diagnosis for time sample

        ts = self.age_at_vd_val

        dx = torch.max(self.val_dx, dim=1)[0]  # (N,1)
        if (dx_filter):
            maps = [m[dx == dx_filter] for m in maps]
            dx = dx[dx == dx_filter]
            ts = ts[dx == dx_filter]

        ts_list = [ts, ts.clone().detach()]

        return maps, ts_list, dx



def save_error(model_name, target_modality, errors, nv, exp_type, save_dir='results'):
    """

    :param model_name:
    :param target_modality:
    :param errors: list of errors
    :param nv number of visits
    :exp_type one of num_visits or larger_gap
    :return:
    """
    assert exp_type in ['num_visits', 'larger_gap'], 'Invialid param for exp_type'

    nv =[ str(i)+'_'+str(v) for i, v in enumerate(nv)]
    if (not os.path.exists(save_dir)):
        os.mkdir(save_dir)
    df = pd.DataFrame(dict(zip(nv, errors)))
    filename = '_'.join([model_name, target_modality, exp_type]) + '.csv'
    out_filename = os.path.join(save_dir, filename)
    df.to_csv(out_filename)



if __name__=='__main__':
    ms = MatchedTestSet()
