from data.data_loader import get_data_ts, resize_stack
import torch
import numpy as np
#from dataflow import DataFlow, BatchData

from cutils.common import normalize_range




def normalize_range(X, source_range, target_range):
    xmin = source_range[0]
    xmax = source_range[1]
    ratio = (target_range[1] - target_range[0]) * 1.0 / (xmax - xmin)
    # print 'ratio ', ratio
    X = target_range[0] + ratio * (X - xmin)
    return X



##Compute the mask of RNFL region for calculating error.
def  compute_mask_base(gt_stack, threshold=50):
    """

    :param gt_stack: (N,t, c,H,W)
    :param threshold:
    :return: (c,H,W) mask
    """
    N, t, c, h, w = gt_stack.shape


    temp = gt_stack.reshape(N*t,c,h,w)
    mask = torch.mean(temp, dim=0) > threshold
    #mask = mask.repeat(N,t,1,1,1)

    return mask

# HHYU: commented this out to avoid loading DataFlow
# class RNFLData(DataFlow):

#     def __init__(self, x, t, dx, metadata_val):
#         super(RNFLData, self).__init__()

#         self.x =x
#         self.t =t
#         self.dx =dx
#         self.metadata_val = metadata_val

#     def __iter__(self):

#         for k in range(self.t.shape[0]):
#             data =  [ d[[k]] for d in self.x]
#             data.extend([ self.t[[k]], self.dx[[k]], self.metadata_val[[k]] ])

#             yield data #self.x[[k]], self.t[[k]], self.dx[[k]], self.metadata_val[[k]]

#     def size(self):
#         return  self.t.shape[0]


# class BatchDataPT(DataFlow):

#     def __init__(self, ds,batch_size,remainder):
#         super(BatchDataPT, self).__init__()
#         self.ds_batchdata = BatchData(ds,batch_size=batch_size, remainder=remainder, use_list=True)

#     def __iter__(self):
#         for dp in  self.ds_batchdata .get_data():
#             pt_dp =[torch.cat(d, dim=0) for d in dp]
#             yield  pt_dp


def filteridx_by_progression_rate(rnfls, times, slope_threshold=0, mask_base=None):
    """

    :param rnfls: (N,t,1,H,W) - unnormalized values
    :param times: (N,t), age at visit in years
    :return: same size as input except in batch dim
    """
    rnfls = rnfls.cpu()
    times = times.cpu()

    if(mask_base is None):
        mask_base = compute_mask_base(rnfls, threshold=50).float()

    N, t, c, h, w = rnfls.shape
    mask = mask_base.repeat(N, t, 1, 1, 1)

    # compute global mean
    flattened_im = rnfls.view(rnfls.shape[0], rnfls.shape[1], -1)  # (N,t,num_pixels)

    # compute masked global mean
    flattened_mask = mask.view(mask.shape[0], mask.shape[1], -1)  # (N,t,num_pixels)
    global_means_mask = torch.sum(flattened_im * flattened_mask.float(), dim=2) / torch.sum(flattened_mask, dim=2)
    from sklearn.linear_model import LinearRegression
    slopes = []
    for i in range(times.shape[0]):
        x = global_means_mask[[i], :].transpose(1, 0)
        t = times[[i], :].transpose(1, 0)
        reg = LinearRegression()
        reg.fit(t, x)
        slope  = reg.coef_
        slopes.append(slope)
    slopes = np.squeeze(np.concatenate(slopes, axis=0))
    idx = np.where(slopes <= slope_threshold)
    #filtered_rnfls  = global_means_mask[idx]

    return idx, mask_base


#todo  while generating npy file
def fix_missing_vf(vft):
    temp =[]
    for vf in vft:
        sum_seq = np.sum(vf.reshape(vf.shape[0], 32 * 32 * 1), axis=1)
        idx = np.where(sum_seq==0)[0]
        if(len(idx)==1): #fill the one remaining one
            id = idx[0]
            if(id==0):
                vf[id] = vf[1]
            else:
                vf[id] = vf[id-1]
        if(len(idx)==3):
            id_nomiss = list(set([0,1,2,3]) -set(idx))[0]
            for id in idx:
                vf[id] = vf[id_nomiss]

        if(len(idx)==2):
            pass
        if(0 in idx and 1 in idx):
            vf[0] = vf[2]
            vf[1] = vf[2]
        elif(2 in idx and 3 in idx):
            vf[2]=vf[1]
            vf[3] = vf[1]
        elif (0 in idx and 3 in idx):
            vf[0]= vf[1]
            vf[3]= vf[2]
        elif (1 in idx and 2 in idx):
            vf[1]= vf[0]
            vf[2]= vf[3]
        elif (1 in idx and 3 in idx):
            vf[1]= vf[0]
            vf[3]= vf[2]
        elif (0 in idx and 2 in idx):
            vf[0]= vf[1]
            vf[2]= vf[3]
        temp.append(vf)

    vft_fixed = np.asanyarray(temp)
    return vft_fixed
















def process(obsmaps, age_at_vd, gmetadata, image_dim, device):
    rnfls, gcls, vft, proj = obsmaps


    rnfls = resize_stack(rnfls, (image_dim, image_dim))  # (N,4,128,128,1)
    rnfls = rnfls.transpose([0, 1, 4, 2, 3])  # rnfls (N,4,1, 128,128)


    gcls = resize_stack(gcls, (image_dim, image_dim))  # (N,4,128,128,1)
    gcls = gcls.transpose([0, 1, 4, 2, 3])  # rnfls (N,4,1, 128,128)


    vft = resize_stack(vft, (32, 32))  # (N,4,128,128,1)
    vft = vft.transpose([0, 1, 4, 2, 3])  # rnfls (N,4,1, 128,128)
    vft = fix_missing_vf(vft)

    proj = resize_stack(proj, (image_dim, image_dim))  # (N,4,128,128,1)
    proj = proj.transpose([0, 1, 4, 2, 3])  # rnfls (N,4,1, 128,128)



    dx = gmetadata[0]  # (N,4)
    dx = torch.from_numpy(dx).float().to(device)

    tinc = np.asanyarray([[0.1, 0.2, 0.3, 0.4]])
    tinc = np.repeat(tinc, age_at_vd.shape[0], axis=0)
    age_at_vd = age_at_vd + tinc
    rnfls = torch.from_numpy(rnfls).float().to(device)
    age_at_vd = torch.from_numpy(age_at_vd).float().to(device)
    gcls = torch.from_numpy(gcls).float().to(device)
    vft = torch.from_numpy(vft).float().to(device)
    proj = torch.from_numpy(proj).float().to(device)



    obsmaps =[rnfls, gcls, vft, proj]
    return obsmaps, age_at_vd, dx



def filter_by_pr(data, mask_base=None):
    obs_maps = data[0]
    rnfls = obs_maps[0] * 200.0

    age_at_vd = data[1] /12.0
    idx,  mask_base = filteridx_by_progression_rate(rnfls, age_at_vd, slope_threshold=1.0, mask_base=mask_base)
    data_ = [ d[idx]for d in data[1:]]
    obs_maps=[d[idx] for d in obs_maps]


    rnfls = obs_maps[0] * 200.0
    v4 = torch.mean(rnfls[:, -1].view(rnfls.size(0), -1), dim=1)
    v3 = torch.mean(rnfls[:, -2].view(rnfls.size(0), -1), dim=1)
    idx1 = v4 - v3 <= 1

    data_ = [d[idx1] for d in data_]
    obs_maps = [d[idx1] for d in obs_maps]

    data_.insert(0,obs_maps)

    return data_, mask_base



def process_mdata(mdata, mdata_stats=None, device=None):
    mdata = torch.from_numpy(mdata).float().to(device)

    if(mdata_stats is None):
        N, td, d = mdata.shape
        mdata_temp = mdata.contiguous().view(N * td, d)
        missing_mask = mdata_temp != mdata_temp
        mdata_temp[missing_mask] = 0
        mdata_stats = [mdata_temp.min(dim=0)[0], mdata_temp.max(dim=0)[0]]

    missing_mask = mdata != mdata
    mdata = normalize_range(mdata, mdata_stats, [0,1])
    mdata[missing_mask] = 0



    missing_mask = (missing_mask).type(torch.float32)
    valid_mask = 1 - missing_mask
    mdata = torch.stack([mdata, valid_mask], dim=2)
    return mdata, mdata_stats



class ODEMAPDataLoader():

    def __init__(self, image_dim, device, filter_pr=True):

        self.image_dim= image_dim
        self.device= device

        obsmaps_train, age_at_vd_train, gmetadata_train, mdata_train = get_data_ts('train', mask_disc=True)
        obsmaps_test, age_at_vd_test, gmetadata_test, mdata_test = get_data_ts('test', mask_disc=True)

        obsmaps_train =   [ np.vstack(obsmap) for obsmap in zip(obsmaps_train, obsmaps_test)]

        age_at_vd_train = np.vstack([age_at_vd_train, age_at_vd_test])
        mdata_train = np.vstack([mdata_train, mdata_test])
        gmetadata_train = [ np.vstack([x,y]) for x,y in zip(gmetadata_train, gmetadata_test)]



        obsmaps_train, age_at_vd_train, dx_train = process(obsmaps_train, age_at_vd_train, gmetadata_train, self.image_dim, self.device)

        mdata_train, mdata_stats = process_mdata(mdata_train, device=device)

        (self.obsmaps_train, self.age_at_vd_train, self.dx_train, self.metadata_train), mask_base = filter_by_pr([obsmaps_train, age_at_vd_train, dx_train, mdata_train])




        rnfls_val, age_at_vd_val, gmetadata_val, mdata_val = get_data_ts('val', mask_disc=True)
        rnfls_val, age_at_vd_val, dx_val = process(rnfls_val, age_at_vd_val, gmetadata_val,self.image_dim, self.device)
        mdata_val, _ = process_mdata(mdata_val, mdata_stats=mdata_stats, device=device)

        (self.obsmaps_val, self.age_at_vd_val, self.dx_val, self.metadata_val), _ = filter_by_pr([rnfls_val, age_at_vd_val, dx_val, mdata_val], mask_base=mask_base)

        #to years
        self.age_at_vd_train = self.age_at_vd_train / 12.0
        self.age_at_vd_val = self.age_at_vd_val / 12.0

        # transform in [-1, 1]
        self.age_at_vd_train = normalize_range(self.age_at_vd_train, [20, 80], [-1, 1])
        self.age_at_vd_val   = normalize_range(self.age_at_vd_val, [20, 80], [-1, 1])


    def get_data_rnfl_map_train(self, batch_size=16):

        ids = torch.randperm(self.obsmaps_train[0].shape[0], device=self.device)
        batch_ids= ids[:batch_size]

        x = [obsmap[batch_ids] for obsmap in self.obsmaps_train]# self.obsmaps_train [batch_ids]
        t = self.age_at_vd_train [batch_ids]
        dxb =self.dx_train[batch_ids] #(batchsize,4)

        mdata = self.metadata_train[batch_ids]

        x = [x_.contiguous() for x_ in x]
        t = t.contiguous()

        x.extend ([t, dxb, mdata])
        return x

    def get_data_rnfl_map_val(self, BATCH_SIZE):
        """

        :param BATCH_SIZE: if <=0, returs a single batch of data size
        :return: Batch dataflow instance
        """
        #nval = (self.obsmaps_val.shape[0] // BATCH_SIZE) * BATCH_SIZE
        #obsmaps_val, age_at_vd_val, dx_val = self.obsmaps_val[:nval], self.age_at_vd_val[:nval], self.dx_val[:nval]

        rnfls_val = [x.contiguous() for x in self.obsmaps_val]

        ds_val = RNFLData(rnfls_val, self.age_at_vd_val, self.dx_val, self.metadata_val)
        if(BATCH_SIZE <=0):
            BATCH_SIZE = ds_val.size()

        val_ds_batch = BatchDataPT(ds_val, batch_size=BATCH_SIZE, remainder=True)
        return val_ds_batch





if(__name__=='__main__'):
    image_dim=128
    device='cpu'
    rnfldata = ODEMAPDataLoader(image_dim=image_dim, device=device)
    print('done')


