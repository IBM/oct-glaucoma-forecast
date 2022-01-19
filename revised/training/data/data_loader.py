import random

import cv2
import numpy as np
from utils.utils import Splitrandom

def create_data_loader_train(batch_size=8):
    data = np.load('../temp/lstmdata_train.npz')
    rnfls = data['rnfl']
    rnfls = np.clip(rnfls, 0, 255)
    rnfls = rnfls / 255.0

    n = len(rnfls)
    idx = np.arange(n)

    while True:
        random.shuffle(idx)
        batch = rnfls[idx[:batch_size]]
        x = batch[:, :3, :, :, :]
        y = batch[:, 3:, :, :, :]
        y = np.squeeze(y, axis=1)
        yield x, y


def get_data(type, use_deltat=False, mask_disc=False):
    if (type == 'train'):
        data = np.load('../temp/lstmdata_train.npz')
    elif (type == 'val'):
        data = np.load('../temp/lstmdata_val.npz')

    elif (type == 'test'):
        data = np.load('../temp/lstmdata_test.npz')
    else:
        ValueError('Invalid type', type)

    rnfls = data['rnfl']
    if (mask_disc): rnfls = mask_disc_area(rnfls)

    rnfls = np.clip(rnfls, 0, 200)
    rnfls = rnfls / 200.0

    x = rnfls[:, :3, :, :, :]
    y = rnfls[:, 3:, :, :, :]
    y = np.squeeze(y, axis=1)

    if (use_deltat):
        vd = data['ageat_visit_dates_months']
        vd = np.diff(vd, axis=1)
        vd = np.expand_dims(vd, -1)
        vd = np.expand_dims(vd, -1)
        vd = np.repeat(vd, rnfls.shape[2], axis=2)
        vd = np.repeat(vd, rnfls.shape[3], axis=3)
        vd = np.expand_dims(vd, -1)
        vd = vd / 24.0  # assume max is 2 years. there may be samples >2 years interval but it is ok as long as loss is mse
        x = np.concatenate([x, vd], axis=4)

    return x, y


def merge_class_dx(dx):
    if (dx == 'POAG'):
        return 1
    else:
        return 0


def get_data_ts(type, mask_disc=False, filter_missing_vft=False):
    """
    get data as tme series i.e no input output
    :param type:
    :param use_deltat:
    :return:
    """
    pickle = True
    if (type == 'train'):
        data = np.load('../oct_forecasting/temp/lstmdata_train.npz', allow_pickle=pickle)
    elif (type == 'val'):
        data = np.load('../oct_forecasting/temp/lstmdata_val.npz', allow_pickle=pickle)

    elif (type == 'test'):
        data = np.load('../oct_forecasting/temp/lstmdata_test.npz', allow_pickle=pickle)
    else:
        ValueError('Invalid type', type)

    rnfls = data['rnfl']
    gcls = data['gcl']
    vft = data['vfimage']
    proj = data['proj']

    # gcls = np.expand_dims(gcls, -1)
    vft = np.expand_dims(vft, -1)
    # proj = np.expand_dims(proj, -1)
    if (mask_disc): rnfls = mask_disc_area(rnfls)
    # if( mask_disc) : gcls = mask_disc_area(gcls)

    rnfls_ = np.clip(rnfls, 0, 200.0)
    rnfls = rnfls_ / 200.0

    gcls_ = rnfls_ + gcls
    gcls = np.clip(gcls_, 0, 255.0)
    gcls = gcls / 255.0

    vft = np.clip(vft, 0, 40.0)
    vft = vft / 40.0

    proj = proj / 255.0

    vd = data['ageat_visit_dates_months']

    gmdata = data['global_metadata']
    metadata = data['metadata']

    # filter time series sample where all the vft images are missings
    if (filter_missing_vft):
        aa = np.sum(vft.reshape((vft.shape[0], -1)), axis=1)
        idx = np.where(aa > 0)[0]
        datalist = [rnfls, gcls, vft, proj, vd, gmdata, metadata]
        rnfls, gcls, vft, proj, vd, gmdata, metadata = [d[idx] for d in datalist]

    dx = [x[0] for x in gmdata[:, 1]]
    dx_int = [merge_class_dx(dxi) for dxi in dx]
    dx_int = np.expand_dims(dx_int, -1)
    dx_int = np.tile(dx_int, rnfls.shape[1])

    return [rnfls, gcls, vft, proj], vd, [dx_int], metadata


def resize_stack(stack, newsize):
    """

    :param stack: (N, nt, H,W,1)
    :return:
    """
    stack = np.squeeze(stack)  # (N, nt, H,W)
    stack = stack.transpose([0, 2, 3, 1])  ##(N,  H,W, nt)
    images_new = []
    for im in stack:
        imnew = cv2.resize(im, newsize, interpolation=cv2.INTER_LINEAR)
        images_new.append(imnew)
    images_new = np.asanyarray(images_new)  # (N,  H,W, nt)
    images_new = images_new.transpose([0, 3, 1, 2])
    images_new = np.expand_dims(images_new, -1)
    return images_new


def resize_stack(stack, newsize):
    """

    :param stack: (N, nt, H,W,1)
    :return:
    """
    stack = np.squeeze(stack)  # (N, nt, H,W)
    stack = stack.transpose([0, 2, 3, 1])  ##(N,  H,W, nt)
    images_new = []
    for im in stack:
        imnew = cv2.resize(im, newsize, interpolation=cv2.INTER_LINEAR)
        images_new.append(imnew)
    images_new = np.asanyarray(images_new)  # (N,  H,W, nt)
    images_new = images_new.transpose([0, 3, 1, 2])
    images_new = np.expand_dims(images_new, -1)
    return images_new


def resize_stack_nonseries(stack, newsize):
    """
    :param stack: (N,  H,W,1)
    :return:
    """
    stack = np.squeeze(stack)  # (N,  H,W)
    images_new = []
    for im in stack:
        imnew = cv2.resize(im, newsize, interpolation=cv2.INTER_LINEAR)
        images_new.append(imnew)
    images_new = np.asanyarray(images_new)  # (N,  H,W)

    images_new = np.expand_dims(images_new, -1)  # (N,  H,W,1)
    return images_new


def mask_disc_area(stack):
    """

    :param stack: (N, nt, H,W,1) [pixel range [0,255]
    :return:
    """
    (N, nt, H, W, ch) = stack.shape
    assert H == W and ch == 1, ' invalid image format'

    disc_dia = H * 1.92 / 6.0
    disc_radius = int(disc_dia / 2.0)
    cx, cy = int(W / 2), int(H / 2)

    disc_synt = cv2.circle(np.zeros((H, W)), (cx, cy), disc_radius, 255, -1).astype(np.bool)
    disc_synt = np.expand_dims(disc_synt, -1)  # (H,W,1)
    stack = stack * ~disc_synt
    return stack


def get_data_ts_onh_mac_(mask_onh_rnfl_disc, filter_missing_vft=True, suffix=''):
    """

    :param mask_onh_rnfl_disc:
    :param filter_missing_vft:
    :param suffix: used when number of visits is specified
    :return:
    """
    pickle = True
    #data_dir = "/dccstor/aurmmaret1/Datasets/suman_data_2021_orig/"
    data_dir = "/dccstor/aurmmaret1/Datasets/suman_data_2021/"    
    print("loading from", data_dir)
    data = np.load(data_dir + 'onh_mac_vft_data'+suffix+'.npz', allow_pickle=pickle)

    rnfls_onh = data['rnfls_onh']
    gcls_onh = data['gcls_onh']
    metadata_onh = data['metadata_onh']

    rnfls_mac = data['rnfls_mac']
    gcls_mac = data['gcls_mac']
    metadata_mac = data['metadata_mac']

    vft = data['vft']
    ismissing_vft = data['ismissing_vft']

    subject_id = data['subject_id']
    eye = data['eye']

    # add extra dimension at the end
    maps = [rnfls_onh, gcls_onh, rnfls_mac, gcls_mac, vft]
    maps = [np.expand_dims(m, -1) for m in maps]
    rnfls_onh, gcls_onh, rnfls_mac, gcls_mac, vft = maps

    # vft = np.expand_dims(vft, -1)

    if (mask_onh_rnfl_disc): rnfls_onh = mask_disc_area(rnfls_onh)
    # if( mask_disc) : gcls = mask_disc_area(gcls)

    rnfls_onh = np.clip(rnfls_onh, 0, 200.0)
    rnfls_onh = rnfls_onh / 200.0

    gcls_onh = rnfls_onh + gcls_onh
    gcls_onh = np.clip(gcls_onh, 0, 255.0)
    gcls_onh = gcls_onh / 255.0

    vft = np.clip(vft, 0, 40.0)
    vft = vft / 40.0

    rnfls_mac = np.clip(rnfls_mac, 0, 200.0)
    rnfls_mac = rnfls_mac / 200.0

    gcls_mac = np.clip(gcls_mac, 0, 200.0)
    gcls_mac = gcls_mac / 200.0

    vd = data['age_at_visit_date_months']

    if (filter_missing_vft):
        aa = np.sum(vft.reshape((vft.shape[0], -1)), axis=1)
        idx = np.where(aa > 0)[0]
        datalist = [rnfls_onh, gcls_onh, rnfls_mac, gcls_mac, vft, vd, subject_id, metadata_onh, metadata_mac]
        rnfls_onh, gcls_onh, rnfls_mac, gcls_mac, vft, vd, subject_id, metadata_onh, metadata_mac = [d[idx] for d in
                                                                                                     datalist]

    dx = metadata_onh[:, :, -1]
    dx_int = dx.copy()
    poag = dx_int == 'POAG'
    suspects = dx_int =='GS'
    normal = dx_int =='Normal'
    dx_int[poag] = 2
    dx_int[suspects] = 1
    dx_int[normal] = 0

    dx_int = dx_int.astype(np.uint8)

    return [rnfls_onh, gcls_onh, rnfls_mac, gcls_mac, vft], vd, [dx_int, subject_id], metadata_onh


def get_data_ts_onh_mac(mask_onhrnfl_disc=True, filter_missing_vft=True, fold_seed=4, suffix=''):
    """
    Generates the synced onh and macular thicness maps + vft in time series format and splits into three groups
    :return:
    """
    print("HHYU get_data_ts_onh_mac: fold_seed=", fold_seed)
    maps, vd, [dx_int, subject_id], metadata_onh = get_data_ts_onh_mac_(mask_onhrnfl_disc, filter_missing_vft=filter_missing_vft, suffix=suffix)
    subject_id1 = [sid[0] for sid in subject_id]
    idx = list(range(len(subject_id1)))


    idx_subs = list(zip(idx, subject_id1))

    Splitter = Splitrandom((0.7, 0.1, 0.2), seed=fold_seed, group_func=lambda x: x[1])
    train, val, test = Splitter(idx_subs)
    idx_train = [id for id, sub in train]
    idx_val = [id for id, sub in val]
    idx_test = [id for id, sub in test]

    def apply_splits(data):
        if (type(data) is list):
            data_train = [m[idx_train] for m in data]
            data_val = [m[idx_val] for m in data]
            data_test = [m[idx_test] for m in data]
        else:
            data_train = data[idx_train]
            data_val = data[idx_val]
            data_test = data[idx_test]

        return data_train, data_val, data_test

    maps_train, maps_val, maps_test = apply_splits(maps)
    mdata_train, mdata_val, mdata_test = apply_splits(metadata_onh)
    dxint_train, dxint_val, dxint_test = apply_splits(dx_int)
    subject_id_train, subject_id_val, subject_id_test = apply_splits(subject_id)
    #ismissing_vft_train, ismissing_vft_val, ismissing_vft_test = apply_splits(ismissing_vft)

    vd_train, vd_val, vd_test = apply_splits(vd)

    return [maps_train, vd_train, [dxint_train, subject_id_train],
            mdata_train], \
           [maps_val, vd_val, [dxint_val, subject_id_val], mdata_val], \
           [maps_test, vd_test, [dxint_test, subject_id_test], mdata_test]


if (__name__ == '__main__'):
    # x, t, dx_list, mdata = get_data_ts('val')
    # print('resized', x[0].shape)

    train, val, test = get_data_ts_onh_mac()
    # print('resized', x[0].shape)
