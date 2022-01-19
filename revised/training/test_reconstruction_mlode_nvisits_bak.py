import os

import numpy as np
from data.utils import mask_rnfl
from scripts.datautils import MatchedTestSet, save_error
from scripts.eval_mlode_sync import evaluate_reconstruction_error, create_vft_mask
from train_multimodal_latentodegru_sync import getConfig
from utils.oct_utils import get_quardrants
import torch


np.set_printoptions(precision=2)
import tempfile
import cv2


def boxplot(e, show=True):
    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(121)
    ax.set_title('RNFL')

    data_to_plot = list(np.clip(e[:, 0, 0], 0, 15))  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot)

    # HHYU: disabled
    #ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE micron')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax = fig.add_subplot(122)
    ax.set_title('VFT')

    data_to_plot = list(np.clip(e[:, 0, 2], 0, 9))  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot)

    # HHYu: disabled
    #ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE DB')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def boxplotv1(e_rnfl, e_vft, show=True):
    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(121)
    ax.set_title('RNFL')

    data_to_plot = e_rnfl  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE micron')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax = fig.add_subplot(122)
    ax.set_title('VFT')

    data_to_plot = e_vft  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot)
    ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE DB')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def boxplot_rnfl(e_rnfl, show=True, title='RNFL', showmeans=True):
    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    data_to_plot = e_rnfl  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot, showmeans=showmeans)
    plt.ylim([0, 20])

    major_ticks = np.arange(0, 20, 5)
    minor_ticks = np.arange(0, 20, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(b=True, which='both')

    # HHYU disbaled
    #ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE micron')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot_rnfl.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def boxplot_vft(e_vft, show=True, title='VFT', showmeans=True):
    import matplotlib.pyplot as plt


    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    data_to_plot = e_vft  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot, showmeans=showmeans)
    plt.ylim([0, 10])

    major_ticks = np.arange(0, 10, 5)
    minor_ticks = np.arange(0, 10, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(b=True, which='both')

    # HHYU disabled
    #ax.set_xticklabels(['#visits3', '#visits2', '#nvisits1'])
    ax.set_ylabel('MAE DB')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot_vft.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def line_plot(e, show=True):
    import matplotlib.pyplot as plt
    fig = plt.figure(2, figsize=(9, 6))
    plt.subplot(1, 2, 1)
    plt.title('RNFL')
    plt.ylim([2, 5])
    plt.plot([3, 2, 1], np.mean(e[:, 0, 0], axis=1), 'g')  # RNFL for [1,0,1]
    plt.plot([3, 2, 1], np.mean(e[:, 1, 0], axis=1), 'b')  # #RNFL for [1,0,0]
    plt.plot([3, 2, 1], np.mean(e[:, 2, 0], axis=1), 'r')  # #RNFL for [0,0,1]
    plt.xlabel('#nvisits')
    plt.ylabel('MAE micron')

    plt.subplot(1, 2, 2)
    plt.title('VFT')
    plt.ylim([2.5, 4.5])
    plt.plot([3, 2, 1], np.mean(e[:, 0, 2], axis=1), 'g')  # VFT for [1,0,1]
    plt.plot([3, 2, 1], np.mean(e[:, 2, 2], axis=1), 'b')  # VFT for [0,0,1]
    plt.plot([3, 2, 1], np.mean(e[:, 1, 2], axis=1), 'r')  # VFT for [1,0,0]
    plt.xlabel('#nvisits')
    plt.ylabel('MAE DB')

    temp_file = os.path.join(tempfile.gettempdir(), 'line_plot.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def compute_rnfl_error_base(rnfl, rnfl_pred, rnfl_dia_mm=6, disc_dia_mm=1.92, use_ma=False, quadrant=None,
                            mode='forecast'):
    """

    :param rnfl: (N, T,1,H,W)
    :param rnfl_pred: (N, T,1,H,W)
    :param rnfl_dia_mm:
    :param disc_dia_mm:
    :param use_ma:
    :param quadrant:
    :param mode:
    :return:
    """

    rnfl_masked, mask = mask_rnfl(rnfl, channel_last=False, rnfl_diam_mm=rnfl_dia_mm)
    rnfl_pred_masked, _ = mask_rnfl(rnfl_pred, channel_last=False, rnfl_diam_mm=rnfl_dia_mm)

    if (quadrant is not None):
        qm = get_quardrants(rnfl_diam_mm=rnfl_dia_mm, image_size_return=rnfl.shape[3], disc_dia_mm=disc_dia_mm)
        mask = qm[quadrant]
        mask = np.expand_dims(mask, 0).astype(np.bool)

    if (use_ma):
        mask = mask[np.newaxis, np.newaxis, :, :, :].repeat(rnfl.shape[0], axis=0).repeat(rnfl.shape[1], axis=1)
        rnfl_masked = np.ma.masked_array(rnfl, mask=~mask)
        rnfl_pred_masked = np.ma.masked_array(rnfl_pred, mask=~mask)

    if (mode == 'forecast'):
        rnfl_masked = rnfl_masked[:, [-1]]
        rnfl_pred_masked = rnfl_pred_masked[:, [-1]]

    # MAE error
    errors = np.mean(np.abs(rnfl_masked - rnfl_pred_masked).reshape(rnfl_masked.shape[0], -1), axis=1) * 200
    err, std = np.mean(errors), np.std(errors)

    # global_mean
    errors_gm = np.abs(np.mean(rnfl_masked.reshape(rnfl_masked.shape[0], -1), axis=1) - np.mean(
        rnfl_pred_masked.reshape(rnfl.shape[0], -1), axis=1)) * 200
    err_gm, std_gm = np.mean(errors_gm), np.std(errors_gm)

    return [err, std], [err_gm, std_gm], [errors, errors_gm]


def compute_rnfl_error(inputs, preds, rnfl_dia_mm=6, disc_dia_mm=1.92, use_ma=False, quadrant=None, mode='forecast',
                       comb_index=0):
    """
    Computes RNFL error- MAE, gobal mean and
    :param inputs:
    :param preds:
    :param rnfl_dia_mm:
    :param disc_dia_mm:
    :param use_ma:
    :param quadrant:
    :param mode:
    :comb_index inputs or preds[comb_index] [0], length of preds[0] is 3  ie for 11, 01 annd 10 inputs
    :return:
    """
    rnfl = inputs[comb_index][0].detach().cpu().numpy()
    rnfl_pred = preds[comb_index][0].detach().cpu().numpy()

    rnfl_masked, mask = mask_rnfl(rnfl, channel_last=False, rnfl_diam_mm=rnfl_dia_mm)
    rnfl_pred_masked, _ = mask_rnfl(rnfl_pred, channel_last=False, rnfl_diam_mm=rnfl_dia_mm)

    if (quadrant is not None):
        qm = get_quardrants(rnfl_diam_mm=rnfl_dia_mm, image_size_return=rnfl.shape[3], disc_dia_mm=disc_dia_mm)
        mask = qm[quadrant]
        mask = np.expand_dims(mask, 0).astype(np.bool)

    if (use_ma):
        mask = mask[np.newaxis, np.newaxis, :, :, :].repeat(rnfl.shape[0], axis=0).repeat(rnfl.shape[1], axis=1)
        rnfl_masked = np.ma.masked_array(rnfl, mask=~mask)
        rnfl_pred_masked = np.ma.masked_array(rnfl_pred, mask=~mask)

    if (mode == 'forecast'):
        rnfl_masked = rnfl_masked[:, [-1]]
        rnfl_pred_masked = rnfl_pred_masked[:, [-1]]

    # MAE error
    errors = np.mean(np.abs(rnfl_masked - rnfl_pred_masked).reshape(rnfl_masked.shape[0], -1), axis=1) * 200
    err, std = np.mean(errors), np.std(errors)

    # global_mean
    errors_gm = np.abs(np.mean(rnfl_masked.reshape(rnfl_masked.shape[0], -1), axis=1) - np.mean(
        rnfl_pred_masked.reshape(rnfl.shape[0], -1), axis=1)) * 200
    err_gm, std_gm = np.mean(errors_gm), np.std(errors_gm)

    return [err, std], [err_gm, std_gm], [errors, errors_gm]


def compute_vft_error_base(vft, vft_pred, mask, mode='forecast'):
    assert mode in ['forecast', 'rec'], 'Mode shold be one of forecast or rec'
    # mask = mask[np.newaxis, np.newaxis, :, :, :].repeat(vft.shape[0], axis=0).repeat(vft.shape[1], axis=1)
    vft_masked = np.ma.masked_array(vft, mask=~mask)
    vft_pred_masked = np.ma.masked_array(vft_pred, mask=~mask)

    if (mode == 'forecast'):
        vft_masked = vft_masked[:, [-1]]
        vft_pred_masked = vft_pred_masked[:, [-1]]
    else:
        vft_masked = vft_masked[:, :-1]
        vft_pred_masked = vft_pred_masked[:, :-1]

        vft_masked = vft_masked.reshape((vft_masked.shape[0] * vft_masked.shape[1], -1))
        vft_pred_masked = vft_pred_masked.reshape((vft_pred_masked.shape[0] * vft_pred_masked.shape[1], -1))

    # MAE error
    errors = np.mean(np.abs(vft_masked - vft_pred_masked).reshape(vft_masked.shape[0], -1), axis=1) * 40
    err, std = np.mean(errors), np.std(errors)

    return [err, std], errors


def compute_vft_error(inputs, preds, mode='forecast', exp_ind=0):
    """

    :param inputs:
    :param preds:
    :param mode:
    :param exp_ind:  represents different combination of inputs 0 - vft,rnfl, 1- rnfl only, 2 vft only
    :return:
    """

    vft = inputs[exp_ind][2]
    vft_pred = preds[exp_ind][2]
    mask = create_vft_mask(vft)

    vft = vft.cpu().numpy()
    vft_pred = vft_pred.cpu().numpy()
    mask = mask.cpu().numpy()
    mask = mask.astype(np.bool)

    # mask = mask[np.newaxis, np.newaxis, :, :, :].repeat(vft.shape[0], axis=0).repeat(vft.shape[1], axis=1)
    vft_masked = np.ma.masked_array(vft, mask=~mask)
    vft_pred_masked = np.ma.masked_array(vft_pred, mask=~mask)

    if (mode == 'forecast'):
        vft_masked = vft_masked[:, [-1]]
        vft_pred_masked = vft_pred_masked[:, [-1]]
    else:
        vft_masked = vft_masked[:, :-1]
        vft_pred_masked = vft_pred_masked[:, :-1]

        vft_masked = vft_masked.reshape((vft_masked.shape[0] * vft_masked.shape[1], -1))
        vft_pred_masked = vft_pred_masked.reshape((vft_pred_masked.shape[0] * vft_pred_masked.shape[1], -1))

    # MAE error
    errors = np.mean(np.abs(vft_masked - vft_pred_masked).reshape(vft_masked.shape[0], -1), axis=1) * 40
    err, std = np.mean(errors), np.std(errors)

    return [err, std], errors


def compare_prediction_images(preds, inputs, errors, save_dir):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    if (not os.path.exists(save_dir)): os.mkdir(save_dir)
    import matplotlib.pyplot as plt
    gt_vft = inputs[0][1]
    pred_vft = preds[0][1]
    err_vft = errors[0][1]

    gt_rnfl = inputs[0][0]
    pred_rnfl = preds[0][0]
    err_rnfl = errors[0][0]

    # indices 66, 5, 70
    indices = range(gt_rnfl.shape[0])
    for idx in indices:
        # idx = np.argmin(err_vft)

        # vft
        plt.subplot(221)
        showvft = lambda x: plt.imshow(x[2:-2, 2:-2], cmap='gray')
        pos = showvft(gt_vft[idx, -1, 0] * 40)
        plt.colorbar(pos)
        plt.clim(0, 40)
        plt.title('ground truth')
        plt.axis('off')
        plt.subplot(222)
        pos = showvft(pred_vft[idx, -1, 0] * 40)
        plt.colorbar(pos)
        plt.clim(0, 40)
        plt.title('forecast')
        plt.axis('off')

        # rnfl
        plt.subplot(223)
        pos = plt.imshow(gt_rnfl[idx, -1, 0] * 200, cmap='jet')
        plt.colorbar(pos)
        plt.clim(0, 200)
        #plt.title('ground truth RNFL-TM')
        plt.axis('off')
        plt.subplot(224)
        pos = plt.imshow(pred_rnfl[idx, -1, 0] * 200, cmap='jet')
        plt.colorbar(pos)
        plt.clim(0, 200)
        #plt.title('forecast RNFL-TM')
        plt.axis('off')
        # plt.show()
        plt.suptitle('R{:.2f}, V{:.2f}'.format(err_rnfl[idx], err_vft[idx]))
        plt.savefig(os.path.join(save_dir, 'results' + str(idx)) + '.jpeg')
        plt.close()


def compare_prediction_imagesV1(preds, inputs, errors, save_dir):
    import matplotlib
    matplotlib.rcParams.update({'font.size': 14})
    if (not os.path.exists(save_dir)): os.mkdir(save_dir)
    import matplotlib.pyplot as plt
    gt_vft = inputs[0][1]
    pred_vft = preds[0][1]

    gt_vft = gt_vft[:, :, :, 2:-2, 2:-2]
    pred_vft = pred_vft[:, :, :, 2:-2, 2:-2]

    err_vft = errors[0][1]

    gt_rnfl = inputs[0][0]
    pred_rnfl = preds[0][0]
    err_rnfl = errors[0][0]

    def resize(x):
        out = torch.nn.functional.interpolate(x, size=(32, 32), mode='bicubic', align_corners=False)
        return out

    def stack_tensor(x, pred, pad=0):
        """
        :param x: (T,c, H,W) tensor
        :param pred: (T,c, H,W) tensor
        :return:  # c, H, T*W' image where W' is effective width after padding
        """

        x = torch.cat([x, pred[ [-1], :, :, :]], dim=0)
        padder = torch.nn.ZeroPad2d((1,1,0,0))
        x=padder(x)
        #x = resize(x)
        out=torch.cat(list(x), dim=2)

        return out



    # indices 66, 5, 70
    indices = range(gt_rnfl.shape[0])
    for idx in indices:
        # idx = np.argmin(err_vft)

        # vft
        out_vft = stack_tensor(gt_vft[idx], pred_vft[idx])
        out_rnfl = stack_tensor(gt_rnfl[idx], pred_rnfl[idx])
        plt.subplot(2,1,1)
        pos= plt.imshow(out_vft[0]* 40, cmap='gray')
        plt.colorbar(pos)
        plt.clim(0, 40)
        plt.axis('off')

        plt.subplot(2, 1, 2)
        pos = plt.imshow(out_rnfl[0] * 200, cmap='jet')
        plt.colorbar(pos)
        plt.clim(0, 200)
        plt.axis('off')
        plt.suptitle('R{:.2f}, V{:.2f}'.format(err_rnfl[idx], err_vft[idx]))

        plt.savefig(os.path.join(save_dir, 'results_traj' + str(idx)) + '.jpeg')
        plt.close()




if (__name__ == '__main__'):

    # modalities_exp = [ [1,0,0],[0,0,1],[1, 0, 1]]

    modalities_exp = [[1, 1]]  # [[1, 0, 1]]
    experts = ['poe']  # ,'poe']
    fold_seed = 2
    results = {}
    # for seed 2
    idx_r = [107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0

    # idx_r = [ 84, 324, 162, 3]  # filter for [1,0] input ie rnfl comb index=2
    # rnfl_comb_Index 0 - inputs 11 and 2 means 1,0 and 1 means 01 the input comb in format [rnfl, vft]
    #rnfl_comb_index = 2

    # data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
    data = MatchedTestSet()
    print('Number of test samples', data.val_rnflonh.shape[0])
    # suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path = ''
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)

            results[expert + str(mm)] = [[], []]
            savedir = os.path.join(config.LOG_ROOT_DIR, 'testdata')
            if (not os.path.exists(savedir)): os.mkdir(savedir)
            key = expert + str(mm)
            errors_quads = []


            nvisits_all = []
            errors_rnfl_all = []
            errors_vft_all = []


            ## variable visit but target index is fixed, use MatchedTestSet
            exp_type = 'num_visits'
            for nv_fc, si, ti in zip([1, 2, 3, 4, 5], [4, 3, 2, 1, 0], [-1,-1, -1, -1,-1]):
            #for nv_fc, si, ti in zip([4, 5], [ 1, 0], [ -1, -1]):# only few check

            ##fixed visit and target index is moved to increase gap (note 0,0, 1 in si) to make sure e inc,
            ##use this with MatchedTesSet which gives data of sequence length=6
            #exp_type = 'larger_gap'
            #for nv_fc, si, ti in zip([3, 3, 3], [0, 0, 0], [3, 4, 5]):

                print('# NV ', nv_fc)
                errors, inputs_c, preds, inputs = evaluate_reconstruction_error(config, data, mode='forecast',
                                                                                nv_fc=nv_fc, start_index=si,
                                                                                target_index=ti)
                print(config.prefix)

                results[key][0].append(errors)
                results[key][1].append([inputs, preds, inputs_c])

                # compute_vft_error(inputs, preds,mode='forecast', exp_ind=0)

                # save  forecasting predictions
                if(nv_fc >0):
                    #compare_prediction_imagesV1(preds, inputs, errors,
                    #                      save_dir=os.path.join(config.LOG_ROOT_DIR, 'testdata', 'viz'))

                    compare_prediction_imagesV1(preds, inputs, errors,
                                                save_dir=os.path.join(config.LOG_ROOT_DIR, 'testdata'+exp_type,
                                                                      'viz_traj' + str(nv_fc)+ str(si)+str(ti) ))


                if (mm[1] == 1): errors_vft_all.append(errors[1][1]) # for input [0,1]
                nvisits_all.append(nv_fc)

                for ii, e in zip(inputs_c, errors):
                    print(ii,
                          ["{0:0.2f}+-{1:0.2f}".format(np.mean(i), np.std(i)) if i is not None else None for i in e])

                for rnfl_comb_index in [0,2]:
                    print('RNFL comb index', rnfl_comb_index)
                    if (mm[0] == 1):
                        # [err, std], [err_gm, std_gm] = compute_rnfl_error(inputs, preds, rnfl_dia_mm=6, use_ma=True)
                        [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error(inputs, preds,
                                                                                                 rnfl_dia_mm=7, use_ma=True,
                                                                                                 quadrant=None,
                                                                                                 comb_index=rnfl_comb_index)
                        if(rnfl_comb_index ==2): errors_rnfl_all.append(abs_err_gm)
                        print('Global [', err_gm, '+-', std_gm, ']')

                        for q in [0, 1, 2, 3]:
                            [err, std], [err_gm, std_gm], [abs_err, abs_err_gm] = compute_rnfl_error(inputs, preds,
                                                                                                     rnfl_dia_mm=7,
                                                                                                     use_ma=True,
                                                                                                     quadrant=q,
                                                                                                     disc_dia_mm=0.4,
                                                                                                     comb_index=rnfl_comb_index)
                            print('Quad', q, '[', err_gm, '+-', std_gm, ']')


            print('saving rnfl errors with means', [np.mean(e) for e in errors_rnfl_all], 'columns', nvisits_all)
            print('saving vft errors with means', [np.mean(e) for e in errors_vft_all],'columns', nvisits_all)

            save_error(model_name='mlode_joint', target_modality='rnfl', errors=errors_rnfl_all, nv=nvisits_all,
                       exp_type=exp_type, save_dir='results')
            save_error(model_name='mlode_joint', target_modality='vft', errors=errors_vft_all, nv=nvisits_all,
                       exp_type=exp_type, save_dir='results')

            np.save(os.path.join(savedir, 'testdata_forecast.npy'), results[key])
            e = np.asanyarray(results[key][0])
            e_rnfl = list(np.clip(e[:, 0, 0], 0, 20))

            e_vft = list(np.clip(e[:, 0, 1], 0, 15))  # [:,0,2] using rnfl+vft and [:,2,2] only using vft

            e_rnfl_fromvft = list(np.clip(e[:, 2, 0], 0, 30))

            e_vft_fromrnfl = list(np.clip(e[:, 1, 1], 0, 15))

            # imbp = boxplotv1(e_rnfl, e_vft, show=True)
            # cv2.imwrite(os.path.join(savedir, 'box_plot.jpeg'), imbp)
            # imlp = line_plot(np.asanyarray(e), show=True)
            # cv2.imwrite(os.path.join(savedir, 'line_plot.jpeg'), imlp)

            imbp = boxplot_vft(e_vft, title='VFT', show=True)
            cv2.imwrite(os.path.join(savedir, 'box_plot_vft.jpeg'), imbp)

            imbp = boxplot_vft(e_vft_fromrnfl, title='VFT from RNFL', show=True)
            cv2.imwrite(os.path.join(savedir, 'box_plot_vft_from_rnfl.jpeg'), imbp)

            imbp = boxplot_rnfl(e_rnfl, title='RNFL global mean', show=True)
            cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl_global.jpeg'), imbp)

            imbp = boxplot_rnfl(e_rnfl_fromvft, title='RNFL from VFT', show=True)
            cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl_from_vft.jpeg'), imbp)

            # quadrants
            # HHYU: disabled
            #errors_quads = np.asanyarray(errors_quads)
            #loc = ['Superior', 'Inferior', 'Temporal', 'Nasal']
            #for i in range(errors_quads.shape[1]):
            #    imbp = boxplot_rnfl(list(errors_quads[:, i, :]), title='RNFL ' + loc[i], show=True)
            #    cv2.imwrite(os.path.join(savedir, 'box_plot_rnfl' + loc[i] + '.jpeg'), imbp)

