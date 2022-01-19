import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy import stats

np.set_printoptions(precision=2)


def boxplot_gen(errors, labels, show=True, title='VFT', showmeans=True):

    fig = plt.figure(1, figsize=(9, 6))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    data_to_plot = errors  # create N_t lists each of size N_samples
    bp = ax.boxplot(data_to_plot, showmeans=showmeans)
    plt.ylim([0, 20])

    major_ticks = np.arange(0, 10, 5)
    minor_ticks = np.arange(0, 10, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(b=True, which='both')

    ax.set_xticklabels(labels)
    ax.set_ylabel('MAE DB')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot_vft.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def boxplot_grouped(errors, labels, show=True, title='VFT', showmeans=True, colors=None, y_label='', exclude_lrv12=False):
    import matplotlib.pyplot as plt

    fig = plt.figure(1, figsize=(10, 7))
    ax = fig.add_subplot(111)

    ax.set_title(title)
    data_to_plot = errors  # create N_t lists each of size N_samples
    # bp = ax.boxplot(data_to_plot, showmeans=showmeans)

    # bp=ax.boxplot(errors[0], positions=[1,4,7,10,13])
    # bp=ax.boxplot(errors[1], positions=[2,5,8,11,14])

    n_models = len(errors)
    Nt = len(errors[1])
    colors_all=[]
    color_dict={}
    bp_list=[]
    for i in range(n_models):
        post = list(range(i + 1, Nt * (n_models + 1), (n_models + 1)))
        print(post)
        if(exclude_lrv12 and i==0):
            post= post[2:]
            errs_plt= errors[i][2:]
        else:
            errs_plt = errors[i]

        for j in post: color_dict[j]= colors[i]

        bp = ax.boxplot(errs_plt, positions=post,showmeans=True, patch_artist=True)
        bp_list.append(bp)

    for abp,col in zip(bp_list, colors):
        for patch in abp['boxes']:
            patch.set_facecolor(col)


    plt.ylim([0, 15])
    plt.xlim(0, (Nt * (n_models + 1)))

    major_ticks = np.arange(0, 10, 5)
    minor_ticks = np.arange(0, 10, 1)
    ax.set_yticks(major_ticks)
    ax.set_yticks(minor_ticks, minor=True)
    ax.grid(b=True, which='both')

    labels_disp = labels #['#nvisits '+l for l in labels]
    #ax.set_xticklabels(labels_disp, ha='right')
    ax.set_ylabel(y_label)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    temp_file = os.path.join(tempfile.gettempdir(), 'box_plot_vft.jpeg')
    plt.savefig(temp_file)
    if (show): plt.show()
    im = cv2.imread(temp_file, cv2.IMREAD_COLOR)
    return im


def get_errors(f):
    df1 = pd.read_csv(f)

    errors1 = [df1[c].to_list() for c in df1.columns if 'Unnamed' not in c]
    labels1 = [c for c in df1.columns if 'Unnamed' not in c]
    return errors1, labels1



def display_errors(errors, models, labels):

    for i in range(len(models)):
        errsi = errors[i]
        model_name =models[i]
        for err, l, ref_err in zip(errsi, labels, errors[-1]):
            t,p = t_test(err, ref_err)
            print(model_name, l, np.mean(err),'+-', np.std(err), 'p=', p)


def t_test(errors0, errors1):

    t2, p2 = stats.ttest_ind(errors0, errors1, equal_var=False)
    return t2, p2



if (__name__ == '__main__'):
    result_dir = '../results_orig'
    mode = 'num_visits'
    #mode = 'larger_gap'
    
    target = 'rnfl'
    models = ['pwlr', 'ode_gru','lode', 'mlode_joint']
    colors=['red','blue','green','yellow']

    filename = lambda x, y, z: '_'.join([x, y, z]) + '.csv'
    files = [os.path.join(result_dir, filename(m, target, mode)) for m in models]

    errors, labels = zip( *[get_errors(f) for f in files] )
    #errors is list of size len(modes) and each item is a list of size N_t.

    #errors1, labels1 = get_errors(f1)
    #errors2, labels2 = get_errors(f2)
    #errors3, labels2 = get_errors(f3)

    if(mode=='num_visits'):
        label_disp = [1, 2, 3, 4, 5]
        label_disp = ['#nvisits '+str(d) for d in label_disp]
    else:

        label_disp=  [1,2,3]
        label_disp = ['#visit gap '+str(d) for d in label_disp]

    display_errors(errors, models, label_disp)

    filename =  'boxplot'.join([target, mode]) + '.png'
    y_label='MAE (micron)' if target=='rnfl' else 'MAE (DB)'
    im = boxplot_grouped(errors, label_disp, show=True, title='', showmeans=True, colors=colors, y_label=y_label)
    cv2.imwrite('results/'+filename, im)
    #plt.savefig('results/boxplot_'+mode+'.jpeg')
