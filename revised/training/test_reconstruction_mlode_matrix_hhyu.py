import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

from scripts.datautils import MatchedTestSet
# from data.utils import mask_rnfl
from scripts.eval_mlode_sync import evaluate_forecast_matrix_largegap, create_vft_mask, create_rnfl_mask
from train_multimodal_latentodegru_sync import getConfig

np.set_printoptions(precision=2)
import pandas as pd
import matplotlib.pyplot as plt

def plot_matrices():
    import seaborn as sns

    #rnfl_mod pre_process : df = rnfl.reindex(sorted(rnfl.columns[1:],reverse=True), axis=1) and mod
    rnfl_matrix = pd.read_csv('/Users/ssedai/git/multimodal_latentode/results/mlode_matrix_rnfl_mod.csv',index_col=0 )
    vft_matrix = pd.read_csv('/Users/ssedai/git/multimodal_latentode/results/mlode_matrix_vft_mod.csv',index_col=0)
    rnfl_matrix[rnfl_matrix <0]=np.nan
    vft_matrix[vft_matrix < 0] = np.nan


    fig = plt.figure()

    #ax.axis('equal')
    rnfl  = rnfl_matrix.values[1:, :]
    yticklabels = list(range(1, rnfl.shape[0] + 1))
    ax=sns.heatmap(rnfl, annot=True, cbar_kws={"pad": 0.01}, yticklabels=yticklabels)
    plt.xlabel('#VFTM visits')
    plt.ylabel('#RNFL-TM visits')
    ax.invert_yaxis()
    plt.title('RNFL-TM errors')
    plt.savefig('results/rnfltm_matrix.png')




    plt.figure()
    vft = vft_matrix.values[:, 1:]
    xticklabels = list(range(1,vft.shape[1]+1))
    ax = sns.heatmap(vft, annot=True,cbar_kws={"pad": 0.01}, xticklabels=xticklabels,)
    #ax.axis('equal')
    plt.xlabel('#VFTM visits')
    plt.ylabel('#RNFL-TM visits')
    ax.invert_yaxis()
    plt.title('VFTM errors')

    plt.savefig('results/vftm_matrix.png')




    plt.show()

    #vft_matrix.values[:,1:]


#if (__name__ == '__main__'):
#    plot_matrices()


if (__name__ == '__main__'):

    # modalities_exp = [ [1,0,0],[0,0,1],[1, 0, 1]]

    rnfl_dia_mm = 7
    modalities_exp = [[1, 1]]  # [[1, 0, 1]]
    experts = ['poe']  # ,'poe']
    fold_seed = 1
    results = {}
    # for seed 2
    #idx_r = [ 107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0

    # idx_r = [ 84, 324, 162, 3]  # filter for [1,0] input ie rnfl comb index=2
    # rnfl_comb_Index 0 - inputs 11 and 2 means 1,0 and 1 means 01 the input comb in format [rnfl, vft]
    # HHYU - was 2, changed to 0
    rnfl_comb_index = 0

    # data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
    data = MatchedTestSet(seed = fold_seed)
    print('Number of test samples', data.val_rnflonh.shape[0])
    # suffix_model_path ='_epoch15_rnflerr272'# ''
    suffix_model_path = ''
    for ei, expert in enumerate(experts):

        for mi, mm in enumerate(modalities_exp):
            Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
            config = Config()
            config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)
            inputs, ts_list_val, dx = data.get_val()
            rnfl_mask = create_rnfl_mask(inputs[0], rnfl_diam_mm=rnfl_dia_mm)
            vft_mask = create_vft_mask(inputs[1])
            masks = [rnfl_mask, vft_mask]
            out = evaluate_forecast_matrix_largegap(config.model, ts_list_val, inputs, masks, mm)

            print('RNFL')
            rnfl_matrix = []
            vft_matrix = []
            for rn, row in enumerate(out):
                rowdisp = '#' + str(rn)
                rnfl_row = []
                vft_row = []
                for cn, col in enumerate(row):
                    errors = col[0]
                    rnfl_err = errors[0].detach().cpu().numpy()
                    vft_err = errors[1].detach().cpu().numpy()
                    rowdisp += '#' + str(cn) + ' ' + str(np.mean(rnfl_err)) + ' '
                    rnfl_row.append(np.mean(rnfl_err))
                    vft_row.append(np.mean(vft_err))

                rnfl_matrix.append((rnfl_row))
                vft_matrix.append(vft_row)

                print(rowdisp)

            rnfl_matrix.reverse()
            vft_matrix.reverse()
                
            # no_cols=6 for normal gao
            # no cols = 5 for larger gap
            no_cols=6
            pd.DataFrame(rnfl_matrix, columns=['row' + str(r) for r in range(no_cols)]).to_csv(
                'results_' + str(fold_seed) + '/mlode_matrix_rnfl_lg.csv', float_format='%.3f')
            pd.DataFrame(vft_matrix, columns=['row' + str(r) for r in range(no_cols)]).to_csv('results_' + str(fold_seed) + '/mlode_matrix_vft_lg.csv', float_format='%.3f')

            print('VFT')
            for rn, row in enumerate(out):
                rowdisp = '#' + str(rn)
                for cn, col in enumerate(row):
                    errors = col[0]
                    rnfl_err = errors[0].detach().cpu().numpy()
                    vft_err = errors[1].detach().cpu().numpy()
                    rowdisp += '#' + str(cn) + ' ' + str(np.mean(vft_err)) + ' '

                print(rowdisp)

            print('Done')
