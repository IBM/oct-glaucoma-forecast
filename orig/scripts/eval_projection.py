import os

import numpy as np
from scripts.datautils import MatchedTestSet
from train_multimodal_latentodegru_sync import getConfig, visualize_embedding
from    sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib
matplotlib.rcParams.update({'font.size': 14})

import matplotlib.pyplot as plt


def tsne_proj(mu, perplexity=40):

    tsne = TSNE(n_components=2, verbose=1, perplexity=perplexity, n_iter=10000, learning_rate=10)
    tsne_results = tsne.fit_transform(mu)
    return tsne_results



def plot(z, labels, save_path=None):
    labels_dict = {2: 'Glaucoma', 1: 'GS', 0: 'Normal'}
    replace = lambda x: labels_dict[x]
    groups = np.array(list(map(replace, labels)))

    cdict = {'Glaucoma': 'red', 'Normal': 'green', 'GS': 'green'}
    mdict = {'Glaucoma': 'o', 'Normal': '^', 'GS': '^'}
    ldisp = {'Glaucoma': 'Glaucoma', 'Normal': 'Normal', 'GS': 'Glaucoma Suspect'}

    fig, ax = plt.subplots(figsize=(8, 6))
    for g in np.unique(groups):
        print('Group',g)
        ix = np.where(groups == g)
        print(ix)
        ax.scatter(z[ix, 0], z[ix, 1], color="None", marker=mdict[g], edgecolors=cdict[g], linewidth=2, label=ldisp[g], s=100)
    ax.legend()
    plt.xlabel('z [0]')
    plt.ylabel('z [1]')
    plt.xlim(-3.2,3.2)
    plt.ylim(-3.2, 3.2)
    if(save_path is not None):
        plt.savefig(save_path)

    plt.show()




modalities_exp = [[1, 1]]  # [[1, 0, 1]]
experts = ['poe']  # ,'poe']
fold_seed = 2
results = {}
# for seed 2
#idx_r = [107, 342, 191, 325, 20, 330, 155, 329, 340, 85, 324, 162, 3]  # for [1,1] input filter rnfl comb index=0



# data = MultimodalTimeSeriesData(fold_seed=fold_seed, idx_r=idx_r)
data = MatchedTestSet()
print('Number of test samples', data.val_rnflonh.shape[0])
latent_dim=2
suffix_model_path = ''
for ei, expert in enumerate(experts):

    for mi, mm in enumerate(modalities_exp):
        Config = getConfig(mm, expert, latent_dim_=latent_dim, fold_seed_=fold_seed)
        config = Config()
        config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)
        proj, mu, labels = visualize_embedding(config=config, data=data, epoch=None)
        print('Done')
        z_tsne = tsne_proj(mu, perplexity=10)
        if(latent_dim==2):
            plot(mu, labels, save_path='results/zplot_ld2.png')
            plot(z_tsne, labels, save_path='results/zplottsne_ld2.png')
        else:
            plot(z_tsne, labels)



