import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 14})


np.set_printoptions(precision=2)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#rnfl_mod pre_process : df = rnfl.reindex(sorted(rnfl.columns[1:],reverse=True), axis=1) and mod
rnfl_matrix = pd.read_csv('/Users/hhyu/Documents/Projects-CAR/2021-suman paper revision/Artifacts/results/analysis/results_pooled/mlode_matrix_rnfl_lg.csv', header=None)
vft_matrix = pd.read_csv('/Users/hhyu/Documents/Projects-CAR/2021-suman paper revision/Artifacts/results/analysis/results_pooled/mlode_matrix_vft_lg.csv', header=None)
rnfl_matrix[rnfl_matrix <0]=np.nan
vft_matrix[vft_matrix < 0] = np.nan


fig = plt.figure()

#ax.axis('equal')
rnfl  = rnfl_matrix.values[1:, :]
print(rnfl.shape)
yticklabels = list(range(1, rnfl.shape[0] + 1))
ax=sns.heatmap(rnfl, annot=True, cbar_kws={"pad": 0.01}, yticklabels=yticklabels)
plt.xlabel('#VFTM visits')
plt.ylabel('#RNFL-TM visits')
ax.invert_yaxis()
plt.title('RNFL-TM errors')
plt.savefig('rnfltm_matrix.png')


plt.figure()
vft = vft_matrix.values[:, 1:]
print(vft)
xticklabels = list(range(1,vft.shape[1]+1))
ax = sns.heatmap(vft, annot=True,cbar_kws={"pad": 0.01}, xticklabels=xticklabels,)
#ax.axis('equal')
plt.xlabel('#VFTM visits')
plt.ylabel('#RNFL-TM visits')
ax.invert_yaxis()
plt.title('VFTM errors')

plt.savefig('vftm_matrix.png')



 
