import pandas as pd
import numpy as np

num_subj = np.array([205, 225, 218, 257, 237])

def read_matrix(filename):
    df = pd.read_csv(filename)
    data = df.values
    
    # return a 6x6 numpy matrix
    return data[:, 1:]

def read_matrices(stem):
    res = []
    for i in ["1", "2", "3", "4", "5"]:
        filename = "../training/results_" + i + "/" + stem
        data = read_matrix(filename)
        res.append(data)
    return res

# mlode_matrix_vft_lg.csv
rnfl_matrices = np.stack(read_matrices('mlode_matrix_rnfl_lg.csv'))
rnfl_avg = np.average(rnfl_matrices, axis=0, weights=num_subj)
rnfl_avg = np.flip(rnfl_avg, axis=0) # reverse the order of the rows (i.e, first row is no rnfl input)
np.savetxt("results_pooled/mlode_matrix_rnfl_lg.csv", rnfl_avg, delimiter=",")

vft_matrices = np.stack(read_matrices('mlode_matrix_vft_lg.csv'))
vft_avg = np.average(vft_matrices, axis=0, weights=num_subj)
vft_avg = np.flip(vft_avg, axis=0)
np.savetxt("results_pooled/mlode_matrix_vft_lg.csv", vft_avg, delimiter=",")

