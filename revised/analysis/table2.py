import pandas as pd
#result_dir = "../training/results_5/"
result_dir = "results_pooled/"

models = ['control', 'pwlr', 'ode_gru', 'lode']
modalities = ['rnfl', 'vft']
exp_type = 'larger_gap'

for model in models:
    for modality in modalities:
        filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
        df = pd.read_csv(filename)
        for col in ['0_3', '1_3', '2_3']:
            data = df[col]
            m = data.mean()
            std = data.std()
            print(f"{m:.2f}({std:.2f})\t", end="")
        print("|\t", end="")
    print("")

print("")

filename = result_dir + 'mlode_joint_unimodal_rnfl_' + exp_type + '.csv'
df = pd.read_csv(filename)
for col in ['0_3', '1_3', '2_3']:
    data = df[col]
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")
print("")

filename = result_dir + 'mlode_joint_unimodal_vft_' + exp_type + '.csv'
df = pd.read_csv(filename)
print("           \t           \t           \t", end="")
print("|\t", end="")
for col in ['0_3', '1_3', '2_3']:
    data = df[col]
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")
print("")

for model in ['mlode_joint']:
    for modality in modalities:
        filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
        df = pd.read_csv(filename)
        for col in ['0_3', '1_3', '2_3']:
            data = df[col]
            m = data.mean()
            std = data.std()
            print(f"{m:.2f}({std:.2f})\t", end="")
        print("|\t", end="")
    print("")
