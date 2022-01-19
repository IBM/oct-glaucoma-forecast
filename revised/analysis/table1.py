import pandas as pd
#result_dir = "../training/results_1/"
result_dir = "results_pooled/"

models = ['control', 'pwlr', 'ode_gru', 'lode']
modalities = ['rnfl', 'vft']
exp_type = 'num_visits'

for model in models:
    modality = 'rnfl'
    for q in [0, 1, 2, 3]:
        filename = result_dir + model + '_seg' + str(q) + '_' + modality + '_' + exp_type + '.csv'
        df = pd.read_csv(filename)
        data = df['4_5']
        m = data.mean()
        std = data.std()
        print(f"{m:.2f}({std:.2f})\t", end="")

    modality = 'rnfl'
    filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
    df = pd.read_csv(filename)
    data = df['4_5']
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")
    print("|\t", end="")

    modality = 'vft'
    filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
    df = pd.read_csv(filename)
    data = df['4_5']
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")
    print("|\t", end="")
    print("")

modality = 'rnfl'
model = 'mlode_joint_unimodal'
for q in [0, 1, 2, 3]:
    filename = result_dir + model + '_seg' + str(q) + '_' + modality + '_' + exp_type + '.csv'
    df = pd.read_csv(filename)
    data = df['4_5']
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")

modality = 'rnfl'
model = 'mlode_joint_unimodal'
filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
df = pd.read_csv(filename)
data = df['4_5']
m = data.mean()
std = data.std()
print(f"{m:.2f}({std:.2f})\t", end="")
print("")

modality = 'vft'
model = 'mlode_joint_unimodal'
filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
df = pd.read_csv(filename)
print("           \t           \t           \t           \t           \t   \t", end="")
data = df['4_5']
m = data.mean()
std = data.std()
print(f"{m:.2f}({std:.2f})\t", end="")
print("")

modality = 'rnfl'
model = 'mlode_joint'
for q in [0, 1, 2, 3]:
    filename = result_dir + model + '_seg' + str(q) + '_' + modality + '_' + exp_type + '.csv'
    df = pd.read_csv(filename)
    data = df['4_5']
    m = data.mean()
    std = data.std()
    print(f"{m:.2f}({std:.2f})\t", end="")

modality = 'rnfl'
filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
df = pd.read_csv(filename)
data = df['4_5']
m = data.mean()
std = data.std()
print(f"{m:.2f}({std:.2f})\t", end="")
print("|\t", end="")

modality = 'vft'
filename = result_dir + model + '_' + modality + '_' + exp_type + '.csv'
df = pd.read_csv(filename)
data = df['4_5']
m = data.mean()
std = data.std()
print(f"{m:.2f}({std:.2f})\t", end="")
print("|\t", end="")
print("")
