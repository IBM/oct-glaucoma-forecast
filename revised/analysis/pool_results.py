from glob import glob

models = ['control', 'pwlr', 'ode_gru', 'lode', 'mlode_joint']
modalities = ['rnfl', 'vft']
exp_types = ['num_visits', 'larger_gap']
folds = ['1', '2', '3', '4', '5']

def pool(filename):
    res = []
    header = ""
    for f in folds:
        filename = "../training/results_" + f + "/" + filename_
        fp = open(filename)
        lines = fp.readlines()
        fp.close()
        header = lines[0]
        data = lines[1:]
        res = res + data
                
    res = [header] + res
    out_filename = "results_pooled/" + filename_;
    fp = open(out_filename, 'w')
    fp.writelines(res)
    fp.close()
            
for model in models:
    for modality in modalities:
        for exp_type in exp_types:
            filename_ = model + '_' + modality + '_' + exp_type + '.csv'
            pool(filename_)

            if exp_type == 'num_visits' and modality == 'rnfl':
                filename_ = model + '_seg0_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_seg1_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_seg2_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_seg3_' + modality + '_' + exp_type + '.csv'
                pool(filename_)                

            if model == "mlode_joint":
                filename_ = model + '_unimodal_' + modality + '_' + exp_type + '.csv'
                pool(filename_)

            if model == "mlode_joint" and exp_type == 'num_visits' and modality == 'rnfl':
                filename_ = model + '_unimodal_seg0_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_unimodal_seg1_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_unimodal_seg2_' + modality + '_' + exp_type + '.csv'
                pool(filename_)
                filename_ = model + '_unimodal_seg3_' + modality + '_' + exp_type + '.csv'
                pool(filename_)

filename_ = 'strucfunc_estimation.csv'
pool(filename_)
filename_ = 'strucfunc_forecast.csv'
pool(filename_)
