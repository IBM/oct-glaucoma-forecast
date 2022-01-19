from train_multimodalvae import getConfig, MultimodalData, evaluate_reconstruction_error, visualize_embedding
import numpy as np

np.set_printoptions(precision=2)

#modalities_exp = [[1, 1, 0], [1,0,0]]
modalities_exp = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]]

modalities_exp = [ [1, 0, 0]]


experts = ['moe']#,'poe']

data = MultimodalData()
for ei, expert in enumerate(experts):
    #if (expert == 'poe'): modalities_exp.append([1, 1, 1])

    for mi, mm in enumerate(modalities_exp):
        Config = getConfig(mm, expert, latent_dim_=32)
        config = Config()

        config.model = config.create_model(load_weights=True)
        errors, inputs_c, preds, inputs = evaluate_reconstruction_error(config, data)
        print(config.prefix)
        for i, e in zip(inputs_c, errors):
            #print(i, np.asanyarray(e))#'{:.2f}'.format(e))
            print(i, ["{0:0.2f}".format(i) if i is not None else None for i in e])
        #visualize_embedding(config,data,type='umap')


#First rehshape or stack time axis if it exists
from data.utils import mask_rnfl
rnfl = inputs[0][0].numpy()
rnfl_pred = preds[0][0].numpy()
rnfl_masked = mask_rnfl(rnfl, channel_last=False)[0]
rnfl_pred_masked = mask_rnfl(rnfl_pred, channel_last=False)[0]
# MAE error
errors=np.mean(np.abs(rnfl_masked-rnfl_pred_masked).reshape(rnfl_masked.shape[0],-1),axis=1)*200
err, std = np.mean(errors),np.std(errors)

#global_mean
errors_gm=  np.abs(np.mean(rnfl_masked.reshape(rnfl_masked.shape[0],-1), axis=1) -np.mean(rnfl_pred_masked.reshape(rnfl.shape[0],-1), axis=1))*200
err_gm, std_gm = np.mean(errors_gm),np.std(errors_gm)