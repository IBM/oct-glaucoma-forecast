import numpy as np
from train_multimodal_latentodegru_sync import getConfig

modalities_exp = [[1, 0]]
mm = modalities_exp[0]
experts = ['moe']
expert = experts[0]
fold_seed = 2

suffix_model_path = ''

Config = getConfig(mm, expert, latent_dim_=32, fold_seed_=fold_seed)
config = Config()
config.model = config.create_model(load_weights=True, suffix_model_path=suffix_model_path)

model = config.model


# from https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/6
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])

print(params)
