import torch
import torch.nn as nn
from models.model_blocks import RNFLDecoder, RNFLEncoder
rd=RNFLDecoder(z_size=2,rnfl_imgChans=1, rnfl_fBase=32)
z=torch.rand((5,2))

o = rd(z)
print(o.shape)

im=torch.rand((5,1,64,64))
re = RNFLEncoder(latent_dim=2,rnfl_fBase=32)
mulogvar = re(im)
print(mulogvar.shape)





