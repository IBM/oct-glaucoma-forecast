import torch
from utils.viz_latent import generate_latent_space, generate_random_reconstructions

if (__name__ == '__main__'):
    from train_multimodalvae import  getConfig

    GPU = 0
    device = torch.device('cuda:' + str(GPU)
                          if torch.cuda.is_available() else 'cpu')

    modalities=[1,0,1]
    Config = getConfig(modalities, 'moe')
    config = Config()
    model = config.create_model(load_weights=True).to(device)
    generators = [model.vae_rnfl.decoder, model.vae_gcl.decoder, model.vae_vft.decoder]
    generators = [g for g, m in zip(generators, modalities) if m]
    viz_latent = generate_latent_space(generators, device=device, latent_dim=model.latent_dim, image_size=32,
                                     grid_size=16)
    #viz_rand = generate_random_reconstructions(generators, device=device, latent_dim=model.latent_dim,
    #                                           image_size=32)

    from PIL import Image

    Image.fromarray(viz_latent).show()
