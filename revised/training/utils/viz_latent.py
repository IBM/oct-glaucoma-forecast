import torch
import cv2


class Generator:#(torch.nn.Module):

    def __init__(self, models, device, image_size):
        #super(Generator, self).__init__()
        self.models = models
        self.device = device
        self.image_size = image_size

    def forward(self, z):
        z = z.to(self.device)
        outs = []
        for m in self.models:
            #out = m.forward(z)
            out = m(z)
            out = torch.sigmoid(out)

            outs.append(out.cpu().numpy())
        # resize all to same as first modality

        resize = lambda xx, yy: ([cv2.resize(x[0], (self.image_size, self.image_size), cv2.INTER_NEAREST) for x in xx])
        stack = lambda x: np.vstack([xi[np.newaxis, np.newaxis, :, :] for xi in x])

        outs = [resize(o, outs[0]) for o in outs]
        outs = [stack(o) for o in outs]
        outs = np.concatenate(outs, axis=1)

        return outs


import numpy as np


# from cutils.common import normalize


def makegrids(n):
    grid_x = np.linspace(-4, 4, n)
    # to make the grid such that buttom left is (-4,-4) and top right is (4,4) in line to the z plot
    # grid_y = np.asarray(list(reversed(np.linspace(-4, 4, n)[::-1])))
    grid_y = np.asarray(np.linspace(-4, 4, n)[::-1])
    grids = np.meshgrid(grid_x, grid_y)
    grid1 = np.vstack(list(map(lambda x: x[np.newaxis, :, :], grids))) # (2, n,n)
    z_vevtors = grid1.reshape(2, n*n) #(2,n*n)
    return z_vevtors.transpose() #(n*n x2)


# HHYU - I can't find this, but I found a function of the same name in https://github.ibm.com/aur-mma/uncertainty-segmentation/blob/master/seguq/utils/vizutils.py
#from  retseg.utils.vizutils import stack_patches

def stack_patches(patches, nr_row, nr_col):
    """
    Stack patches, i.e,  convert the image patches to a single image
    :param patches:  N x H x W x c
    :return: nr_row*H x nr_col*W image
    """
    assert (patches.shape[0] <= nr_row * nr_col), 'The number of patches should be equal to nr_row*nr_col' + str(
        patches.shape[0]) + '<=' + str(nr_row) + 'x' + str(nr_col)

    n_blank = nr_col * nr_row - patches.shape[0]
    assert (n_blank <= nr_row and n_blank <= nr_col), ' too many blank grids,' + str(n_blank)

    patches = _pad_patches_stack(patches, nr_row * nr_col)



    rows = []
    for r in range(nr_row):
        cols = []
        for c in range(nr_col):
            # print r,c, r * nr_row + c
            cols.append(patches[r * nr_col + c, :, :, :])
        col = np.concatenate(cols, axis=1)
        rows.append(col)

    row = np.concatenate(rows, axis=0)
    return row


def generate_images_latentspace(generator, grid_size, img_chns, file_name=None):


    zz= makegrids(grid_size)
    zz = torch.from_numpy(zz.astype(np.float32))

    x_decoded = generator.forward(zz)
    x_decoded_s = x_decoded.transpose([0, 2, 3, 1])
    figure = stack_patches(x_decoded_s, grid_size,grid_size)
    opim = (figure * 255).astype(np.uint8)
    opim = cv2.copyMakeBorder(opim, 40, 40, 40, 40, cv2.BORDER_CONSTANT, None, 0)
    if (img_chns == 1): opim = np.expand_dims(opim, -1)  # add dimension removed by cv2.makeborder
    opim = opim.transpose([2, 0, 1])
    opim = np.concatenate(list(opim), axis=1)

    if (file_name is not None):
        opim_pil = Image.fromarray(opim)
        opim_pil.save(file_name)
    return opim





def generate_images_latentspace_depceciated(generator, grid_size, image_size, img_chns, file_name=None):
    # This function has issue with multiple forward passes. Produces same output. Use generate_images_latentspace
    n = grid_size
    digit_size = image_size
    digit_chn = img_chns
    figure = np.zeros((image_size * n, image_size * n, img_chns))
    batch_size = 4

    grid_x = np.linspace(-4, 4, n)
    # to make the grid such that buttom left is (-4,-4) and top right is (4,4) in line to the z plot
    #grid_y = np.asarray(list(reversed(np.linspace(-4, 4, n)[::-1])))
    grid_y = np.asarray(np.linspace(-4, 4, n)[::-1])




    for i, yi in enumerate(grid_x):
        for j, xi in enumerate(grid_y):
            z_sample = np.array([[xi, yi]])
            # print (z_sample)
            z_sample = np.tile(z_sample, batch_size).reshape(batch_size, 2)
            #print(z_sample)
            z_sample = torch.from_numpy(z_sample.astype(np.float32))


            x_decoded = generator.forward(z_sample)
            #x_decoded = generator(z_sample)

            # print ('##shape of op', np.shape(x_decoded))
            digit = x_decoded[0].transpose([1, 2, 0])

            figure[i * digit_size: (i + 1) * digit_size,
            j * digit_size: (j + 1) * digit_size] = digit

    opim = (figure * 255).astype(np.uint8)
    # opim = opim.transpose([1, 2, 0])
    opim = cv2.copyMakeBorder(opim, 40, 40, 40, 40, cv2.BORDER_CONSTANT, None, 0)
    if (img_chns == 1): opim = np.expand_dims(opim, -1)  # add dimension removed by cv2.makeborder

    opim = opim.transpose([2, 0, 1])
    opim = np.concatenate(list(opim), axis=1)

    if (file_name is not None):
        opim_pil = Image.fromarray(opim)
        opim_pil.save(file_name)
    return opim


def generate_images_random(generator, grid_size, image_size, img_chns, latent_dim, file_name=None, seed=0):
    """
    Use this when the dimension of the latent space is greater than 2
    :param generator:
    :param grid_size:
    :param image_size:
    :param img_chns:
    :param file_name:
    :return:
    """
    n = grid_size
    figure = np.zeros((img_chns, image_size * n, image_size * n))
    batch_size = 64

    num_images = grid_size * grid_size
    np.random.seed(seed)
    z_sample = np.random.random((num_images, latent_dim)) * 6 - 3  # np.array([[rnfl_xi, yi]])
    z_sample = torch.from_numpy(z_sample.astype(np.float32))

    x_decoded = generator.forward(z_sample)

    # x_decoded = x_decoded.cpu().numpy()

    for i in range(grid_size):
        for j in range(grid_size):
            img = x_decoded[i * grid_size + j].reshape(img_chns, image_size, image_size)

            figure[:, i * image_size: (i + 1) * image_size,
            j * image_size: (j + 1) * image_size] = img

    opim = (figure * 255).astype(np.uint8)

    opim = opim.transpose([1, 2, 0])
    opim = cv2.copyMakeBorder(opim, 40, 40, 40, 40, cv2.BORDER_CONSTANT, None, 0)
    if (img_chns == 1): opim = np.expand_dims(opim, -1)  # add dimension removed by cv2.makeborder

    opim = opim.transpose([2, 0, 1])
    opim = np.concatenate(list(opim), axis=1)

    if (file_name is not None):
        from PIL import Image
        opim = Image.fromarray(opim)
        opim.save(file_name)

    return opim


def generate_random_reconstructions(decoders, device, latent_dim, image_size=64):
    with torch.no_grad():
        gen_model = Generator(decoders, device, image_size=image_size)  # , model.decoder_gcl])
        viz_rand = generate_images_random(generator=gen_model, grid_size=10, image_size=image_size,
                                          img_chns=len(decoders),
                                          latent_dim=latent_dim)
    return viz_rand


def generate_latent_space(decoders, device, latent_dim, image_size=64, grid_size=10):
    assert latent_dim == 2, 'only latent_dim==2 is suported'
    with torch.no_grad():
        gen_model = Generator(decoders, device, image_size=image_size)  # , model.decoder_gcl])
        viz_rand = generate_images_latentspace(generator=gen_model, grid_size=grid_size,
                                               img_chns=len(decoders))
    return viz_rand

