import numpy as np
import cv2

def resize_stack(stack, newsize):
    """

    :param stack: (N, nt, H,W,1)
    :return:
    """
    stack = np.squeeze(stack)  # (N, nt, H,W)
    stack = stack.transpose([0, 2, 3, 1])  ##(N,  H,W, nt)
    images_new = []
    for im in stack:
        imnew = cv2.resize(im, newsize, interpolation=cv2.INTER_LINEAR)
        images_new.append(imnew)
    images_new = np.asanyarray(images_new)  # (N,  H,W, nt)
    images_new = images_new.transpose([0, 3, 1, 2])
    images_new = np.expand_dims(images_new, -1)
    return images_new


def resize_stack_nonseries(stack, newsize):
    """
    :param stack: (N,  H,W,1)
    :return:
    """
    stack = np.squeeze(stack)  # (N,  H,W)
    images_new = []
    for im in stack:
        imnew = cv2.resize(im, newsize, interpolation=cv2.INTER_LINEAR)
        images_new.append(imnew)
    images_new = np.asanyarray(images_new)  # (N,  H,W)

    images_new = np.expand_dims(images_new, -1)  # (N,  H,W,1)
    return images_new


def get_mask_rnfl(H,W, disc_dia_mm=1.92, rnfl_diam_mm=5):
    assert H == W , ' invalid image format'

    disc_dia = H * disc_dia_mm / 6.0
    disc_radius = int(disc_dia / 2.0)
    cx, cy = int(W / 2), int(H / 2)

    rnfl_dia = H * rnfl_diam_mm / 6.0
    rnfl_radius = int(rnfl_dia / 2.0)

    disc_synt = cv2.circle(np.zeros((H, W)), (cx, cy), disc_radius, 255, -1).astype(np.bool)
    mask_ring = cv2.circle(np.zeros((H, W)), (cx, cy), rnfl_radius, 255, -1).astype(np.bool)
    mask_ring = mask_ring * ~disc_synt
    return mask_ring




def mask_rnfl(stack, channel_last=True, rnfl_diam_mm=5):
    """

    :param stack: (N, nt, H,W,1)  if channel_last=True else (N, nt, 1, H,W)   [pixel range [0,255]
    :return: masked stack and the mask
    """
    if(len(stack.shape)==5):
        if(channel_last):
            (N, nt, H, W, ch) = stack.shape
        else:
            (N, nt, ch, H, W) = stack.shape
    elif (len(stack.shape)==4):
        if(channel_last):
            (N,  H, W, ch) = stack.shape
        else:
            (N, ch, H, W) = stack.shape


    assert H == W and ch == 1, ' invalid image format'

    disc_dia = H * 1.92 / 6.0
    disc_radius = int(disc_dia / 2.0)
    cx, cy = int(W / 2), int(H / 2)

    rnfl_dia = H*rnfl_diam_mm/6.0
    rnfl_radius = int(rnfl_dia / 2.0)

    disc_synt = cv2.circle(np.zeros((H, W)), (cx, cy), disc_radius, 255, -1).astype(np.bool)
    mask_ring = cv2.circle(np.zeros((H, W)), (cx, cy), rnfl_radius, 255, -1).astype(np.bool)
    mask_ring = mask_ring * ~disc_synt

    if(channel_last):
        mask_ring = np.expand_dims(mask_ring, -1)  # (H,W,1)
    else:
        mask_ring = np.expand_dims(mask_ring, 0)  # (H,W,1)

    stack = stack * mask_ring
    return stack, mask_ring

