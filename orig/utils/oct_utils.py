import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.utils import get_mask_rnfl
def draw_triangle(pt1, pt2, pt3, image):


    triangle_cnt = np.array([pt1, pt2, pt3]).astype(np.int32)

    triangle_cnt = np.expand_dims(triangle_cnt, 1)# (N,1,2)
    #print(triangle_cnt)

    cv2.drawContours(image, [triangle_cnt], 0, 255, -1)
    return image


def get_clockhours_reference(theta=15, image_size=200):

    image = np.zeros((image_size, image_size))
    center = [ int(image_size/2), (image_size/2)]
    center = np.asanyarray(center)
    p1 = np.asanyarray([image_size*2, int(image_size/2)])
    p11 = np.concatenate([p1, np.asanyarray([1])])
    M_up = cv2.getRotationMatrix2D( tuple(center) , theta, 1)
    p2 = np.matmul(M_up, p11) #[:2]
    image=draw_triangle(tuple(center),tuple(p1),tuple(p2), image)

    M_down = cv2.getRotationMatrix2D( tuple(center), -theta, 1)
    p2 = np.matmul(M_down, p11) #[:2]
    image=draw_triangle(tuple(center),tuple(p1),tuple(p2), image)

    image = image/255.0

    return image


def get_quardrants(rnfl_diam_mm=5, image_size_return=200, disc_dia_mm=1.92):
    """

    :param rnfl_diam_mm:
    :return: the list of masks in order of superior, inferior, temporal, nasal: temporal is side where fovea is located
    and nasal is beyond disc nose
    """

    image_size=200
    mask_rnfl = get_mask_rnfl(image_size,image_size,rnfl_diam_mm=rnfl_diam_mm, disc_dia_mm=disc_dia_mm)

    masks =[]
    center = [int(image_size / 2), (image_size / 2)]
    for q in [90, 270, 0, 180]:
        im = get_clockhours_reference(theta=60, image_size=image_size)
        M = cv2.getRotationMatrix2D(tuple(center), q, 1)
        im = cv2.warpAffine(im, M, tuple(im.shape[::-1]), borderValue=0.0)
        im= im*mask_rnfl
        im = cv2.resize(im, (image_size_return, image_size_return), interpolation = cv2.INTER_NEAREST)
        masks.append(im)
    return masks




if (__name__=='__main__'):
    #im = get_clockhours_reference(theta=30)
    #M= cv2.getRotationMatrix2D((100, 100), 270, 1)
    #im = cv2.warpAffine(im, M, tuple(im.shape[::-1]))
    quadrants = get_quardrants(rnfl_diam_mm=7.5, disc_dia_mm=0.5, image_size_return=32)
    for im in quadrants:
        plt.imshow(im)
        plt.show()


