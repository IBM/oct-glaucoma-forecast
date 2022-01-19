import numpy as np
import matplotlib.pyplot as plt
import cv2
data=np.load('/Users/ssedai/git/oct_forecasting/temp/onh_mac_vft_data_1000.npz')
rnfl_mac = data['rnfls_mac']
gcl_mac = data['gcls_mac']
rnfl_onh = data['rnfls_onh']
vft = data['vft']
mdata_mac = data['metadata_mac']
eye = data['eye']


def plots(idx, mac, onh, vft, mdata,eye):
    dx = mdata[idx, 0, -1]
    eye = eye[idx,0]
    for i in range(4):
        #plt.close()
        plt.subplot(3,4,i+1)
        plt.imshow(mac[idx,i],cmap='jet')

        plt.subplot(3, 4, 4+i + 1)
        plt.imshow(onh[idx, i], cmap='jet')
        plt.subplot(3, 4, 8 + i + 1)
        vftim = vft[idx, i]
        #vftim=cv2.rotate(vftim,cv2.ROTATE_90_COUNTERCLOCKWISE)
        plt.imshow(vftim, cmap='gray')
        plt.title(str(dx)+'-'+eye+'{:.1f}'.format(np.mean(vftim)))

    plt.show()


#vft= 40 - vft
plots(178, rnfl_mac,rnfl_onh, vft, mdata_mac,eye)
dx = mdata_mac[:,0,-1]
normal_idx = np.where(dx=='Normal')[0]


# ecxample of thinner rnfl 755 and 766

#759 is interestingh

#in vft value =  40 - threshold ==>
# black pixel means higher threshold  -- should be lower thickness??
#white pixels means lower threshold -- should be larger thickness??

#for patient 775, the thickness is low but vft is white

#Normal 779
#138 for viz in paper intro (normal)



# patient 175 #Progressinve loss of vft and rnfl
#patient 189, no rnfl but vft
