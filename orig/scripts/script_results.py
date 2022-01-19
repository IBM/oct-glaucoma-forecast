import cv2
import matplotlib.pyplot as plt


#display joint latent space
im=cv2.imread('/Users/ssedai/git/multimodal_latentode/MSEsync_nonadj_euler_ce_temp_mlode2_foldseed2/sync_multimoda_latentode211_poe/latent30latentspace.png', cv2.IMREAD_GRAYSCALE)

fig = plt.figure(figsize=(10, 8), dpi=100)
#ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

rnfl= im[:,:int(1184/2)]
rnfl = rnfl * (200/255.0)
pos=plt.imshow( rnfl[40: rnfl.shape[0]-40, 40:rnfl.shape[1]-40], cmap='jet', vmin=0, vmax=200)
cbar=plt.colorbar(pos, pad=0.01, aspect=45)
cbar.ax.tick_params(labelsize=14)

plt.axis('off')
plt.savefig('results/latentviz_rnfl.png')

fig = plt.figure(figsize=(10, 8), dpi=100)
#ax = fig.add_axes([0, 0, 1, 1], frameon=False, xticks=[], yticks=[])

vft = im[:,int(1184/2):]
vft = vft * (40/255)
pos=plt.imshow(vft[40:vft.shape[0]-40, 40:vft.shape[1]-40], cmap='gray', vmin=0, vmax=40)
cbar = plt.colorbar(pos, pad=0.01, aspect=45)
cbar.ax.tick_params(labelsize=14)
plt.axis('off')

plt.savefig('results/latentviz_vft.png')





