import matplotlib.pyplot as plt
from train_multimodalvae import getConfig, viz_latent_space
import cv2
import  os
modalities_exp = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 0]]
experts = ['moe', 'poe']
from shutil import copyfile


for ei, expert in enumerate(experts):
    if (expert == 'poe'): modalities_exp.append([1, 1, 1])
    for mi, mm in enumerate(modalities_exp):
        #plt.subplot(2,5,ei*5+(mi+1))
        Config = getConfig(mm, expert)
        config = Config()
        summary_dir =os.path.join(config.LOG_ROOT_DIR, 'summary')
        mkdir = lambda x: os.mkdir(x) if not os.path.exists(x) else 0
        mkdir(summary_dir)

        #src_imfile=os.path.join(config.RESULT_DIR,'latent90latentspace.png')
        #dst_imfile=os.path.join(summary_dir, 'latent90latentspace'+config.prefix+'.png')
        # copyfile(src_imfile, dst_imfile)

        viz_latent_space(config.create_model(load_weights=True),mm, os.path.join(summary_dir,'newlatentspace_'+config.prefix+'.jpeg'), image_size=32)


        #im = cv2.imread(src_imfile)

        #plt.imshow(im)
        #plt.title(config.prefix)



    #plt.savefig(os.path.join(config.LOG_ROOT_DIR,'summary/summary_losses.jpeg'))

