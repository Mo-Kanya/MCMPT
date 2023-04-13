import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

for view in ['cam', 'bev']:
    fig_ = plt.figure()
    plt.axis('off')
    frames = []
    
    for img in sorted(glob.glob(f'SORT_outputs/*_{view}.jpg')):
        frames.append([plt.imshow(Image.open(img), animated=True)])
        plt.axis('off')
        os.remove(img)

    outfile = img[:-14] #'_00000_{view}.jpg' = 14 characters
    outfile += f'_{view}'
    # if 'bev' in img:
    #     outfile += '_bev'
    # elif 'cam' in img:
    #     outfile += '_cam'

    ani = animation.ArtistAnimation(fig_, frames, interval=200, blit=True,
                                    repeat=False)
    print(f'{outfile}.mp4')
    ani.save(f'{outfile}.mp4')
    plt.close()