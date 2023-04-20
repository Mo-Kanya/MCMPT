import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

dpi = 100

for view in ['bev', 'cam']:
    fig_ = plt.figure()
    if view == 'cam':
        fig_.set_size_inches(6.4*4, 4.8*3, True)
    elif view == 'bev':
        fig_.set_size_inches(6.4, 4.8, True)
    plt.axis('off')
    plt.margins(0,0)
    plt.tight_layout()
    frames = []
    
    for img in sorted(glob.glob(f'SORT_outputs/*_{view}.jpg')):
        frames.append([plt.imshow(Image.open(img), animated=True)])
        # frames.append([plt.imshow(Image.open(img), animated=True, aspect='auto')])
        plt.axis('off')
        # os.remove(img)

    outfile = img[:-14] #'_00000_{view}.jpg' = 14 characters
    outfile += f'_{view}'

    ani = animation.ArtistAnimation(fig_, frames, interval=200, blit=True,
                                    repeat=False)
    print(f'{outfile}.mp4')
    ani.save(f'{outfile}.mp4', dpi=dpi)
    plt.close()
    
    
    
    # plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)