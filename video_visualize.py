import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

plt.axis('off')

fig_ = plt.figure()
fig_.set_axis_off()
frames = []
for img in sorted(glob.glob('SORT_outputs/*_bev.jpg')):
    frames.append([plt.imshow(Image.open(img), animated=True)])
    plt.axis('off')
    os.remove(img)

outfile = img[:-14] #'_00000_xxx.jpg' = 14 characters
if 'bev' in img:
    outfile += '_bev'
elif 'cam' in img:
    outfile += '_cam'

ani = animation.ArtistAnimation(fig_, frames, interval=200, blit=True,
                                repeat=False)
print(f'{outfile}.mp4')
ani.save(f'{outfile}.mp4')