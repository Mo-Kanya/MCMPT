import os
import glob

from PIL import Image
from matplotlib import pyplot as plt
import matplotlib.animation as animation

fig_ = plt.figure()
frames = []
for img in sorted(glob.glob('SORT_outputs/*.jpg')):
    frames.append([plt.imshow(Image.open(img), animated=True)])
    os.remove(img)

outfile = img[:-10] #'_00000.jpg' = 10 characters
ani = animation.ArtistAnimation(fig_, frames, interval=200, blit=True,
                                repeat=False)
print(f'{outfile}.mp4')
ani.save(f'{outfile}.mp4')