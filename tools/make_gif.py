import os
import imageio
images = []
dirname = "sample_outputs/2023-12-09-21:58:29"

for filename in os.listdir(dirname):
    if filename.endswith(".jpg"):
        f = os.path.join(dirname, filename)
        img = imageio.imread(f)
        if img.shape != (1175, 2580, 3):
            continue
        images.append(img)
imageio.mimsave('sample_outputs.gif', images, duration=200)