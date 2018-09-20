import matplotlib.pyplot as plt
import numpy as np

from skimage.segmentation import slic, quickshift
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io

import os

inputfile = 'coins.jpg'
output_slic = os.path.splitext(inputfile)[0] + '_slic' + '.png'
output_quickshift = os.path.splitext(inputfile)[0] + '_quickshift' + '.png'

image = img_as_float(io.imread(inputfile))

segments_slic = slic(image, n_segments = 100, sigma = 5)
segments_quickshift = quickshift(image, kernel_size = 5, max_dist = 6, ratio = 0.5)

slic_result = mark_boundaries(image, segments_slic, color = (1, 0, 0))
quickshift_result = mark_boundaries(image, segments_quickshift, color = (1, 0, 0))

fig, ax = plt.subplots(1, 2, figsize=(20, 10), sharex=True, sharey=True)

ax[0].imshow(slic_result)
ax[0].set_title('slic')
ax[1].imshow(quickshift_result)
ax[1].set_title('quickshift')

for a in ax.ravel():
    a.set_axis_off()

plt.tight_layout()
plt.show()

plt.imsave(output_slic, slic_result)
plt.imsave(output_quickshift, quickshift_result)
