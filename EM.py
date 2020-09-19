import skimage
import skimage.io
from skimage import feature
from skimage import exposure
from skimage import morphology
import numpy as np
from scipy import ndimage as ndi
from scipy.linalg import pascal
from skimage.measure import compare_ssim

SAMP_LEN = 3
WEIGHTS = pascal(SAMP_LEN, 'lower')[-1][:SAMP_LEN] # binomial weights

def process(pg):
    p = ndi.gaussian_filter(pg, 1)
    p = exposure.equalize_adapthist(p)
    s = skimage.filters.sobel(p)
    s = morphology.area_closing(s)
    sc = s * 255 / np.max(s)
    p = p * s
    p = p * 255 / np.max(p)
    return p, sc

''' Process a sequence of images where x is the index of the middle image to be processed
and the sequence is formed by x +/- num.'''
def processSeq(x, num, prefix):
    inFile = prefix + 'piece-' + str(x).rjust(4, '0') + '.tif'
    return

def initialize(prefix):
    global gradients
    x = 0
    while True:
        x += 1
        try:
            inFile = prefix + str(x).rjust(4, '0') + '.tif'
            p, s = process(skimage.io.imread(inFile))
            gradients[x] = s
            skimage.io.imsave(inFile.replace('.tif', '-sobel.tif'), s.astype(np.uint8))
        except IOError:
            break

gradients = {}








# def descent():
#     if len(gradients) < 2:
#         return







# sa = 2 * sc1 + sc2 + sc3
# sa = exposure.equalize_hist(sa)
# sc = sa * 255 / np.max(sa)
# skimage.io.imsave('em/pieces-sobel.tif', sc.astype(np.uint8))
# #sc = morphology.area_closing(sc, connectivity=2)
#
# ssq = np.sqrt(sc)
# pm = pg * sc
# pm = exposure.equalize_hist(pm)
# pm = pm * 255 / np.max(pm)
# skimage.io.imsave('em/pieces-mult.tif', pm.astype(np.uint8))
#
# bw = skimage.filters.threshold_local(sa, block_size=35)
# th = (sa > bw) * 255
# skimage.io.imsave('em/pieces-bw.tif', th.astype(np.uint8))
#
#
# pm = morphology.closing(th, morphology.square(2))
# skimage.io.imsave('em/pieces-bw-close.tif', pm.astype(np.uint8))

# visual conformation of closing function
# delta = pm - th
# skimage.io.imsave('em/pieces-bw-delta.tif', delta.astype(np.uint8))


# th = morphology.skeletonize(pm.astype(np.bool))
# skimage.io.imsave('em/pieces-bw-thin.tif', pm.astype(np.uint8))



# x = skimage.dtype_limits(s)[1]
# s = s * 255 / np.max(s)

