#USE PYTHON 2 to access cellprofiler project and dependencies

import sys

import math
import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import centrosome.threshold
import scipy.ndimage
import scipy.sparse
import skimage.morphology
import cellprofiler.gui.help
import cellprofiler.gui.help.content
import cellprofiler.object
import cellprofiler.setting
from cellprofiler.modules import _help, threshold, identifyprimaryobjects
from cellprofiler.modules.identifyprimaryobjects import *
import cellprofiler.modules
import cellprofiler.image
from cellprofiler.image import *

import skimage
from skimage import external
import pickle

''' 
Changed required to CellProfiler Source code:
In identifyprimaryobjects.py

    LINE 1131: def separate_neighboring_objects(self, workspace, labeled_image, object_count):
    CHANGE TO: def separate_neighboring_objects(self, cpimage, labeled_image, object_count):

    COMMENT LINES 1147 AND 1148:         
    	#cpimage = workspace.image_set.get_image(
        ##        self.x_name.value, must_be_grayscale=True)

'''
class IdentifyPrimaryObjectsBridge(IdentifyPrimaryObjects):

	def __init__(self):

		self.threshold = threshold.Threshold()
		super(IdentifyPrimaryObjects, self).__init__()
		self.use_advanced.value = True
		self.__dict__['unclump_method'] = UN_SHAPE

inFile, binFile, labeledFile = sys.argv[1], sys.argv[2], sys.argv[3]

with skimage.external.tifffile.TiffFile(inFile) as pic:
	pic_array = pic.asarray()

with skimage.external.tifffile.TiffFile(binFile) as binary:
	bin_array = binary.asarray()

with open(labeledFile, 'rb') as labeled:
	labeled_array = pickle.load(labeled)

primaryObjects = IdentifyPrimaryObjectsBridge()
inputImage = Image(image=pic_array, mask=bin_array)
#watershed_boundaries, object_count, max_suppression = primaryObjects.separate_neighboring_objects(inputImage, labeled_array, len(labeled_array))

#primaryObjects.__dict__['unclump_method'] = UN_SHAPE
#$watershed_boundaries, object_count, max_suppression = primaryObjects.separate_neighboring_objects(inputImage, watershed_boundaries, len(watershed_boundaries))
watershed_boundaries, object_count, max_suppression = primaryObjects.separate_neighboring_objects(inputImage, labeled_array, len(labeled_array))


with open(inFile.replace('.tif', '_segmented.pkl'), 'wb') as segmentedFile:
	pickle.dump(watershed_boundaries, segmentedFile)

print("num_objects", object_count)
print(watershed_boundaries)






