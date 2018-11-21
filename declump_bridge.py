#USE PYTHON 2

import sys

import math
import centrosome.cpmorphology
import centrosome.outline
import centrosome.propagate
import centrosome.threshold
import numpy
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

class IdentifyPrimaryObjectsBridge(IdentifyPrimaryObjects):

	def __init__(self):

		super(IdentifyPrimaryObjects, self).__init__()
		self.advanced = True
		self.unclump_method = UN_SHAPE
		
