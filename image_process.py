import numpy as np 

import skimage
from skimage import io
from skimage import external
from skimage import morphology
import math


class Compartment:
	def __init__(self, left, right, top, bottom, image_values):
		self.values = []
		for i in range(top, bottom):
			for j in range(left, right):
				self.values.append(float(image_values[i][j]))
		self.coordinates = {'left': left, 'right': right, 'top': top, 'bottom': bottom}
		self.noise_compartment = False
		self.hasBorder = True

	def __str__(self):
		return self.coordinates

	def std(self):
		if self.values == []:
			print(self.coordinates)
			return 5
		return np.std(self.values, dtype=np.float64)

	def mean(self):
		return np.mean(self.values)

	def borderPercent(self, binary):
		border = 0
		for i in range(self.coordinates['top'], self.coordinates['bottom']):
			for j in range(self.coordinates['left'], self.coordinates['right']):
				if binary[i][j] == 0:
					border += 1
		return border / ((self.coordinates['right'] - self.coordinates['left']) * (self.coordinates['bottom'] - self.coordinates['top'])) 

	def set_noise_compartment(self, value):
		self.noise_compartment = bool(value)

	def avgBorder(self, binary):
		k, border = 0, []
		for i in range(self.coordinates['top'], self.coordinates['bottom']):
			for j in range(self.coordinates['left'], self.coordinates['right']):
				if binary[i][j] == 0:
					border.append(self.values[k])
				k += 1
		if border != []:
			return np.mean(border)

		self.hasBorder = False
		print(self.str())
		return None


class Compartmentalize:
	def __init__(self, image_values, size):
		self.compartments = []
		i_max, j_max, ki, kj = len(image_values), len(image_values[0]), 0, 0

		while ki < i_max:
			while kj < j_max:
				#print(ki, kj, '***')
				self.compartments.append(Compartment(kj, min(kj+size, j_max), ki, min(ki+size, i_max), image_values))
				kj += size
			kj = 0
			ki+= size
		self.size = size
		self.i_max = i_max
		self.j_max = j_max

	def getCompartment(self, i, j):
		return self.compartments[(j // self.size) + (i // self.size) * (self.j_max // self.size)]

	def getAvgBorder(self, binary):
		for c in self.compartments:
			k = c.borderPercent(binary)
			if k > 0.7:
				c.set_noise_compartment(True)

		not_Noise = filter(lambda c: not bool(c.noise_compartment), self.compartments)
		
		return np.mean([c.avgBorder(binary) for c in not_Noise if c.hasBorder])


class Noise:

	def __init__(self, image_values, regions=None, iterations=3, binary=False):
		if not binary:
			assert isinstance(regions, Compartmentalize), "Compartments are required for noise editing unless binary image"
			self.regions = regions
		self.image_values = image_values
		self.iterations = iterations
		self.binary = binary
		self.detectNoise = self.detectNoise_BIN if binary else self.detectNoise_OG

	def detectNoise_OG(self, i, j, neighbors):
		cut = regions.getCompartment(i, j).std() / 3
		num = len(neighbors)
		hits = 0
		for (y, x) in neighbors:
			if abs(float(self.image_values[i][j]) - float(self.image_values[y][x])) > cut:
				hits += 1
		return [hits > (num * 0.5), 0]

	def detectNoise_BIN(self, i, j, neighbors):
		num = len(neighbors)
		diff = 0
		for (y, x) in neighbors:
			if self.image_values[i][j] != self.image_values[y][x]:
				diff += 1
		return [diff > (num * 0.5), WHITE if self.image_values[i][j] == 0 else 0]
		#return [diff > (num * 0.5) and self.image_values[i][j] != 0, 0]

	def smooth(self):
		for i in range(len(self.image_values)):
			for j in range(len(self.image_values)):
				change, value = self.detectNoise(i, j, getNeighborIndices(self.image_values, i, j))
				if change:
					self.image_values[i][j] = value

	def reduce(self):
		for _ in range(self.iterations):
			self.smooth()


'''
class Cell:
	def __init__(self, pivot)
	#Max rdius
	#propagate'''

def getNeighborIndices(image_values, i, j):
	neighbors = []
	for ki in range(-1, 2):
		for kj in range(-1, 2):
			if i + ki >= 0 and j + kj >= 0 and i + ki < len(image_values) and j + kj < len(image_values[0]):
				neighbors.append((i+ki, j+kj))
	return neighbors


def basicEdge(pic_array, out_array, regions):
	for i in range(len(pic_array) - 1):
		for j in range(len(pic_array[0]) - 1):

			left, right = float(pic_array[i][j]), float(pic_array[i][j+1])
			bottom, diagonal = float(pic_array[i+1][j]), float(pic_array[i+1][j+1])

			''' May require revision '''
			# Make borders thinner! if cut change within 5 pixels ?
			# find average color of borders and find missing border pixels ... closing may be needed thereafter
			cut = regions.getCompartment(i, j).std() / 3  # 3 basic, 5 sensitive

			if abs(left - right) > cut:
				out_array[i][j] = 0
			elif abs(left - bottom) > cut:
				out_array[i][j] = 0
			elif abs(left - diagonal) > cut:
				out_array[i][j] = 0
			else:
				out_array[i][j] = WHITE

def enhanceEdges(pic_array, out_array, regions):
	borderMean = regions.getAvgBorder(out_array)
	tolerance = 0.1 * WHITE # +/- 10 % of total image range
	for i in range(len(out_array)):
		for j in range(len(out_array[0])):
			if borderMean - tolerance <= pic_array[i][j] and borderMean + tolerance >= pic_array[i][j]:
				out_array[i][j] = 0


cells = io.imread('/mnt/c/Users/Arjit/Documents/Kramer Lab/vit A/Cell_sizes/Cell Size Project/Vitamin A Free Diet in 3 RD1 mice/TRIAL/piece-0019.tif')

skimage.external.tifffile.imsave('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIALPY.tif', cells)

with skimage.external.tifffile.TiffFile('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIALPY.tif') as pic:

	pic_array = pic.asarray();
	out_array = pic.asarray(); #copy dimensions
	WHITE = skimage.dtype_limits(pic_array, True)[1]
	#pic_array = [[float(x) for x in row] for row in pic_array]
	regions = Compartmentalize(pic_array, 32)

	basicEdge(pic_array, out_array, regions) # out_array is modified

	skimage.external.tifffile.imsave('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIAL_PROCESSED.tif', out_array)

	enhanceEdges(pic_array, out_array, regions)
	skimage.external.tifffile.imsave('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIAL_Edge.tif', out_array)


	noise_handler = Noise(out_array, iterations=3, binary=True)
	noise_handler.reduce()

	skimage.external.tifffile.imsave('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIAL_NOISE_reversefunc3_s.tif', out_array)


	#skimage.external.tifffile.imsave('/mnt/c/Users/Arjit/Documents/Kramer Lab/TRIAL_skeleton.tif', skimage.morphology.opening(pic_array))
#out_regions = Compartmentalize(out_array, 16)






